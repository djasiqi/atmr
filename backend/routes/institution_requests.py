# routes/institution_requests.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Routes pour la gestion des demandes de transport institutionnelles.

Endpoints:
- POST /api/v1/institutions/requests - Créer une demande
- GET /api/v1/institutions/requests - Lister les demandes
- GET /api/v1/institutions/requests/{id} - Détail demande
- PUT /api/v1/institutions/requests/{id} - Modifier demande
- POST /api/v1/institutions/requests/{id}/send - Envoyer aux transporteurs
- POST /api/v1/institutions/requests/{id}/external-carrier - Affecter transporteur externe
- POST /api/v1/institutions/requests/{id}/external-completion - Déclarer mission externe réalisée
- POST /api/v1/institutions/requests/{id}/cancel - Annuler demande
"""

import logging
from datetime import UTC, date, datetime
from typing import Any, cast

import sentry_sdk
from flask import g, request
from flask_jwt_extended import get_jwt
from flask_restx import Namespace, Resource, fields
from marshmallow import ValidationError

from ext import db
from models import InstitutionPatient, TransportRequest
from models.enums import InstitutionRole, MissionType, RequestStatus
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from schemas.institution_schemas import (
    AssignExternalCarrierSchema,
    CompleteExternalMissionSchema,
    TransportRequestCreateSchema,
    TransportRequestQuerySchema,
    TransportRequestUpdateSchema,
)
from security.api_key_auth import api_key_or_jwt_required
from security.audit_log import AuditLogger
from security.authorization import AuthorizationService, get_user_team_ids
from shared.error_handlers import APIErrorHandler
from shared.time_utils import normalize_mission_wall_clock

logger = logging.getLogger(__name__)

# Règle d'architecture : écritures mission via normalize_mission_wall_clock() uniquement.
# parse_iso8601() interdit pour les écritures (aware) — validation/comparaison seulement.


def _apply_return_fields(transport_req: TransportRequest, validated: dict[str, Any]) -> None:
    """Persiste return_date, return_time et return_time_confirmed sur une demande."""
    if validated.get("return_time"):
        transport_req.return_time = normalize_mission_wall_clock(validated["return_time"])
        explicit = validated.get("return_time_confirmed")
        transport_req.return_time_confirmed = (
            bool(explicit) if explicit is not None else True
        )
        if transport_req.return_time is not None:
            transport_req.return_date = transport_req.return_time.date()
    elif validated.get("return_date"):
        try:
            transport_req.return_date = date.fromisoformat(str(validated["return_date"]))
        except ValueError:
            transport_req.return_date = None
        transport_req.return_time = None
        explicit = validated.get("return_time_confirmed")
        transport_req.return_time_confirmed = (
            bool(explicit) if explicit is not None else False
        )
    elif "return_time" in validated or "return_date" in validated:
        transport_req.return_time = None
        transport_req.return_date = None
        if "return_time_confirmed" in validated:
            transport_req.return_time_confirmed = bool(
                validated.get("return_time_confirmed")
            )


def _derive_multi_stop_return_fields(
    transport_req: TransportRequest, scheduled_time: datetime | None
) -> None:
    """Multi-stop + retour institution : synchronise depuis mission_date."""
    mission_day = getattr(transport_req, "mission_date", None)
    if mission_day is not None:
        transport_req.return_date = mission_day
    elif scheduled_time is not None:
        transport_req.return_date = scheduled_time.date()
    transport_req.return_time = None
    transport_req.return_time_confirmed = False


def _return_leg_schedule(validated: dict[str, Any]) -> tuple[Any, bool]:
    """Extrait horaire retour institution depuis le payload."""
    raw = validated.get("return_scheduled_time") or validated.get("return_time")
    confirmed = validated.get("return_time_confirmed")
    if confirmed is None:
        confirmed = bool(raw)
    return raw, bool(confirmed)


_CARRIER_IMPACT_FIELDS = frozenset(
    {
        "pickup_location",
        "dropoff_location",
        "dropoff_establishment",
        "dropoff_service",
        "dropoff_doctor",
        "intermediate_stops",
        "multi_stop",
        "return_to_institution",
        "scheduled_time",
        "scheduled_time_type",
        "mission_date",
        "pickup_time_confirmed",
        "appointment_time_confirmed",
        "return_scheduled_time",
        "return_time",
        "return_date",
        "return_time_confirmed",
        "mobility",
        "is_round_trip",
    }
)


def _requires_carrier_acknowledgement(
    transport_req: TransportRequest, validated: dict[str, Any]
) -> bool:
    if transport_req.status not in (
        RequestStatus.SENT.value,
        RequestStatus.ACCEPTED.value,
    ):
        return False
    return bool(set(validated.keys()) & _CARRIER_IMPACT_FIELDS)


def _notify_companies_request_updated(
    transport_req: TransportRequest,
    *,
    updated_fields: list[str],
) -> None:
    """Informe les entreprises avec offre PENDING qu'une demande envoyée a changé."""
    from models import OfferStatus, RequestOffer
    from services.events.institution_events import persist_company_notification

    pending_offers = RequestOffer.query.filter_by(
        transport_request_id=transport_req.id,
        status=OfferStatus.PENDING.value,
    ).all()
    if not pending_offers:
        return

    institution = transport_req.institution
    inst_name = institution.name if institution else "Institution"
    patient = transport_req.patient
    patient_name = (
        f"{patient.first_name} {patient.last_name}".strip() if patient else ""
    )
    sched = transport_req.scheduled_time
    time_str = sched.strftime("%d.%m.%Y %H:%M") if sched else ""
    message = f"{inst_name} — {patient_name} — parcours modifié"
    if time_str:
        message = f"{message} — {time_str}"

    # Jour de la demande (pour préselectionner le filtre date côté entreprise)
    mission_date_iso = None
    if transport_req.mission_date is not None:
        mission_date_iso = transport_req.mission_date.isoformat()
    elif sched is not None:
        mission_date_iso = sched.date().isoformat()

    for offer in pending_offers:
        metadata = {
            "request_id": transport_req.id,
            "public_id": str(transport_req.public_id),
            "offer_id": offer.id,
            "institution_name": inst_name,
            "updated_fields": updated_fields,
        }
        if mission_date_iso:
            metadata["mission_date"] = mission_date_iso
        if transport_req.booking_id:
            metadata["booking_id"] = transport_req.booking_id
        revision = int(getattr(transport_req, "revision", None) or 1)
        metadata["revision"] = revision

        try:
            notif = persist_company_notification(
                company_id=offer.company_id,
                event_type="request_updated",
                title="Demande modifiée par l'institution",
                message=message.strip(" —"),
                metadata=metadata,
                dedupe_key=(
                    f"request_updated:{transport_req.id}:{offer.company_id}:{revision}"
                ),
            )
            if notif is None:
                continue
            try:
                from services.notifications.institution_new_request_push import (
                    enqueue_institution_company_push_message,
                )
                from services.notifications.push_message_builder import (
                    build_push_for_institution_request_updated,
                )

                push_msg = build_push_for_institution_request_updated(
                    transport_request=transport_req,
                    offer_id=offer.id,
                    company_id=offer.company_id,
                    institution_name=inst_name,
                    patient_name=patient_name,
                    title="Demande modifiée par l'institution",
                    message=message.strip(" —"),
                    dedupe_key=(
                        f"request_updated:{transport_req.id}:{offer.company_id}:{revision}"
                    ),
                    revision=revision,
                    mission_date_iso=mission_date_iso,
                )
                enqueue_institution_company_push_message(
                    company_id=offer.company_id,
                    msg=push_msg,
                )
            except Exception as push_err:
                logger.warning(
                    "[TransportRequests] Push update company=%s request=%s: %s",
                    offer.company_id,
                    transport_req.id,
                    push_err,
                )
        except Exception as notify_err:
            logger.warning(
                "[TransportRequests] Notification update company=%s request=%s: %s",
                offer.company_id,
                transport_req.id,
                notify_err,
            )


def _record_request_updated_timeline(
    transport_req: TransportRequest,
    *,
    updated_fields: list[str],
    user_id: int | None,
    carrier_notified: bool,
) -> None:
    from services.institutions.transport_timeline_service import (
        TimelineActor,
        record_event,
        resolve_actor_name,
    )

    record_event(
        "field_updated",
        institution_id=transport_req.institution_id,
        transport_request_id=transport_req.id,
        actor=TimelineActor(actor_type="institution_user", actor_user_id=user_id),
        payload={
            "changed_fields": updated_fields,
            "carrier_notified": carrier_notified,
            "actor_name": resolve_actor_name(user_id),
            "after_send": transport_req.status
            in (RequestStatus.SENT.value, RequestStatus.ACCEPTED.value),
        },
        correlation_id=f"request_field_updated:{transport_req.id}:{int(datetime.now(UTC).timestamp())}",
    )


def _persist_legs_from_validated(
    transport_req: TransportRequest, validated: dict[str, Any]
) -> None:
    """Construit et persiste les legs (multi-stop ou trajet simple legacy RDV)."""
    from services.institutions.mission_schedule import legacy_arrival_schedule
    from services.institutions.transport_request_legs_service import (
        build_legs_chain,
        build_simple_trip_leg,
        is_multi_stop_enabled,
        new_route_group_id,
        persist_legs,
        return_stop_from_validated,
        stops_from_validated,
        sync_return_fields_from_legs,
    )

    return_raw, return_confirmed = _return_leg_schedule(validated)

    if validated.get("multi_stop"):
        if not is_multi_stop_enabled():
            raise PermissionError("Parcours multi-étapes non activé sur ce serveur.")

        transport_req.multi_stop = True
        transport_req.return_to_institution = validated.get(
            "return_to_institution", True
        )
        transport_req.route_group_id = new_route_group_id()
        transport_req.is_round_trip = False
        transport_req.return_time = None

        stops = stops_from_validated(validated)
        return_stop = None
        if transport_req.return_to_institution:
            return_stop = return_stop_from_validated(
                validated,
                return_location=validated["pickup_location"],
                return_lat=validated.get("pickup_lat"),
                return_lng=validated.get("pickup_lng"),
                return_scheduled_time=return_raw,
                return_time_confirmed=return_confirmed,
            )
        legs_data = build_legs_chain(
            origin_location=validated["pickup_location"],
            origin_lat=validated.get("pickup_lat"),
            origin_lng=validated.get("pickup_lng"),
            stops=stops,
            return_to_institution=transport_req.return_to_institution,
            institution_return_location=validated["pickup_location"],
            institution_return_lat=validated.get("pickup_lat"),
            institution_return_lng=validated.get("pickup_lng"),
            return_scheduled_time=return_raw,
            return_time_confirmed=return_confirmed,
            return_stop=return_stop,
        )
        persist_legs(transport_req.id, legs_data)

        if transport_req.return_to_institution:
            sync_return_fields_from_legs(transport_req)
        else:
            transport_req.return_date = None
            transport_req.return_time_confirmed = False

        if legs_data:
            first = legs_data[0]
            transport_req.dropoff_location = first["dropoff_location"]
            transport_req.dropoff_lat = first.get("dropoff_lat")
            transport_req.dropoff_lng = first.get("dropoff_lng")
        return

    arrival, arrival_confirmed = legacy_arrival_schedule(validated)
    if arrival is not None or (
        validated.get("scheduled_time_type") == "arrival"
        and validated.get("scheduled_time")
    ):
        dropoff = (validated.get("dropoff_location") or "").strip()
        if dropoff:
            appt_time = arrival
            if appt_time is None and validated.get("scheduled_time"):
                appt_time = normalize_mission_wall_clock(validated["scheduled_time"])
            legs_data = build_simple_trip_leg(
                pickup_location=validated["pickup_location"],
                pickup_lat=validated.get("pickup_lat"),
                pickup_lng=validated.get("pickup_lng"),
                dropoff_location=dropoff,
                dropoff_lat=validated.get("dropoff_lat"),
                dropoff_lng=validated.get("dropoff_lng"),
                appointment_time=appt_time,
                time_confirmed=arrival_confirmed,
                dropoff_establishment=validated.get("dropoff_establishment"),
                dropoff_service=validated.get("dropoff_service"),
                dropoff_doctor=validated.get("dropoff_doctor"),
            )
            if legs_data:
                persist_legs(transport_req.id, legs_data)

# Namespace
institution_requests_ns = Namespace(
    "institution_requests",
    description="Gestion des demandes de transport institutionnelles",
)

# Modèles Swagger
api_error_model = create_api_error_model(institution_requests_ns)
not_found_error_model = create_not_found_error_model(institution_requests_ns)
permission_error_model = create_permission_error_model(institution_requests_ns)
validation_error_model = create_validation_error_model(institution_requests_ns)

# Schemas
request_create_schema = TransportRequestCreateSchema()
request_update_schema = TransportRequestUpdateSchema()
request_query_schema = TransportRequestQuerySchema()
assign_external_carrier_schema = AssignExternalCarrierSchema()
complete_external_mission_schema = CompleteExternalMissionSchema()

# Modèle de réponse demande
transport_request_model = institution_requests_ns.model(
    "TransportRequest",
    {
        "id": fields.Integer(description="ID interne"),
        "public_id": fields.String(description="ID public UUID"),
        "external_reference": fields.String(description="Référence externe DPI"),
        "patient_id": fields.Integer(description="ID patient"),
        "patient": fields.Raw(description="Détails patient"),
        "mission_type": fields.String(description="Type de mission"),
        "delivery_description": fields.String(description="Description livraison"),
        "scheduled_time": fields.String(description="Date/heure prévue"),
        "scheduled_time_type": fields.String(
            description="Type d'horaire: departure (départ) ou arrival (rendez-vous)",
            enum=["departure", "arrival"],
        ),
        "pickup_location": fields.String(description="Adresse départ"),
        "dropoff_location": fields.String(description="Adresse arrivée"),
        "is_round_trip": fields.Boolean(description="Aller-retour"),
        "mobility": fields.Raw(description="Infos mobilité"),
        "contact_on_site": fields.Raw(description="Contact sur site"),
        "billing_intent": fields.String(description="Intention facturation"),
        "status": fields.String(description="Statut"),
        "is_editable": fields.Boolean(description="Modifiable"),
        "is_cancellable": fields.Boolean(description="Annulable"),
        "created_at": fields.String(description="Date création"),
        "sent_at": fields.String(description="Date envoi"),
        "cancelled_at": fields.String(description="Date annulation"),
    },
)

request_list_model = institution_requests_ns.model(
    "TransportRequestList",
    {
        "requests": fields.List(fields.Nested(transport_request_model)),
        "total": fields.Integer(description="Nombre total de résultats"),
        "page": fields.Integer(description="Page courante"),
        "per_page": fields.Integer(description="Résultats par page"),
        "pages": fields.Integer(description="Nombre total de pages"),
    },
)


def get_institution_context():
    """Récupère le contexte institution pour actions d'écriture (JWT ou API Key).

    Rôles autorisés: admin, requester, billing, curator.

    Returns:
        Tuple (institution_id, user_id_or_none)
    """
    # Si authentifié par API Key
    if hasattr(g, "institution_id") and g.get("auth_method") == "api_key":
        return g.institution_id, None

    # Sinon JWT
    institution, user = AuthorizationService.require_institution_role(
        InstitutionRole.ADMIN.value,
        InstitutionRole.REQUESTER.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.CURATOR.value,
    )
    return institution.id, user.id


def get_institution_read_context():
    """Récupère le contexte institution pour lecture seule (JWT ou API Key).

    Rôles autorisés: admin, requester, billing, reader, curator, reception.

    Returns:
        Tuple (institution_id, user_id_or_none, role_or_none)
    """
    # Si authentifié par API Key
    if hasattr(g, "institution_id") and g.get("auth_method") == "api_key":
        return g.institution_id, None, None

    # Sinon JWT — tous les rôles institution peuvent lire
    institution, user = AuthorizationService.require_institution_role(
        InstitutionRole.ADMIN.value,
        InstitutionRole.REQUESTER.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.READER.value,
        InstitutionRole.CURATOR.value,
        InstitutionRole.RECEPTION.value,
    )
    return institution.id, user.id, user.institution_role


def resolve_patient(institution_id: int, data: dict[str, Any]) -> int | None:
    """Résout l'ID patient depuis patient_id ou patient_external_reference.

    Args:
        institution_id: ID de l'institution
        data: Données de la requête

    Returns:
        patient_id ou None

    Raises:
        ValueError si patient non trouvé
    """
    patient_id = data.get("patient_id")
    patient_ext_ref = data.get("patient_external_reference")

    if patient_id:
        # Vérifier que le patient appartient à l'institution
        patient = InstitutionPatient.query.filter_by(
            id=patient_id,
            institution_id=institution_id,
        ).first()
        if not patient:
            raise ValueError(
                f"Patient ID {patient_id} non trouvé dans cette institution"
            )
        return patient.id

    if patient_ext_ref:
        patient = InstitutionPatient.find_by_external_reference(
            institution_id, patient_ext_ref
        )
        if not patient:
            raise ValueError(
                f"Patient avec external_reference '{patient_ext_ref}' non trouvé"
            )
        return patient.id

    return None


@institution_requests_ns.route("")
class TransportRequestList(Resource):
    """Endpoints pour lister et créer des demandes."""

    @institution_requests_ns.doc(
        description="Liste les demandes de transport de l'institution.",
        params={
            "status": "Filtre par statut (DRAFT, SENT, CANCELLED, etc.)",
            "external_reference": "Filtre par référence externe",
            "patient_id": "Filtre par patient",
            "date_from": "Date début (YYYY-MM-DD)",
            "date_to": "Date fin (YYYY-MM-DD)",
            "page": "Numéro de page (défaut: 1)",
            "per_page": "Résultats par page (défaut: 20, max: 100)",
        },
    )
    @institution_requests_ns.response(200, "Succès", request_list_model)
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @api_key_or_jwt_required(scopes=["requests:read"])
    def get(self):
        """Liste les demandes de transport.

        Auth: JWT (tous rôles institution) ou API Key (scope requests:read)
        """
        try:
            institution_id, _, _ = get_institution_read_context()

            # Valider query params
            try:
                params = cast(dict[str, Any], request_query_schema.load(request.args))
            except ValidationError as err:
                return {"error": "Paramètres invalides", "details": err.messages}, 400

            # Base query
            query = TransportRequest.query.filter_by(institution_id=institution_id)

            # Filtrage curator : si des équipes sont assignées, ne voir que les requests
            # liées aux patients de ses équipes (ou patients sans équipe).
            # Si aucune équipe n'est assignée → pas de filtre (mode bootstrapping).
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if institution_role == InstitutionRole.CURATOR.value:
                user = AuthorizationService.require_user()
                team_ids = get_user_team_ids(user.id)
                if team_ids:
                    patient_ids_subq = (
                        db.session.query(InstitutionPatient.id)
                        .filter(
                            InstitutionPatient.institution_id == institution_id,
                            db.or_(
                                InstitutionPatient.curator_team_id.in_(team_ids),
                                InstitutionPatient.curator_team_id.is_(None),
                            ),
                        )
                        .subquery()
                    )
                    query = query.filter(
                        db.or_(
                            TransportRequest.patient_id.in_(
                                db.session.query(patient_ids_subq.c.id)
                            ),
                            TransportRequest.patient_id.is_(None),
                        )
                    )

            # Filtres
            if params.get("status"):
                query = query.filter_by(status=params["status"])

            if params.get("carrier_source"):
                query = query.filter_by(carrier_source=params["carrier_source"])

            if params.get("external_reference"):
                query = query.filter_by(external_reference=params["external_reference"])

            if params.get("patient_id"):
                query = query.filter_by(patient_id=params["patient_id"])

            if params.get("date_from"):
                date_from = datetime.strptime(params["date_from"], "%Y-%m-%d").date()
                query = query.filter(TransportRequest.mission_date >= date_from)

            if params.get("date_to"):
                date_to = datetime.strptime(params["date_to"], "%Y-%m-%d").date()
                query = query.filter(TransportRequest.mission_date <= date_to)

            # Pagination
            page = params.get("page", 1)
            per_page = params.get("per_page", 20)
            total = query.count()
            pages = (total + per_page - 1) // per_page

            requests = (
                query.order_by(
                    TransportRequest.mission_date.desc(),
                    TransportRequest.id.desc(),
                )
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all()
            )

            return {
                "requests": [r.serialize for r in requests],
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": pages,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[TransportRequests] GET error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

    @institution_requests_ns.doc(
        description="Crée une nouvelle demande de transport.",
    )
    @institution_requests_ns.response(201, "Demande créée", transport_request_model)
    @institution_requests_ns.response(400, "Données invalides", validation_error_model)
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(
        409, "Référence externe déjà utilisée", api_error_model
    )
    @api_key_or_jwt_required(scopes=["requests:write"])
    def post(self):
        """Crée une nouvelle demande de transport.

        Auth: JWT (institution_admin/requester) ou API Key (scope requests:write)

        Idempotence: Si external_reference est fournie et existe déjà, retourne 409.
        """
        try:
            institution_id, user_id = get_institution_context()

            data = request.get_json() or {}

            # Valider
            try:
                validated = cast(dict[str, Any], request_create_schema.load(data))
            except ValidationError as err:
                return {"error": "Données invalides", "details": err.messages}, 400

            # Vérifier unicité external_reference (si fournie)
            ext_ref_raw = validated.get("external_reference")
            ext_ref = ext_ref_raw.strip() if isinstance(ext_ref_raw, str) else None
            if ext_ref:
                existing = TransportRequest.find_by_external_reference(
                    institution_id, ext_ref
                )
                if existing:
                    return {
                        "error": f"Demande avec external_reference '{ext_ref}' existe déjà",
                        "existing_request_id": existing.id,
                        "existing_request_public_id": existing.public_id,
                        "existing_request_status": existing.status,
                    }, 409

            # Résoudre patient
            try:
                patient_id = resolve_patient(institution_id, validated)
            except ValueError as err:
                return {"error": str(err)}, 400

            transport_req = TransportRequest()
            transport_req.institution_id = institution_id
            transport_req.created_by_user_id = user_id
            transport_req.external_reference = ext_ref
            transport_req.patient_id = patient_id
            transport_req.mission_type = validated.get(
                "mission_type", MissionType.PATIENT_TRANSPORT.value
            )
            transport_req.delivery_description = validated.get("delivery_description")

            from services.institutions.mission_schedule import apply_departure_schedule

            try:
                apply_departure_schedule(transport_req, validated)
            except ValueError as sched_err:
                return {"error": str(sched_err)}, 400

            # Lieux
            transport_req.pickup_location = validated["pickup_location"]
            transport_req.pickup_lat = validated.get("pickup_lat")
            transport_req.pickup_lng = validated.get("pickup_lng")
            transport_req.pickup_floor = validated.get("pickup_floor")
            transport_req.pickup_door_code = validated.get("pickup_door_code")

            transport_req.dropoff_location = validated["dropoff_location"]
            transport_req.dropoff_lat = validated.get("dropoff_lat")
            transport_req.dropoff_lng = validated.get("dropoff_lng")
            transport_req.dropoff_floor = validated.get("dropoff_floor")
            transport_req.dropoff_door_code = validated.get("dropoff_door_code")

            # Type de lieu
            transport_req.pickup_type = validated.get("pickup_type")
            transport_req.dropoff_type = validated.get("dropoff_type")
            transport_req.pickup_entry_point = validated.get("pickup_entry_point")
            transport_req.dropoff_entry_point = validated.get("dropoff_entry_point")

            # Options
            transport_req.is_round_trip = validated.get("is_round_trip", False)
            transport_req.is_urgent = bool(validated.get("is_urgent", False))
            if transport_req.is_round_trip:
                _apply_return_fields(transport_req, validated)
            logger.info(
                "[CreateRequest] is_round_trip=%s, return_date=%r, return_time=%r, return_time_confirmed=%s",
                transport_req.is_round_trip,
                transport_req.return_date,
                transport_req.return_time,
                transport_req.return_time_confirmed,
            )

            # Mobilité
            transport_req.mobility = validated.get("mobility")

            # Accès
            transport_req.floor_elevator_info = validated.get("floor_elevator_info")

            # Contact
            transport_req.contact_on_site = validated.get("contact_on_site")

            # Notes
            transport_req.notes = validated.get("notes")

            # Facturation
            transport_req.billing_intent = validated.get("billing_intent", "patient")
            transport_req.billing_details = validated.get("billing_details")

            # Statut initial
            transport_req.status = RequestStatus.DRAFT.value

            db.session.add(transport_req)
            db.session.flush()

            try:
                _persist_legs_from_validated(transport_req, validated)
            except PermissionError as perm_err:
                db.session.rollback()
                return {"error": str(perm_err)}, 403
            except ValueError as leg_err:
                db.session.rollback()
                return {"error": str(leg_err)}, 400

            # Timeline transport: request_created
            try:
                from services.institutions.transport_timeline_service import (
                    TimelineActor,
                    record_event,
                )

                record_event(
                    "request_created",
                    institution_id=institution_id,
                    transport_request_id=transport_req.id,
                    actor=TimelineActor(
                        actor_type="institution_user" if user_id else "api_key",
                        actor_user_id=user_id,
                    ),
                    correlation_id=f"request_created:{transport_req.id}",
                )
            except Exception as timeline_err:
                logger.warning(
                    "[TransportRequests] Timeline recording failed: %s", timeline_err
                )

            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="request_created",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution" if user_id else "api_key",
                    institution_id=institution_id,
                    result_status="success",
                    action_details={
                        "request_id": transport_req.id,
                        "external_reference": ext_ref,
                        "mission_type": transport_req.mission_type,
                        "patient_id": patient_id,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[TransportRequests] Audit log error: %s", audit_err)

            logger.info(
                "[TransportRequests] Demande créée: id=%s, ext_ref=%s, institution=%s",
                transport_req.id,
                ext_ref,
                institution_id,
            )

            return transport_req.serialize, 201

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[TransportRequests] POST error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@institution_requests_ns.route("/<int:request_id>")
class TransportRequestDetail(Resource):
    """Endpoints pour détail et modification d'une demande."""

    @institution_requests_ns.doc(
        description="Récupère les détails d'une demande.",
    )
    @institution_requests_ns.response(200, "Succès", transport_request_model)
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(404, "Demande non trouvée", not_found_error_model)
    @api_key_or_jwt_required(scopes=["requests:read"])
    def get(self, request_id: int):
        """Récupère les détails d'une demande.

        Auth: JWT (tous rôles institution) ou API Key (scope requests:read)
        """
        try:
            institution_id, _, _ = get_institution_read_context()

            transport_req = TransportRequest.query.filter_by(
                id=request_id,
                institution_id=institution_id,
            ).first()

            if not transport_req:
                return {"error": "Demande non trouvée"}, 404

            # Vérifier accès curator : si le curateur a des équipes,
            # autoriser accès si le patient est dans son équipe ou sans équipe
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if (
                institution_role == InstitutionRole.CURATOR.value
                and transport_req.patient_id
            ):
                user = AuthorizationService.require_user()
                team_ids = get_user_team_ids(user.id)
                if team_ids:
                    patient = InstitutionPatient.query.get(transport_req.patient_id)
                    if (
                        patient
                        and patient.curator_team_id
                        and patient.curator_team_id not in team_ids
                    ):
                        return {"error": "Demande non trouvée"}, 404

            from services.institutions.booking_change_service import (
                mask_financial_fields,
            )

            role = None
            if getattr(g, "auth_method", None) == "api_key":
                role = InstitutionRole.ADMIN.value
            else:
                role = get_jwt().get("institution_role")
            return mask_financial_fields(transport_req.serialize, role), 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[TransportRequests] GET /%s error: %s", request_id, e)
            return APIErrorHandler.handle_exception(e, logger)

    @institution_requests_ns.doc(
        description="Modifie une demande de transport.",
    )
    @institution_requests_ns.response(200, "Demande modifiée", transport_request_model)
    @institution_requests_ns.response(
        400, "Données invalides ou demande non modifiable", validation_error_model
    )
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(404, "Demande non trouvée", not_found_error_model)
    @api_key_or_jwt_required(scopes=["requests:write"])
    def put(self, request_id: int):
        """Modifie une demande de transport.

        Auth: JWT ou API Key (scope requests:write)

        Seules les demandes DRAFT ou SENT peuvent être modifiées.
        """
        try:
            institution_id, user_id = get_institution_context()

            transport_req = TransportRequest.query.filter_by(
                id=request_id,
                institution_id=institution_id,
            ).first()

            if not transport_req:
                return {"error": "Demande non trouvée"}, 404

            if not transport_req.is_editable:
                return {
                    "error": f"Demande non modifiable (statut: {transport_req.status})",
                    "allowed_statuses": [
                        s.value for s in RequestStatus.editable_statuses()
                    ],
                }, 400

            data = request.get_json() or {}

            # Valider
            try:
                validated = cast(dict[str, Any], request_update_schema.load(data))
            except ValidationError as err:
                return {"error": "Données invalides", "details": err.messages}, 400

            carrier_ack_required = _requires_carrier_acknowledgement(
                transport_req, validated
            )
            if carrier_ack_required and not validated.get("acknowledge_carrier_impact"):
                return {
                    "error": (
                        "Confirmation requise : cette modification impacte "
                        "les transporteurs consultés."
                    ),
                    "code": "carrier_ack_required",
                }, 400

            operational_fields = [
                k for k in validated.keys() if k != "acknowledge_carrier_impact"
            ]

            # Résoudre patient si changé
            if "patient_id" in validated or "patient_external_reference" in validated:
                try:
                    transport_req.patient_id = resolve_patient(
                        institution_id, validated
                    )
                except ValueError as err:
                    return {"error": str(err)}, 400

            # Appliquer modifications
            if "mission_type" in validated:
                transport_req.mission_type = validated["mission_type"]
            if "delivery_description" in validated:
                transport_req.delivery_description = validated["delivery_description"]

            if any(
                k in validated
                for k in (
                    "mission_date",
                    "scheduled_time",
                    "scheduled_time_type",
                    "pickup_time_confirmed",
                )
            ):
                from services.institutions.mission_schedule import (
                    apply_departure_schedule,
                )

                merged = {
                    "mission_date": transport_req.mission_date,
                    "scheduled_time": (
                        transport_req.scheduled_time.isoformat()
                        if transport_req.scheduled_time
                        else None
                    ),
                    "scheduled_time_type": transport_req.scheduled_time_type,
                    "pickup_time_confirmed": transport_req.pickup_time_confirmed,
                }
                merged.update(validated)
                try:
                    apply_departure_schedule(transport_req, merged)
                except ValueError as sched_err:
                    db.session.rollback()
                    return {"error": str(sched_err)}, 400

            # Lieux + types de lieu + points d'accueil
            for field in [
                "pickup_location",
                "pickup_lat",
                "pickup_lng",
                "pickup_floor",
                "pickup_door_code",
                "pickup_type",
                "pickup_entry_point",
                "dropoff_location",
                "dropoff_lat",
                "dropoff_lng",
                "dropoff_floor",
                "dropoff_door_code",
                "dropoff_type",
                "dropoff_entry_point",
            ]:
                if field in validated:
                    setattr(transport_req, field, validated[field])

            # Options
            if "is_round_trip" in validated:
                transport_req.is_round_trip = validated["is_round_trip"]
            if "is_urgent" in validated:
                transport_req.is_urgent = bool(validated["is_urgent"])
            if any(
                k in validated
                for k in ("return_time", "return_date", "return_time_confirmed")
            ):
                _apply_return_fields(transport_req, validated)

            # Mobilité, contact, notes
            if "mobility" in validated:
                transport_req.mobility = validated["mobility"]
            if "floor_elevator_info" in validated:
                transport_req.floor_elevator_info = validated["floor_elevator_info"]
            if "contact_on_site" in validated:
                transport_req.contact_on_site = validated["contact_on_site"]
            if "notes" in validated:
                transport_req.notes = validated["notes"]

            # Facturation
            if "billing_intent" in validated:
                transport_req.billing_intent = validated["billing_intent"]
            if "billing_details" in validated:
                transport_req.billing_details = validated["billing_details"]

            # Multi-stop : création ou réorganisation legs (DRAFT/SENT)
            if "intermediate_stops" in validated and (
                getattr(transport_req, "multi_stop", False)
                or validated.get("multi_stop")
            ):
                from services.institutions.transport_request_legs_service import (
                    is_multi_stop_enabled,
                    new_route_group_id,
                    reorganize_multi_stop_legs,
                    stops_from_validated,
                )

                if not is_multi_stop_enabled():
                    db.session.rollback()
                    return {
                        "error": "Parcours multi-étapes non activé sur ce serveur."
                    }, 403

                if validated.get("multi_stop") and not getattr(
                    transport_req, "multi_stop", False
                ):
                    transport_req.multi_stop = True
                    transport_req.is_round_trip = False
                    transport_req.return_time = None
                    if not getattr(transport_req, "route_group_id", None):
                        transport_req.route_group_id = new_route_group_id()

                return_to_inst = validated.get(
                    "return_to_institution",
                    getattr(transport_req, "return_to_institution", False),
                )
                stops = stops_from_validated(validated)
                if not stops:
                    db.session.rollback()
                    return {
                        "error": "Au moins une étape intermédiaire requise."
                    }, 400

                return_raw, return_confirmed = _return_leg_schedule(validated)
                reorganize_multi_stop_legs(
                    transport_req,
                    intermediate_stops=stops,
                    return_to_institution=bool(return_to_inst),
                    return_scheduled_time=return_raw,
                    return_time_confirmed=return_confirmed,
                    actor_user_id=user_id,
                )
            elif "return_to_institution" in validated and getattr(
                transport_req, "multi_stop", False
            ):
                transport_req.return_to_institution = validated["return_to_institution"]

            # Trajet simple : resynchroniser le leg unique (RDV destination)
            elif not getattr(transport_req, "multi_stop", False) and any(
                k in validated
                for k in (
                    "pickup_location",
                    "dropoff_location",
                    "dropoff_establishment",
                    "dropoff_service",
                    "dropoff_doctor",
                    "scheduled_time",
                    "scheduled_time_type",
                    "mission_date",
                    "pickup_time_confirmed",
                    "appointment_time_confirmed",
                )
            ):
                merged: dict[str, Any] = {
                    "pickup_location": transport_req.pickup_location,
                    "pickup_lat": transport_req.pickup_lat,
                    "pickup_lng": transport_req.pickup_lng,
                    "dropoff_location": transport_req.dropoff_location,
                    "dropoff_lat": transport_req.dropoff_lat,
                    "dropoff_lng": transport_req.dropoff_lng,
                    "scheduled_time": (
                        transport_req.scheduled_time.isoformat()
                        if transport_req.scheduled_time
                        else None
                    ),
                    "scheduled_time_type": transport_req.scheduled_time_type,
                    "pickup_time_confirmed": transport_req.pickup_time_confirmed,
                    "appointment_time_confirmed": getattr(
                        transport_req, "appointment_time_confirmed", None
                    ),
                    "mission_date": transport_req.mission_date,
                }
                merged.update(validated)
                try:
                    _persist_legs_from_validated(transport_req, merged)
                except PermissionError as perm_err:
                    db.session.rollback()
                    return {"error": str(perm_err)}, 403
                except ValueError as val_err:
                    db.session.rollback()
                    return {"error": str(val_err)}, 400

            db.session.commit()

            if carrier_ack_required:
                current_revision = int(getattr(transport_req, "revision", None) or 1)
                transport_req.revision = current_revision + 1
                db.session.commit()
                _notify_companies_request_updated(
                    transport_req,
                    updated_fields=operational_fields,
                )
                _record_request_updated_timeline(
                    transport_req,
                    updated_fields=operational_fields,
                    user_id=user_id,
                    carrier_notified=True,
                )

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="request_updated",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution" if user_id else "api_key",
                    institution_id=institution_id,
                    result_status="success",
                    action_details={
                        "request_id": transport_req.id,
                        "updated_fields": operational_fields,
                        "carrier_notified": carrier_ack_required,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[TransportRequests] Audit log error: %s", audit_err)

            logger.info("[TransportRequests] Demande modifiée: id=%s", transport_req.id)

            return transport_req.serialize, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[TransportRequests] PUT /%s error: %s", request_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@institution_requests_ns.route("/<int:request_id>/send")
class TransportRequestSend(Resource):
    """Endpoint pour envoyer une demande aux transporteurs."""

    @institution_requests_ns.doc(
        description="Envoie la demande aux transporteurs (crée des offres).",
    )
    @institution_requests_ns.response(200, "Demande envoyée", transport_request_model)
    @institution_requests_ns.response(
        400, "Demande non envoyable", validation_error_model
    )
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(404, "Demande non trouvée", not_found_error_model)
    @institution_requests_ns.response(409, "Offres déjà en attente", api_error_model)
    @api_key_or_jwt_required(scopes=["requests:write"])
    def post(self, request_id: int):
        """Envoie la demande aux transporteurs.

        Auth: JWT ou API Key (scope requests:write)

        ÉTAPE 4: Crée des RequestOffers selon les préférences:
        - Si préférences définies: mode séquentiel (1 offre à la fois avec timeout)
        - Sinon: mode broadcast (toutes les entreprises éligibles)

        Change le statut de DRAFT à SENT.
        """
        try:
            from application.institutions import SendTransportRequestUseCase
            from application.institutions.send_transport_request import (
                SendTransportRequestInput,
            )

            institution_id, user_id = get_institution_context()

            # Utiliser le use case pour envoyer
            use_case = SendTransportRequestUseCase()
            result = use_case.execute(
                SendTransportRequestInput(
                    transport_request_id=request_id,
                    institution_id=institution_id,
                    user_id=user_id,
                )
            )

            if not result.success:
                return {"error": result.error}, result.status_code

            # Recharger la demande pour retourner la version mise à jour
            transport_req = TransportRequest.query.get(request_id)
            if not transport_req:
                return {"error": "Demande non trouvée après envoi"}, 500

            logger.info(
                "[TransportRequests] Demande envoyée: id=%s, mode=%s, offers=%d",
                transport_req.id,
                result.mode,
                result.offers_created,
            )

            # Retourner avec infos sur les offres créées
            response = transport_req.serialize
            response["send_info"] = {
                "mode": result.mode,
                "offers_created": result.offers_created,
            }

            return response, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[TransportRequests] POST /%s/send error: %s", request_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@institution_requests_ns.route("/<int:request_id>/external-carrier")
class TransportRequestExternalCarrier(Resource):
    """Endpoint pour affecter un transporteur externe."""

    @institution_requests_ns.doc(
        description="Bascule la demande vers un transporteur externe (snapshot).",
    )
    @institution_requests_ns.response(200, "Transporteur externe affecté", transport_request_model)
    @institution_requests_ns.response(400, "Données invalides", validation_error_model)
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(404, "Demande non trouvée", not_found_error_model)
    @institution_requests_ns.response(409, "Transition impossible", api_error_model)
    @api_key_or_jwt_required(scopes=["requests:write"])
    def post(self, request_id: int):
        try:
            from application.institutions import AssignExternalCarrierUseCase
            from application.institutions.assign_external_carrier import (
                AssignExternalCarrierInput,
            )

            institution_id, user_id = get_institution_context()
            data = request.get_json() or {}
            try:
                validated = cast(dict[str, Any], assign_external_carrier_schema.load(data))
            except ValidationError as err:
                return {"error": "Données invalides", "details": err.messages}, 400

            result = AssignExternalCarrierUseCase().execute(
                AssignExternalCarrierInput(
                    transport_request_id=request_id,
                    institution_id=institution_id,
                    user_id=user_id,
                    name=validated["name"],
                    phone=validated.get("phone"),
                    email=validated.get("email"),
                    reference=validated.get("reference"),
                    reason=validated.get("reason"),
                )
            )
            if not result.success:
                return {"error": result.error}, result.status_code

            transport_req = TransportRequest.query.get(request_id)
            if not transport_req:
                return {"error": "Demande non trouvée après affectation"}, 500
            return transport_req.serialize, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[TransportRequests] POST /%s/external-carrier error: %s",
                request_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_requests_ns.route("/<int:request_id>/external-completion")
class TransportRequestExternalCompletion(Resource):
    """Endpoint pour déclarer une mission externe réalisée."""

    @institution_requests_ns.doc(
        description="Déclare une mission externe comme réalisée par l'institution.",
    )
    @institution_requests_ns.response(
        200, "Mission externe déclarée réalisée", transport_request_model
    )
    @institution_requests_ns.response(400, "Données invalides", validation_error_model)
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(404, "Demande non trouvée", not_found_error_model)
    @institution_requests_ns.response(409, "Transition impossible", api_error_model)
    @api_key_or_jwt_required(scopes=["requests:write"])
    def post(self, request_id: int):
        try:
            from application.institutions import CompleteExternalMissionUseCase
            from application.institutions.complete_external_mission import (
                CompleteExternalMissionInput,
            )

            institution_id, user_id = get_institution_context()
            data = request.get_json() or {}
            try:
                validated = cast(
                    dict[str, Any], complete_external_mission_schema.load(data)
                )
            except ValidationError as err:
                return {"error": "Données invalides", "details": err.messages}, 400

            result = CompleteExternalMissionUseCase().execute(
                CompleteExternalMissionInput(
                    transport_request_id=request_id,
                    institution_id=institution_id,
                    user_id=user_id,
                    executed_at=validated.get("executed_at"),
                    notes=validated.get("notes"),
                )
            )
            if not result.success:
                return {"error": result.error}, result.status_code

            transport_req = TransportRequest.query.get(request_id)
            if not transport_req:
                return {"error": "Demande non trouvée après déclaration"}, 500
            return transport_req.serialize, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[TransportRequests] POST /%s/external-completion error: %s",
                request_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_requests_ns.route("/<int:request_id>/cancel")
class TransportRequestCancel(Resource):
    """Endpoint pour annuler une demande."""

    @institution_requests_ns.doc(
        description="Annule une demande de transport.",
    )
    @institution_requests_ns.response(200, "Demande annulée", transport_request_model)
    @institution_requests_ns.response(
        400, "Demande non annulable", validation_error_model
    )
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(404, "Demande non trouvée", not_found_error_model)
    @institution_requests_ns.response(409, "Demande convertie", api_error_model)
    @api_key_or_jwt_required(scopes=["requests:cancel"])
    def post(self, request_id: int):
        """Annule une demande de transport.

        Auth: JWT ou API Key (scope requests:cancel)

        Règles ÉTAPE 5:
        - DRAFT, SENT, ACCEPTED -> CANCELLED (sans frais)
        - CONVERTED -> 409 avec resulting_booking_id (annulation via booking)
        - Annule aussi les offres PENDING associées
        """
        try:
            from models import OfferStatus, RequestOffer

            institution_id, user_id = get_institution_context()

            transport_req = TransportRequest.query.filter_by(
                id=request_id,
                institution_id=institution_id,
            ).first()

            if not transport_req:
                return {"error": "Demande non trouvée"}, 404

            # ÉTAPE 5: Si CONVERTED, renvoyer 409 avec le booking_id
            if transport_req.status == RequestStatus.CONVERTED.value:
                return {
                    "error": "Demande déjà convertie en booking. Annulez le booking directement.",
                    "resulting_booking_id": transport_req.booking_id,
                    "status": transport_req.status,
                }, 409

            if not transport_req.is_cancellable:
                return {
                    "error": f"Demande non annulable (statut: {transport_req.status})",
                    "allowed_statuses": [
                        s.value for s in RequestStatus.cancellable_statuses()
                    ],
                }, 400

            # Récupérer raison si fournie
            data = request.get_json() or {}
            cancel_reason = data.get("reason", "")

            previous_status = transport_req.status

            # ÉTAPE 5: Annuler les offres PENDING associées
            pending_offers = RequestOffer.query.filter_by(
                transport_request_id=transport_req.id,
                status=OfferStatus.PENDING.value,
            ).all()

            cancelled_offers_count = 0
            for offer in pending_offers:
                offer.status = OfferStatus.UNAVAILABLE.value
                offer.responded_at = datetime.now(UTC)
                cancelled_offers_count += 1

            # Annuler la demande
            transport_req.status = RequestStatus.CANCELLED.value
            transport_req.cancelled_at = datetime.now(UTC)

            # Timeline : tracer l'annulation (auteur + motif) pour l'historique
            try:
                from services.institutions.transport_timeline_service import (
                    TimelineActor,
                    record_event,
                    resolve_actor_name,
                )

                record_event(
                    "cancelled",
                    institution_id=institution_id,
                    transport_request_id=transport_req.id,
                    actor=TimelineActor(
                        actor_type="institution_user" if user_id else "api_key",
                        actor_user_id=user_id,
                    ),
                    payload={
                        "actor_name": resolve_actor_name(user_id),
                        "cancellation_display_label": cancel_reason or None,
                        "reason": cancel_reason or None,
                        "previous_status": previous_status,
                    },
                    correlation_id=f"request_cancelled:{transport_req.id}",
                )
            except Exception as timeline_err:
                logger.warning(
                    "[TransportRequests] Timeline cancelled recording failed: %s",
                    timeline_err,
                )

            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="transport_request_cancelled",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution" if user_id else "api_key",
                    institution_id=institution_id,
                    result_status="success",
                    action_details={
                        "request_id": transport_req.id,
                        "external_reference": transport_req.external_reference,
                        "previous_status": previous_status,
                        "cancel_reason": cancel_reason,
                        "cancelled_offers_count": cancelled_offers_count,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[TransportRequests] Audit log error: %s", audit_err)

            logger.info(
                "[TransportRequests] Demande annulée: id=%s, previous_status=%s, offers_cancelled=%d",
                transport_req.id,
                previous_status,
                cancelled_offers_count,
            )

            return transport_req.serialize, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[TransportRequests] POST /%s/cancel error: %s", request_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@institution_requests_ns.route("/by-reference/<string:external_reference>")
class TransportRequestByReference(Resource):
    """Endpoint pour récupérer une demande par référence externe."""

    @institution_requests_ns.doc(
        description="Récupère une demande par sa référence externe DPI.",
    )
    @institution_requests_ns.response(200, "Succès", transport_request_model)
    @institution_requests_ns.response(401, "Non authentifié", permission_error_model)
    @institution_requests_ns.response(403, "Accès refusé", permission_error_model)
    @institution_requests_ns.response(404, "Demande non trouvée", not_found_error_model)
    @api_key_or_jwt_required(scopes=["requests:read"])
    def get(self, external_reference: str):
        """Récupère une demande par référence externe.

        Auth: JWT (tous rôles institution) ou API Key (scope requests:read)
        """
        try:
            institution_id, _, _ = get_institution_read_context()

            transport_req = TransportRequest.find_by_external_reference(
                institution_id, external_reference
            )

            if not transport_req:
                return {"error": "Demande non trouvée"}, 404

            return transport_req.serialize, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[TransportRequests] GET /by-reference/%s error: %s",
                external_reference,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)
