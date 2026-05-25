# routes/institution_requests.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Routes pour la gestion des demandes de transport institutionnelles.

Endpoints:
- POST /api/v1/institutions/requests - Créer une demande
- GET /api/v1/institutions/requests - Lister les demandes
- GET /api/v1/institutions/requests/{id} - Détail demande
- PUT /api/v1/institutions/requests/{id} - Modifier demande
- POST /api/v1/institutions/requests/{id}/send - Envoyer aux transporteurs
- POST /api/v1/institutions/requests/{id}/cancel - Annuler demande
"""

import logging
from datetime import UTC, datetime
from typing import Any, cast

import sentry_sdk
from flask import g, request
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
    TransportRequestCreateSchema,
    TransportRequestQuerySchema,
    TransportRequestUpdateSchema,
)
from security.api_key_auth import api_key_or_jwt_required
from security.audit_log import AuditLogger
from security.authorization import AuthorizationService, get_user_team_ids
from shared.error_handlers import APIErrorHandler
from shared.time_utils import parse_iso8601

logger = logging.getLogger(__name__)

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

    Rôles autorisés: admin, requester, curator.

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
        InstitutionRole.CURATOR.value,
    )
    return institution.id, user.id


def get_institution_read_context():
    """Récupère le contexte institution pour lecture seule (JWT ou API Key).

    Rôles autorisés: admin, requester, billing, reader, curator.

    Returns:
        Tuple (institution_id, user_id_or_none)
    """
    # Si authentifié par API Key
    if hasattr(g, "institution_id") and g.get("auth_method") == "api_key":
        return g.institution_id, None

    # Sinon JWT — tous les rôles institution peuvent lire
    institution, user = AuthorizationService.require_institution_role(
        InstitutionRole.ADMIN.value,
        InstitutionRole.REQUESTER.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.READER.value,
        InstitutionRole.CURATOR.value,
    )
    return institution.id, user.id


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
            institution_id, _ = get_institution_read_context()

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

            if params.get("external_reference"):
                query = query.filter_by(external_reference=params["external_reference"])

            if params.get("patient_id"):
                query = query.filter_by(patient_id=params["patient_id"])

            if params.get("date_from"):
                date_from = datetime.strptime(params["date_from"], "%Y-%m-%d")
                query = query.filter(TransportRequest.scheduled_time >= date_from)

            if params.get("date_to"):
                date_to = datetime.strptime(params["date_to"], "%Y-%m-%d")
                # Inclure toute la journée
                date_to = date_to.replace(hour=23, minute=59, second=59)
                query = query.filter(TransportRequest.scheduled_time <= date_to)

            # Pagination
            page = params.get("page", 1)
            per_page = params.get("per_page", 20)
            total = query.count()
            pages = (total + per_page - 1) // per_page

            requests = (
                query.order_by(TransportRequest.scheduled_time.desc())
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

            # Parser scheduled_time
            scheduled_time = parse_iso8601(validated["scheduled_time"])
            if not scheduled_time:
                return {"error": "scheduled_time invalide"}, 400

            # Créer la demande
            transport_req = TransportRequest()
            transport_req.institution_id = institution_id
            transport_req.created_by_user_id = user_id
            transport_req.external_reference = ext_ref
            transport_req.patient_id = patient_id
            transport_req.mission_type = validated.get(
                "mission_type", MissionType.PATIENT_TRANSPORT.value
            )
            transport_req.delivery_description = validated.get("delivery_description")
            transport_req.scheduled_time = scheduled_time
            transport_req.scheduled_time_type = validated.get(
                "scheduled_time_type", "departure"
            )

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
            if validated.get("return_time"):
                transport_req.return_time = parse_iso8601(validated["return_time"])
            logger.info(
                "[CreateRequest] is_round_trip=%s, raw_return_time=%r, parsed=%s",
                transport_req.is_round_trip,
                validated.get("return_time"),
                transport_req.return_time,
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
            institution_id, _ = get_institution_read_context()

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

            from flask import g
            from flask_jwt_extended import get_jwt
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

            if "scheduled_time" in validated:
                transport_req.scheduled_time = parse_iso8601(
                    validated["scheduled_time"]
                )
            if "scheduled_time_type" in validated:
                transport_req.scheduled_time_type = validated["scheduled_time_type"]

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
            if "return_time" in validated:
                transport_req.return_time = (
                    parse_iso8601(validated["return_time"])
                    if validated["return_time"]
                    else None
                )

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

            db.session.commit()

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
                        "updated_fields": list(validated.keys()),
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
            institution_id, _ = get_institution_read_context()

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
