import logging
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlencode

from flask import request
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_mail import Message  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)

from app import sentry_sdk
from ext import mail, role_required
from middleware.trace_id import get_trace_id
from models import BillingParty, Client, ClientBillingParty, ClientStay, Company, db
from models.enums import GenderEnum, UserRole
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.user_repository import UserRepository
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from services.security.idempotency import IdempotencyService
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import (
    get_current_user_via_use_case,
)

TOTAL_AMOUNT_ZERO = 0

logger = logging.getLogger(__name__)

# Initialisation des repositories et services
user_repo = UserRepository()
client_repo = ClientRepository()
booking_repo = BookingRepository()

clients_ns = Namespace(
    "clients",
    description="Opérations liées aux profils clients et à leurs réservations",
)

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(clients_ns)
validation_error_model = create_validation_error_model(clients_ns)
not_found_error_model = create_not_found_error_model(clients_ns)
permission_error_model = create_permission_error_model(clients_ns)

# Modèle pour la mise à jour du profil client
client_profile_model = clients_ns.model(
    "ClientProfile",
    {
        "first_name": fields.String(description="Prénom", min_length=1, max_length=100),
        "last_name": fields.String(description="Nom", min_length=1, max_length=100),
        "phone": fields.String(
            description="Téléphone (format: +33123456789 ou 0123456789)", max_length=20
        ),
        "address": fields.String(description="Adresse", min_length=1, max_length=500),
        "birth_date": fields.String(
            description="Date de naissance (YYYY-MM-DD)",
            pattern="^\\d{4}-\\d{2}-\\d{2}$",
        ),
        "gender": fields.String(description="Genre", enum=["HOMME", "FEMME", "AUTRE"]),
    },
)

# Modèle pour la création d'une réservation
booking_create_model = clients_ns.model(
    "BookingCreate",
    {
        "dropoff_location": fields.String(required=True, description="Lieu de dépose"),
        "scheduled_time": fields.String(
            required=True, description="Date et heure prévue (format ISO 8601)"
        ),
        "amount": fields.Float(description="Montant de la réservation", default=10),
    },
)

# -------------------------------------------------------------------
# Gestion du profil client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>")
class ManageClientProfile(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )
            if (
                current_user.public_id != public_id
                and current_user.role != UserRole.admin
            ):
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized access",
                    logger_instance=logger,
                )
            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            return cast("Any", client).serialize, 200
        except Exception as e:
            logger.error(
                "❌ ERREUR manage_client_profile GET: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(client_profile_model)
    def put(self, public_id):  # noqa: PLR0911
        try:
            # Validation initiale combinée
            current_user = get_current_user_via_use_case()

            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )

            if (
                current_user.public_id != public_id
                and current_user.role != UserRole.admin
            ):
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized access",
                    logger_instance=logger,
                )

            client = client_repo.find_by_public_id_with_user(public_id)

            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.client_schemas import ClientUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    ClientUpdateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            # ✅ DDD: Utiliser le use case pour mettre à jour le client (champs Client)
            from application.companies.clients.update_company_client import (
                UpdateCompanyClientUseCase,
            )

            # Préparer les données pour le use case (champs Client uniquement)
            client_data = {}
            if "phone" in validated_data:
                client_data["contact_phone"] = validated_data["phone"]
            if "address" in validated_data:
                client_data["domicile_address"] = validated_data["address"]
            if validated_data.get("birth_date"):
                client_data["birth_date"] = validated_data["birth_date"]
            if "avs_number" in validated_data:
                client_data["avs_number"] = validated_data["avs_number"]

            # Utiliser le use case pour les champs Client
            if client_data:
                uc = UpdateCompanyClientUseCase()
                result = uc.execute(client=client, data=client_data)
                if not result.ok:
                    return result.error or {
                        "error": "Failed to update client"
                    }, result.status_code or 400

            # Mise à jour des champs User (non gérés par le use case actuel)
            if validated_data.get("first_name"):
                client.user.first_name = validated_data["first_name"]
            if validated_data.get("last_name"):
                client.user.last_name = validated_data["last_name"]
            if "gender" in validated_data:
                client.user.gender = GenderEnum(validated_data["gender"])

            db.session.commit()
            return {"message": "Profile updated successfully"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "❌ ERREUR manage_client_profile PUT: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Récupération des réservations récentes du client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/recent-bookings")
class RecentBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            bookings = booking_repo.find_models_by_client_id(client.id, limit=4)
            return [cast("Any", booking).serialize for booking in bookings], 200
        except Exception as e:
            logger.error("❌ ERREUR recent_bookings: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Liste et création de réservations pour le client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/bookings")
class ClientBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            bookings = booking_repo.find_models_by_client_id(client.id)
            return [cast("Any", booking).serialize for booking in bookings], 200
        except Exception as e:
            logger.error(
                "❌ ERREUR list_client_bookings: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(booking_create_model)
    @clients_ns.response(200, "Réservation créée avec succès (idempotency)")
    @clients_ns.response(201, "Réservation créée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(
        409, "Réservation déjà existante (idempotency)", api_error_model
    )
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def post(self, _public_id):  # noqa: PLR0911
        """Créer une réservation pour un client.

        ✅ P0: Support idempotency-key pour éviter les doublons.
        """
        try:
            # ✅ P0: Vérifier idempotency-key
            idempotency_key = IdempotencyService.get_idempotency_key_from_request()
            if idempotency_key:
                cached_response = IdempotencyService.check_key(idempotency_key)
                if cached_response[0]:  # Key exists
                    logger.info(
                        "Idempotency key found, returning cached response",
                        extra={
                            "trace_id": get_trace_id(),
                            "idempotency_key": idempotency_key,
                        },
                    )
                    return cached_response[1], 201

            # Validation initiale combinée
            current_user = get_current_user_via_use_case()
            client = None
            if current_user:
                client = client_repo.find_by_user_id(current_user.id)

            # Validation combinée pour réduire les returns
            if not current_user:
                return APIErrorHandler.handle_not_found("User", None, logger)
            if not client:
                return APIErrorHandler.handle_not_found("Client profile", None, logger)

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.booking_schemas import BookingCreateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    BookingCreateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            # Validation du format de date et de l'heure future (regroupée)
            scheduled_time = None
            error_response = None
            try:
                dt = datetime.fromisoformat(validated_data["scheduled_time"])
                scheduled_time = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                if dt.tzinfo:
                    scheduled_time = dt.astimezone(UTC)
                if scheduled_time <= datetime.now(UTC):
                    error_response = APIErrorHandler.handle_validation_error(
                        "Scheduled time must be in the future",
                        field="scheduled_time",
                        logger_instance=logger,
                    )
            except ValueError:
                error_response = APIErrorHandler.handle_validation_error(
                    "Invalid scheduled_time format",
                    field="scheduled_time",
                    logger_instance=logger,
                )

            if error_response or scheduled_time is None:
                return error_response or APIErrorHandler.handle_validation_error(
                    "Invalid scheduled_time",
                    field="scheduled_time",
                    logger_instance=logger,
                )

            # ✅ DDD: Utiliser le use case pour créer la réservation
            from bookings.infrastructure.adapters.booking_service_adapter import (
                create_booking_via_use_case,
            )

            # Préparer les données pour le use case
            booking_data = {
                "customer_name": f"{client.user.first_name} {client.user.last_name}",
                "pickup_location": validated_data["pickup_location"],
                "dropoff_location": validated_data["dropoff_location"],
                "scheduled_time": scheduled_time.isoformat(),
                "amount": validated_data.get("amount", 10),
            }

            try:
                new_booking = create_booking_via_use_case(
                    user_id=current_user.id, client_id=client.id, data=booking_data
                )

                # ✅ P0: Ajouter trace_id dans la réponse
                trace_id = get_trace_id()
                logger.info(
                    "✅ Réservation créée avec succès: booking_id=%s, client_id=%s",
                    new_booking.id if hasattr(new_booking, "id") else None,
                    client.id,
                    extra={
                        "trace_id": trace_id,
                        "booking_id": new_booking.id
                        if hasattr(new_booking, "id")
                        else None,
                        "client_id": client.id,
                    },
                )

                result = {
                    "message": "Booking created successfully",
                    "booking": new_booking.serialize,
                    "trace_id": trace_id,
                }

                # ✅ P0: Stocker la réponse pour idempotency
                if idempotency_key:
                    IdempotencyService.store_response(idempotency_key, result, 201)

                return result, 201
            except (ValueError, RuntimeError) as e:
                # Erreur de validation ou de géocodage
                return APIErrorHandler.handle_validation_error(
                    str(e),
                    logger_instance=logger,
                )

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR create_booking: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Séjours (hospitalisation / établissement) - P2 (backoffice/company)
# -------------------------------------------------------------------


def _serialize_client_stay(stay: ClientStay) -> dict[str, Any]:
    clinic = Company.query.filter_by(id=stay.company_id).first()
    return {
        "id": stay.id,
        "client_id": stay.client_id,
        "company_id": stay.company_id,
        "company_name": clinic.name if clinic else None,
        "start_date": stay.start_date.isoformat() if stay.start_date else None,
        "end_date": stay.end_date.isoformat() if stay.end_date else None,
        "status": stay.status,
        "source": stay.source,
        "notes": stay.notes,
        "created_by_user_id": stay.created_by_user_id,
        "created_at": stay.created_at.isoformat() if stay.created_at else None,
        "updated_at": stay.updated_at.isoformat() if stay.updated_at else None,
    }


def _serialize_client_stay_with_clinic_details(
    stay: ClientStay, *, owner_company_id: int | None = None
) -> dict[str, Any]:
    """Sérialise un séjour avec toutes les informations de la clinique (adresse, tarif, etc.)."""
    clinic = Company.query.filter_by(id=stay.company_id).first()
    if not clinic:
        # Fallback si la clinique n'existe plus
        return {
            "id": stay.id,
            "client_id": stay.client_id,
            "company_id": stay.company_id,
            "clinic": None,
            "start_date": stay.start_date.isoformat() if stay.start_date else None,
            "end_date": stay.end_date.isoformat() if stay.end_date else None,
            "status": stay.status,
            "source": stay.source,
            "notes": stay.notes,
        }

    # Construire l'adresse complète de la clinique
    clinic_address_parts = []
    if clinic.address:
        clinic_address_parts.append(clinic.address)
    elif clinic.domicile_address_line1:
        clinic_address_parts.append(clinic.domicile_address_line1)
        if clinic.domicile_address_line2:
            clinic_address_parts.append(clinic.domicile_address_line2)
    if clinic.domicile_zip:
        clinic_address_parts.append(clinic.domicile_zip)
    if clinic.domicile_city:
        clinic_address_parts.append(clinic.domicile_city)

    clinic_address = ", ".join(clinic_address_parts) if clinic_address_parts else None

    fallback_domicile_address = None
    fallback_domicile_zip = None
    fallback_domicile_city = None

    missing_structured = not (
        clinic.domicile_address_line1
        or clinic.domicile_zip
        or clinic.domicile_city
    )
    if missing_structured:
        fallback_client = None
        if owner_company_id is not None:
            fallback_client = Client.query.filter_by(
                default_billed_to_company_id=clinic.id,
                is_institution=True,
                company_id=owner_company_id,
            ).first()
        if not fallback_client and owner_company_id is not None and clinic.name:
            fallback_client = Client.query.filter_by(
                is_institution=True,
                company_id=owner_company_id,
                institution_name=clinic.name,
            ).first()
        if fallback_client:
            fallback_domicile_address = getattr(
                fallback_client, "domicile_address", None
            )
            fallback_domicile_zip = getattr(fallback_client, "domicile_zip", None)
            fallback_domicile_city = getattr(fallback_client, "domicile_city", None)
            fallback_parts = [
                part
                for part in [
                    fallback_domicile_address,
                    fallback_domicile_zip,
                    fallback_domicile_city,
                ]
                if part
            ]
            if fallback_parts:
                clinic_address = ", ".join(fallback_parts)

    return {
        "id": stay.id,
        "client_id": stay.client_id,
        "company_id": stay.company_id,
        "start_date": stay.start_date.isoformat() if stay.start_date else None,
        "end_date": stay.end_date.isoformat() if stay.end_date else None,
        "status": stay.status,
        "source": stay.source,
        "notes": stay.notes,
        "clinic": {
            "id": clinic.id,
            "name": clinic.name,
            "address": clinic_address,
            "domicile_address_line1": clinic.domicile_address_line1
            or fallback_domicile_address,
            "domicile_address_line2": clinic.domicile_address_line2,
            "domicile_zip": clinic.domicile_zip or fallback_domicile_zip,
            "domicile_city": clinic.domicile_city or fallback_domicile_city,
            "domicile_country": clinic.domicile_country,
            "latitude": float(clinic.latitude) if clinic.latitude else None,
            "longitude": float(clinic.longitude) if clinic.longitude else None,
            "contact_email": clinic.contact_email,
            "contact_phone": clinic.contact_phone,
            "preferential_rate": (
                float(clinic.preferential_rate) if clinic.preferential_rate is not None else None
            ),
        },
    }


def _serialize_client_billing_party_link(
    link: ClientBillingParty,
) -> dict[str, Any]:
    return {
        "id": link.id,
        "client_id": link.client_id,
        "billing_party_id": link.billing_party_id,
        "role": link.role,
        "is_default": bool(link.is_default),
        "contact_name": link.contact_name,
        "contact_email": link.contact_email,
        "contact_phone": link.contact_phone,
        "billing_party": link.billing_party.to_dict() if link.billing_party else None,
    }


def _resolve_client_for_company(client_id: str, company_id: int) -> Client | None:
    """Résout un client par ID numérique ou public_id pour une entreprise."""
    candidate = None
    try:
        if str(client_id).isdigit():
            candidate = Client.query.filter_by(
                id=int(client_id), company_id=company_id
            ).first()
    except (TypeError, ValueError):
        candidate = None

    if candidate:
        return candidate

    return Client.query.filter_by(
        public_id=str(client_id), company_id=company_id
    ).first()


def _ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _validate_no_overlap_active_stays(
    *,
    client_id: int,
    stay_id_to_exclude: int | None,
    start_date,
    end_date,
) -> None:
    """Empêche 2 séjours actifs recouvrants pour un même client (règle P2.3)."""
    from datetime import UTC, datetime

    far_future = datetime(9999, 12, 31, tzinfo=UTC)
    new_end = end_date or far_future

    q = ClientStay.query.filter_by(client_id=client_id, status="active")
    if stay_id_to_exclude:
        q = q.filter(ClientStay.id != stay_id_to_exclude)

    for s in q.all():
        s_end = s.end_date or far_future
        if _ranges_overlap(start_date, new_end, s.start_date, s_end):
            raise ValueError(
                "Un séjour actif existe déjà sur cette période (recouvrement interdit)."
            )


@clients_ns.route("/<int:client_id>/stays")
class ClientStays(Resource):
    """CRUD minimal des séjours (P2)."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def get(self, client_id: int):
        """Lister les séjours d'un client."""
        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        stays = (
            ClientStay.query.filter_by(client_id=client.id)
            .order_by(ClientStay.start_date.desc())
            .all()
        )
        return {"success": True, "data": [_serialize_client_stay(s) for s in stays]}, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self, client_id: int):  # noqa: PLR0911
        """Créer un séjour."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        from routes.companies import get_company_from_token
        from schemas.client_stay_schemas import ClientStayCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request
        from shared.infrastructure.adapters.auth_adapter import (
            get_current_user_via_use_case,
        )
        from shared.time_utils import now_utc, to_utc

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        data = request.get_json() or {}
        try:
            validated = validate_request(ClientStayCreateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        clinic_company_id = int(validated["company_id"])
        clinic = Company.query.filter_by(id=clinic_company_id).first()
        if not clinic:
            return APIErrorHandler.handle_validation_error(
                "Clinique (company_id) introuvable",
                field="company_id",
                logger_instance=logger,
            )

        start_date = to_utc(validated.get("start_date"))
        end_date = to_utc(validated.get("end_date"))
        if start_date is None:
            return APIErrorHandler.handle_validation_error(
                "start_date invalide (ISO datetime requis)",
                field="start_date",
                logger_instance=logger,
            )
        if end_date is not None and start_date > end_date:
            return APIErrorHandler.handle_validation_error(
                "start_date doit être <= end_date",
                logger_instance=logger,
            )

        status = (validated.get("status") or "active").strip().lower()
        source = (validated.get("source") or "manual").strip() or "manual"
        notes = validated.get("notes")

        try:
            if status == "active":
                _validate_no_overlap_active_stays(
                    client_id=client.id,
                    stay_id_to_exclude=None,
                    start_date=start_date,
                    end_date=end_date,
                )
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )

        user = get_current_user_via_use_case()
        stay = ClientStay()
        stay.client_id = client.id
        stay.company_id = clinic_company_id
        stay.start_date = start_date
        stay.end_date = end_date
        stay.status = status
        stay.source = source
        stay.notes = notes
        stay.created_by_user_id = getattr(user, "id", None) if user else None
        stay.created_at = now_utc()
        stay.updated_at = now_utc()

        try:
            db.session.add(stay)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur création ClientStay: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_client_stay(stay)}, 201


@clients_ns.route("/client-stays/<int:stay_id>")
class ClientStayById(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def patch(self, stay_id: int):  # noqa: PLR0911
        """Modifier un séjour."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        from routes.companies import get_company_from_token
        from schemas.client_stay_schemas import ClientStayUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request
        from shared.time_utils import now_utc, to_utc

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        stay = ClientStay.query.filter_by(id=stay_id).first()
        if not stay:
            return APIErrorHandler.handle_not_found("ClientStay", stay_id, logger)

        client = Client.query.filter_by(
            id=stay.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        data = request.get_json() or {}
        try:
            validated = validate_request(ClientStayUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        if "company_id" in validated and validated["company_id"] is not None:
            clinic_company_id = int(validated["company_id"])
            clinic = Company.query.filter_by(id=clinic_company_id).first()
            if not clinic:
                return APIErrorHandler.handle_validation_error(
                    "Clinique (company_id) introuvable",
                    field="company_id",
                    logger_instance=logger,
                )
            stay.company_id = clinic_company_id

        if "start_date" in validated and validated["start_date"] is not None:
            start_date = to_utc(validated["start_date"])
            if start_date is None:
                return APIErrorHandler.handle_validation_error(
                    "start_date invalide (ISO datetime requis)",
                    field="start_date",
                    logger_instance=logger,
                )
            stay.start_date = start_date

        if "end_date" in validated:
            stay.end_date = to_utc(validated.get("end_date"))

        if "status" in validated and validated["status"] is not None:
            stay.status = str(validated["status"]).strip().lower()

        if "source" in validated:
            v = validated.get("source")
            stay.source = (str(v).strip() if isinstance(v, str) else None) or stay.source

        if "notes" in validated:
            stay.notes = validated.get("notes")

        if stay.end_date is not None and stay.start_date > stay.end_date:
            return APIErrorHandler.handle_validation_error(
                "start_date doit être <= end_date",
                logger_instance=logger,
            )

        try:
            if (stay.status or "").strip().lower() == "active":
                _validate_no_overlap_active_stays(
                    client_id=stay.client_id,
                    stay_id_to_exclude=stay.id,
                    start_date=stay.start_date,
                    end_date=stay.end_date,
                )
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )

        stay.updated_at = now_utc()
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur update ClientStay: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_client_stay(stay)}, 200


@clients_ns.route("/<int:client_id>/active-stay")
class ClientActiveStay(Resource):
    """Récupère le séjour actif d'un client avec toutes les informations de la clinique."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def get(self, client_id: int):
        """Récupère le séjour actif d'un client avec les détails complets de la clinique."""
        from datetime import UTC, datetime

        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        # Récupérer le séjour actif (sans date spécifique, on prend le plus récent actif)
        now = datetime.now(UTC)
        stays = (
            ClientStay.query.filter_by(client_id=client.id, status="active")
            .filter(ClientStay.start_date <= now)
            .filter((ClientStay.end_date.is_(None)) | (ClientStay.end_date >= now))
            .order_by(ClientStay.start_date.desc())
            .limit(1)
            .all()
        )

        if not stays:
            return {"success": True, "data": None}, 200

        stay = stays[0]
        return {
            "success": True,
            "data": _serialize_client_stay_with_clinic_details(
                stay, owner_company_id=company.id
            ),
        }, 200


@clients_ns.route("/client-stays/<int:stay_id>/close")
class ClientStayClose(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self, stay_id: int):  # noqa: PLR0911
        """Clôturer un séjour."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        from routes.companies import get_company_from_token
        from schemas.client_stay_schemas import ClientStayCloseSchema
        from schemas.validation_utils import handle_validation_error, validate_request
        from shared.time_utils import now_utc, to_utc

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        stay = ClientStay.query.filter_by(id=stay_id).first()
        if not stay:
            return APIErrorHandler.handle_not_found("ClientStay", stay_id, logger)

        client = Client.query.filter_by(
            id=stay.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        data = request.get_json() or {}
        try:
            validated = validate_request(ClientStayCloseSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        end_date = to_utc(validated.get("end_date")) or now_utc()
        if stay.start_date and end_date < stay.start_date:
            return APIErrorHandler.handle_validation_error(
                "end_date doit être >= start_date",
                logger_instance=logger,
            )

        stay.end_date = end_date
        stay.status = "closed"
        stay.updated_at = now_utc()

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur close ClientStay: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_client_stay(stay)}, 200


# -------------------------------------------------------------------
# Tiers payeurs / curateurs (liens Client ↔ BillingParty)
# -------------------------------------------------------------------


@clients_ns.route("/<string:client_id>/billing-parties")
class ClientBillingParties(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def get(self, client_id: str):
        """Lister les tiers payeurs liés à un client."""
        from sqlalchemy.orm import joinedload

        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = _resolve_client_for_company(client_id, company.id)
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        links = (
            ClientBillingParty.query.options(
                joinedload(ClientBillingParty.billing_party)
            )
            .filter_by(client_id=client.id)
            .order_by(ClientBillingParty.id.desc())
            .all()
        )
        payload = [_serialize_client_billing_party_link(link) for link in links]
        return {"success": True, "data": payload}, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self, client_id: str):
        """Créer (ou mettre à jour) un lien client ↔ tiers payeur."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        from routes.companies import get_company_from_token
        from schemas.client_billing_party_schemas import (
            ClientBillingPartyLinkCreateSchema,
        )
        from schemas.validation_utils import handle_validation_error, validate_request

        company, err, code = get_company_from_token()
        error_response = None
        error_code = None
        if err:
            error_response, error_code = err, code or 400
        elif not company:
            error_response, error_code = APIErrorHandler.handle_not_found(
                "Company", None, logger
            )

        client = None
        if error_response is None:
            assert company is not None
            client = _resolve_client_for_company(client_id, company.id)
            if not client:
                error_response, error_code = APIErrorHandler.handle_not_found(
                    "Client", client_id, logger
                )

        validated = None
        if error_response is None:
            data = request.get_json() or {}
            try:
                validated = validate_request(
                    ClientBillingPartyLinkCreateSchema(), data, strict=False
                )
            except ValidationError as e:
                error_response, error_code = handle_validation_error(e)

        billing_party = None
        if error_response is None and validated is not None:
            assert company is not None
            billing_party_id = int(validated["billing_party_id"])
            billing_party = BillingParty.query.filter_by(
                id=billing_party_id, company_id=company.id
            ).first()
            if not billing_party:
                error_response, error_code = APIErrorHandler.handle_validation_error(
                    "Tiers payeur introuvable ou n'appartient pas à l'entreprise",
                    field="billing_party_id",
                    logger_instance=logger,
                )

        if error_response is not None:
            return error_response, error_code

        assert client is not None
        assert validated is not None
        assert billing_party is not None

        role = (validated.get("role") or "").strip() or None
        is_default = bool(validated.get("is_default"))
        contact_name = (validated.get("contact_name") or "").strip() or None
        contact_email = (validated.get("contact_email") or "").strip() or None
        contact_phone = (validated.get("contact_phone") or "").strip() or None

        link = ClientBillingParty.query.filter_by(
            client_id=client.id, billing_party_id=billing_party.id
        ).first()
        created = False
        if not link:
            link = ClientBillingParty(
                client_id=client.id,
                billing_party_id=billing_party.id,
            )
            created = True
            db.session.add(link)
            db.session.flush()

        link.role = role
        link.is_default = bool(is_default)
        link.contact_name = contact_name
        link.contact_email = contact_email
        link.contact_phone = contact_phone

        if is_default:
            ClientBillingParty.query.filter(
                ClientBillingParty.client_id == client.id,
                ClientBillingParty.id != link.id,
            ).update({"is_default": False})

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur création lien tiers payeur: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        status = 201 if created else 200
        return {
            "success": True,
            "data": _serialize_client_billing_party_link(link),
        }, status


@clients_ns.route("/billing-party-links/<int:link_id>")
class ClientBillingPartyLink(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def patch(self, link_id: int):  # noqa: PLR0911
        """Mettre à jour un lien client ↔ tiers payeur."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        from routes.companies import get_company_from_token
        from schemas.client_billing_party_schemas import (
            ClientBillingPartyLinkUpdateSchema,
        )
        from schemas.validation_utils import handle_validation_error, validate_request

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        link = ClientBillingParty.query.filter_by(id=link_id).first()
        if not link:
            return APIErrorHandler.handle_not_found("ClientBillingParty", link_id, logger)

        client = Client.query.filter_by(
            id=link.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        data = request.get_json() or {}
        try:
            validated = validate_request(
                ClientBillingPartyLinkUpdateSchema(), data, strict=False
            )
        except ValidationError as e:
            return handle_validation_error(e)

        if "role" in validated:
            link.role = (validated.get("role") or "").strip() or None
        if "contact_name" in validated:
            link.contact_name = (
                (validated.get("contact_name") or "").strip() or None
            )
        if "contact_email" in validated:
            link.contact_email = (
                (validated.get("contact_email") or "").strip() or None
            )
        if "contact_phone" in validated:
            link.contact_phone = (
                (validated.get("contact_phone") or "").strip() or None
            )

        if "is_default" in validated:
            is_default = bool(validated.get("is_default"))
            link.is_default = is_default
            if is_default:
                ClientBillingParty.query.filter(
                    ClientBillingParty.client_id == client.id,
                    ClientBillingParty.id != link.id,
                ).update({"is_default": False})

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur update lien tiers payeur: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_client_billing_party_link(link)}, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def delete(self, link_id: int):
        """Supprimer un lien client ↔ tiers payeur."""
        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        link = ClientBillingParty.query.filter_by(id=link_id).first()
        if not link:
            return APIErrorHandler.handle_not_found("ClientBillingParty", link_id, logger)

        client = Client.query.filter_by(
            id=link.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        try:
            db.session.delete(link)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur suppression lien tiers payeur: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "message": "Lien supprimé"}, 200


# -------------------------------------------------------------------
# Génération de QR bill pour le client
# -------------------------------------------------------------------


@clients_ns.route("/me/generate-qr-bill")
class GenerateQRBill(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )
            # Récupérer le client associé à l'utilisateur (avec user pour accéder aux attributs)
            client = client_repo.find_by_user_id_with_user(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            payments = getattr(client, "payments", []) or []
            total_amount = sum(
                (getattr(p, "amount", 0) or 0)
                for p in payments
                if getattr(p, "status", None) == "pending"
            )
            if total_amount <= TOTAL_AMOUNT_ZERO:
                return APIErrorHandler.handle_validation_error(
                    "No pending payments to generate a QR bill",
                    logger_instance=logger,
                )
            upid = current_user.public_id or ""
            params = urlencode({"amount": total_amount, "client": upid})
            qr_code_url = f"https://example.com/qr-payment?{params}"
            return {"qr_code_url": qr_code_url}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "❌ ERREUR generate_qr_bill: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Suppression du compte client
# -------------------------------------------------------------------


@clients_ns.route("/me")
class DeleteAccount(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def delete(self):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            # Récupérer le client associé à l'utilisateur (modèle SQLAlchemy pour modification)
            from models import Client

            client_model = Client.query.filter(
                Client.user_id == current_user.id
            ).first()
            if client_model is None:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )

            if not client_model.is_active:
                return APIErrorHandler.handle_validation_error(
                    "Account is already deactivated",
                    logger_instance=logger,
                )
            client_model.is_active = False
            current_user.is_active = False
            db.session.commit()
            return {"message": "Account deactivated successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR delete_account: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Liste des paiements du client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/payments")
class ListPayments(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = client_repo.find_by_public_id_with_payments(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            payments = client.payments
            if not payments:
                return {"message": "No payments found"}, 404
            return [payment.serialize for payment in payments], 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR list_payments: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Annulation d'une réservation (client)
# -------------------------------------------------------------------


@clients_ns.route("/me/bookings/<int:booking_id>")
class CancelBooking(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.response(200, "Réservation annulée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(404, "Réservation non trouvée", not_found_error_model)
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def delete(self, booking_id):
        """Annuler une réservation.

        ✅ P0: Support trace_id pour le suivi.
        """
        try:
            # 🔒 get user (public_id) → user.id, puis récupérer le client
            current_user = get_current_user_via_use_case()
            if not current_user:
                trace_id = get_trace_id()
                error_response, status_code = APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )
                error_response["trace_id"] = trace_id
                return error_response, status_code
            client = client_repo.find_by_user_id_with_bookings(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            booking = booking_repo.find_model_by_id_and_client(booking_id, client.id)
            if not booking:
                return APIErrorHandler.handle_not_found(
                    "Booking",
                    booking_id if "booking_id" in locals() else None,
                    logger,
                )
            # ✅ DDD: Utiliser le use case pour annuler la réservation
            from application.bookings.cancel_booking import (
                CancelBookingInput,
                CancelBookingUseCase,
            )

            uc = CancelBookingUseCase()
            input_data = CancelBookingInput(booking=booking)
            uc_result = uc.execute(input_data)
            if not uc_result.success:
                return uc_result.error or {
                    "error": "Bad request"
                }, uc_result.status_code or 400

            db.session.commit()

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "✅ Réservation annulée avec succès: booking_id=%s, client_id=%s",
                booking_id,
                client.id,
                extra={
                    "trace_id": trace_id,
                    "booking_id": booking_id,
                    "client_id": client.id,
                },
            )

            return {
                "message": "Booking canceled successfully",
                "trace_id": trace_id,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR cancel_booking: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Réinitialisation du mot de passe client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/reset-password")
class ResetPassword(Resource):
    # ✅ S2: Fresh token requis pour changement de mot de passe (action sensible)
    @jwt_required(fresh=True)
    @role_required(UserRole.client)
    def post(self, public_id):
        try:
            # Validation initiale (regroupée pour réduire les return statements)
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )

            if current_user.public_id != public_id:
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized access",
                    logger_instance=logger,
                )

            data = request.get_json()
            if not data:
                return APIErrorHandler.handle_validation_error(
                    "No data provided",
                    logger_instance=logger,
                )

            old_password = data.get("old_password", "").strip()
            new_password = data.get("new_password", "").strip()
            confirm_password = data.get("confirm_password", "").strip()

            # Validation des champs - combiner toutes les validations
            # pour réduire les returns
            error_message = None

            if not old_password or not new_password or not confirm_password:
                error_message = "All fields are required"
            elif not current_user.check_password(old_password):
                error_message = "Incorrect old password"
            elif new_password != confirm_password:
                error_message = "New passwords do not match"

            # Validation explicite du mot de passe avant set_password (sécurité)
            if not error_message:
                # ✅ S3: Validation avec politique renforcée (complexité + HIBP + historique)
                from security.password_policy import (
                    PasswordPolicyError,
                    PasswordPolicyService,
                )

                try:
                    PasswordPolicyService.validate_password(
                        new_password, user_id=current_user.id, check_history=True
                    )
                except PasswordPolicyError as e:
                    error_message = e.message

            if error_message:
                return APIErrorHandler.handle_validation_error(
                    error_message,
                    field="password",
                    logger_instance=logger,
                )

            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            current_user.set_password(new_password)  # nosem
            db.session.commit()

            # Envoi de l'email de confirmation (regroupé avec le return final)
            result_message = "Password reset successfully"
            if current_user.email:
                msg = Message(
                    subject="Confirmation de changement de mot de passe",
                    sender="support@votreapp.com",
                    recipients=[current_user.email],
                    body=(
                        f"Bonjour {current_user.first_name},\n\n"
                        "Votre mot de passe a été modifié avec succès. "
                        "Si vous n'êtes pas à l'origine de cette modification, "
                        "veuillez contacter immédiatement notre support."
                    ),
                )
                mail.send(msg)
                logger.info(
                    "✅ Mot de passe réinitialisé avec succès pour l'utilisateur %s",
                    current_user.email,
                )
                result_message = (
                    "Password reset successfully and confirmation email sent."
                )
            else:
                logger.warning(
                    "⚠️ Mot de passe réinitialisé mais email non trouvé pour l'utilisateur %s",
                    current_user.public_id,
                )

            return {"message": result_message}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR reset_password: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Recherche et création de clients pour l'autocomplete / inline
# -------------------------------------------------------------------
@clients_ns.route("/")
class ClientsList(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """GET /clients?search=<query>
        Retourne les clients dont le prénom ou le nom contient <query>.
        """
        try:
            q = request.args.get("search", "")
            # Si pas de query, on renvoie une liste vide
            if not q:
                return [], 200

            # Requête sur le champ first_name et last_name
            clients = client_repo.find_by_search_with_user(q)

            # Sérialisation
            return [cast("Any", c).serialize for c in clients], 200

        except Exception as e:
            logger.exception(
                "❌ ERREUR clients GET / : %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    @clients_ns.expect(
        clients_ns.model(
            "ClientCreate",
            {
                "first_name": fields.String(required=True),
                "last_name": fields.String(required=True),
                "email": fields.String(required=True),
                "phone": fields.String(),
                "address": fields.String(
                    description="Adresse de domicile (sera géocodée automatiquement)"
                ),
                "billing_address": fields.String(
                    description=(
                        "Adresse de facturation (optionnelle, "
                        "sera géocodée automatiquement)"
                    )
                ),
                "domicile_address": fields.String(
                    description=(
                        "Adresse de domicile (optionnelle, "
                        "sera géocodée automatiquement)"
                    )
                ),
                "domicile_zip": fields.String(description="Code postal"),
                "domicile_city": fields.String(description="Ville"),
            },
        )
    )
    @clients_ns.response(200, "Client créé avec succès (idempotency)")
    @clients_ns.response(201, "Client créé avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(409, "Client déjà existant (idempotency)", api_error_model)
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def post(self):
        """POST /clients
        Crée un nouveau client avec géocodage automatique des adresses.

        ✅ P0: Support idempotency-key pour éviter les doublons.
        """
        try:
            # ✅ P0: Vérifier idempotency-key
            idempotency_key = IdempotencyService.get_idempotency_key_from_request()
            if idempotency_key:
                cached_response = IdempotencyService.check_key(idempotency_key)
                if cached_response[0]:  # Key exists
                    logger.info(
                        "Idempotency key found, returning cached response",
                        extra={
                            "trace_id": get_trace_id(),
                            "idempotency_key": idempotency_key,
                        },
                    )
                    return cached_response[1], 201

            data = request.get_json() or {}
            # Validation basique
            for field in ("first_name", "last_name", "email"):
                if not data.get(field):
                    return APIErrorHandler.handle_validation_error(
                        f"{field} manquant",
                        field=field,
                        logger_instance=logger,
                    )

            # Obtenir l'utilisateur actuel pour récupérer company_id
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Utilisateur non trouvé",
                    logger_instance=logger,
                )

            # ⚠️ TODO DDD: Migrer vers CreateCompanyClientUseCase une fois l'adapter créé
            # Pour l'instant, création directe car ClientRepository n'implémente pas le port requis
            from repositories.company_repository import CompanyRepository

            # Récupérer la company de l'utilisateur actuel
            company_repo = CompanyRepository()
            company = company_repo.find_by_user_id(current_user.id)
            if not company:
                return APIErrorHandler.handle_permission_error(
                    "Company not found for current user",
                    logger_instance=logger,
                )

            # Créer l'utilisateur (création directe temporaire, à migrer vers use case)
            from models import Client, User

            new_user = cast("Any", User)(
                first_name=data["first_name"],
                last_name=data["last_name"],
                email=data["email"],
                role=UserRole.client,
            )
            db.session.add(new_user)
            db.session.flush()  # récupère new_user.id

            # Créer le client (création directe temporaire, à migrer vers use case)
            new_client = cast("Any", Client)(
                user_id=new_user.id,
                contact_phone=data.get("phone"),
                domicile_zip=data.get("domicile_zip"),
                domicile_city=data.get("domicile_city"),
            )

            # Déterminer l'adresse principale à géocoder
            # Priorité: domicile_address > address
            main_address = data.get("domicile_address") or data.get("address")

            # Géocodage de l'adresse de domicile
            if main_address:
                try:
                    from services.geolocation.maps import geocode_address

                    coords = geocode_address(main_address.strip(), country="CH")
                    if coords:
                        new_client.domicile_address = main_address
                        new_client.domicile_lat = coords.get("lat")
                        new_client.domicile_lon = coords.get("lon")
                        log_msg = (
                            "✅ Adresse de domicile géocodée pour %s %s: %s -> (%s, %s)"
                        )
                        logger.info(
                            log_msg,
                            data["first_name"],
                            data["last_name"],
                            main_address,
                            coords.get("lat"),
                            coords.get("lon"),
                        )
                    else:
                        # Sauvegarde l'adresse même sans coordonnées
                        new_client.domicile_address = main_address
                        logger.warning(
                            "⚠️ Impossible de géocoder l'adresse de domicile: %s",
                            main_address,
                        )
                except Exception as e:
                    # Sauvegarde l'adresse même en cas d'erreur
                    new_client.domicile_address = main_address
                    logger.warning(
                        "⚠️ Erreur lors du géocodage de l'adresse de domicile: %s", e
                    )

            # Géocodage de l'adresse de facturation (si différente)
            billing_address = data.get("billing_address")
            if billing_address and billing_address.strip():
                try:
                    from services.geolocation.maps import geocode_address

                    coords = geocode_address(billing_address.strip(), country="CH")
                    if coords:
                        new_client.billing_address = billing_address
                        new_client.billing_lat = coords.get("lat")
                        new_client.billing_lon = coords.get("lon")
                        log_msg = (
                            "✅ Adresse de facturation géocodée pour %s %s: %s "
                            "-> (%s, %s)"
                        )
                        logger.info(
                            log_msg,
                            data["first_name"],
                            data["last_name"],
                            billing_address,
                            coords.get("lat"),
                            coords.get("lon"),
                        )
                    else:
                        new_client.billing_address = billing_address
                        logger.warning(
                            "⚠️ Impossible de géocoder l'adresse de facturation: %s",
                            billing_address,
                        )
                except Exception as e:
                    new_client.billing_address = billing_address
                    logger.warning(
                        "⚠️ Erreur lors du géocodage de l'adresse de facturation: %s", e
                    )
            elif main_address:
                # Si pas d'adresse de facturation spécifique, copier depuis
                # domicile
                new_client.billing_address = new_client.domicile_address
                new_client.billing_lat = new_client.domicile_lat
                new_client.billing_lon = new_client.domicile_lon

            # Associer le client à la même compagnie que l'utilisateur actuel
            if hasattr(current_user, "company_id") and current_user.company_id:
                new_client.company_id = current_user.company_id

            db.session.add(new_client)
            db.session.commit()

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "✅ Client créé avec succès: %s %s (ID: %s)",
                data["first_name"],
                data["last_name"],
                new_client.id,
                extra={"trace_id": trace_id, "client_id": new_client.id},
            )

            # ✅ P0: Stocker la réponse pour idempotency
            response_data = new_client.serialize
            if isinstance(response_data, dict):
                response_data["trace_id"] = trace_id

            if idempotency_key:
                IdempotencyService.store_response(idempotency_key, response_data, 201)

            return response_data, 201

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "❌ ERREUR clients POST / : %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)


@clients_ns.route("/<int:client_id>/create-company")
class CreateCompanyForInstitutionClient(Resource):
    """Créer une Company pour un Client institution (clinique)."""

    @jwt_required()
    @role_required(UserRole.company)
    @clients_ns.response(200, "Company créée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(404, "Client non trouvé", not_found_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    def post(self, client_id: int):  # noqa: PLR0911
        """Créer une Company pour un Client institution.

        Cette endpoint permet de créer automatiquement une Company à partir
        des informations d'un Client avec is_institution=true. La Company créée
        sera utilisée pour le mapping de facturation.
        """
        try:
            # Récupérer l'entreprise courante
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Utilisateur non trouvé",
                    logger_instance=logger,
                )

            from repositories.company_repository import CompanyRepository

            company_repo = CompanyRepository()
            current_company = company_repo.find_by_user_id(current_user.id)
            if not current_company:
                return APIErrorHandler.handle_permission_error(
                    "Company not found for current user",
                    logger_instance=logger,
                )

            # Récupérer le client
            client = client_repo.find_model_by_id_and_company(
                client_id, current_company.id
            )
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client", client_id, logger
                )

            # Vérifier que c'est une institution
            from models.base import _as_bool

            if not _as_bool(client.is_institution):
                return APIErrorHandler.handle_validation_error(
                    "Ce client n'est pas une institution",
                    logger_instance=logger,
                )

            # Vérifier qu'il n'y a pas déjà une Company associée
            if client.default_billed_to_company_id:
                existing_company = Company.query.filter_by(
                    id=client.default_billed_to_company_id
                ).first()
                if existing_company:
                    return {
                        "success": True,
                        "data": existing_company.serialize,
                        "message": "Company déjà associée",
                    }, 200

            # Créer une nouvelle Company à partir des informations du client
            from models import User

            # Créer un User pour la Company (ou réutiliser celui du client)
            company_user = User.query.filter_by(id=client.user_id).first()
            if not company_user:
                return APIErrorHandler.handle_not_found(
                    "User", client.user_id, logger
                )

            # Créer la Company
            domicile_lat = getattr(client, "domicile_lat", None)
            domicile_lon = getattr(client, "domicile_lon", None)
            domicile_address = getattr(client, "domicile_address", None) or ""
            domicile_zip = getattr(client, "domicile_zip", None) or ""
            domicile_city = getattr(client, "domicile_city", None) or ""
            postal_city = " ".join(part for part in [domicile_zip, domicile_city] if part)
            full_address = (
                f"{domicile_address}, {postal_city}".strip(", ")
                if domicile_address or postal_city
                else ""
            )

            new_company = Company(
                name=client.institution_name or f"Clinique #{client.id}",
                user_id=company_user.id,
                address=full_address,
                domicile_address_line1=domicile_address or None,
                domicile_zip=domicile_zip or None,
                domicile_city=domicile_city or None,
                latitude=(
                    float(domicile_lat) if domicile_lat is not None else None
                ),
                longitude=(
                    float(domicile_lon) if domicile_lon is not None else None
                ),
                contact_email=client.contact_email or company_user.email or "",
                contact_phone=client.contact_phone or company_user.phone or "",
                service_area="",
                max_daily_bookings=50,
                is_approved=False,
                preferential_rate=(
                    client.preferential_rate
                    if getattr(client, "preferential_rate", None) is not None
                    else None
                ),
            )

            db.session.add(new_company)
            db.session.flush()

            # Associer la Company au client
            client.default_billed_to_company_id = new_company.id
            # Créer automatiquement un BillingParty et un mapping pour cette clinique
            billing_address = client.billing_address or client.domicile_address or ""
            if client.domicile_zip and client.domicile_city:
                if billing_address:
                    billing_address = (
                        f"{billing_address}\n{client.domicile_zip} {client.domicile_city}"
                    )
                else:
                    billing_address = f"{client.domicile_zip} {client.domicile_city}"

            from models import BillingParty, BillingPartyType, ClinicBillingPartyMapping

            billing_party = BillingParty(
                company_id=current_company.id,
                type=BillingPartyType.CLINIC,
                display_name=new_company.name,
                billing_address=billing_address or "Adresse non renseignée",
                contact_email=new_company.contact_email,
                contact_phone=new_company.contact_phone,
                external_ref=f"clinic_company:{new_company.id}",
                is_active=True,
            )
            db.session.add(billing_party)
            db.session.flush()

            # Créer le mapping
            mapping = ClinicBillingPartyMapping(
                company_id=current_company.id,
                clinic_company_id=new_company.id,
                billing_party_id=billing_party.id,
                is_active=True,
            )
            db.session.add(mapping)

            db.session.commit()

            logger.info(
                "✅ Company, BillingParty et Mapping créés pour institution client %s: company_id=%s (%s), billing_party_id=%s, mapping_id=%s",
                client_id,
                new_company.id,
                new_company.name,
                billing_party.id,
                mapping.id,
            )

            return {
                "success": True,
                "data": {
                    **new_company.serialize,
                    "billing_party_id": billing_party.id,
                    "billing_party_name": billing_party.display_name,
                    "mapping_id": mapping.id,
                },
                "message": "Company, BillingParty et Mapping créés avec succès",
            }, 201

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "❌ ERREUR création Company pour client %s: %s - %s",
                client_id,
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
