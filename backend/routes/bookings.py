from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from functools import wraps
from io import BytesIO
from os import getenv
from typing import Any

import sentry_sdk
from flask import (
    request,
    send_file,
    url_for,
)
from flask_jwt_extended import jwt_required
from flask_restx import (
    Namespace,
    Resource,
    fields,
)

from ext import db, limiter, role_required
from infrastructure.dispatch import queue_adapter as queue
from middleware.trace_id import get_trace_id
from models.enums import UserRole
from models.payment import Payment
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.driver_repository import DriverRepository
from repositories.user_repository import UserRepository
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from routes.api_error_utils import api_error
from schemas.booking_schemas import BookingCreateSchema
from schemas.validation_utils import handle_validation_error, validate_request
from services.booking.expire_unpaid_client_bookings import (
    expire_awaiting_client_payment_bookings,
)
from services.platform_exceptions import PlatformTenantSuspended
from services.platform_governance_constants import ERROR_TENANT_PLATFORM_SUSPENDED
from services.saferpay.config import saferpay_configured
from services.saferpay.finalize_payment import finalize_saferpay_payment
from services.saferpay.payment_page import create_saferpay_payment_page_initialize
from services.security.idempotency import IdempotencyService
from shared.client_surface_contracts import (
    PRICING_CONTRACT_VERSION,
    PRICING_STATUS_VALUES,
)
from shared.constants import PaginationConstants
from shared.error_handlers import APIErrorHandler
from shared.response_helpers import created_response, success_response
from shared.time_utils import to_utc

# ✅ REFACTORING: Utilisation de constantes centralisées
# Alias pour compatibilité avec le code existant
PAGE_ONE = PaginationConstants.PAGE_ONE

_MONTH_DECEMBER = 12
_PDF_NEW_PAGE_Y_THRESHOLD = 70
_PRICING_ADJUSTMENT_EPSILON = 0.01

logger = logging.getLogger(__name__)

# GET /bookings/ — liste ; surcharge env RATELIMIT_BOOKINGS_LIST
_RATELIMIT_BOOKINGS_LIST = getenv("RATELIMIT_BOOKINGS_LIST", "2000 per hour")

# Initialisation des repositories
booking_repo = BookingRepository()
client_repo = ClientRepository()
driver_repo = DriverRepository()
user_repo = UserRepository()

# Création du Namespace pour les réservations
bookings_ns = Namespace("bookings", description="Opérations relatives aux réservations")

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(bookings_ns)
validation_error_model = create_validation_error_model(bookings_ns)
not_found_error_model = create_not_found_error_model(bookings_ns)
permission_error_model = create_permission_error_model(bookings_ns)

# Modèle Swagger (ajout is_round_trip)
booking_create_model = bookings_ns.model(
    "BookingCreate",
    {
        "customer_name": fields.String(
            required=True, min_length=1, max_length=200, description="Nom du client"
        ),
        "pickup_location": fields.String(
            required=True,
            min_length=1,
            max_length=500,
            description="Lieu de prise en charge",
        ),
        "dropoff_location": fields.String(
            required=True, min_length=1, max_length=500, description="Lieu de dépose"
        ),
        "scheduled_time": fields.String(
            required=True,
            description=(
                "ISO 8601 (ex: 2024-01-15T14:30:00). "
                "Si asap=true, peut être omis ou null : l'API fixe une heure proche."
            ),
        ),
        "asap": fields.Boolean(
            description="« Dès que possible » : scheduled_time dérivé côté serveur",
            default=False,
        ),
        "amount": fields.Float(
            required=True, min=0, description="Montant de la réservation"
        ),
        "medical_facility": fields.String(
            description="Établissement médical", default="", max_length=200
        ),
        "doctor_name": fields.String(
            description="Nom du médecin", default="", max_length=200
        ),
        "hospital_service": fields.String(
            description="Service ou unité (ex. urgences, cardiologie)",
            default="",
            max_length=100,
        ),
        "is_round_trip": fields.Boolean(
            description="Créer également un retour", default=False
        ),
        "return_time": fields.String(
            description="ISO 8601 pour l'heure de retour (optionnel)", default=None
        ),
    },
)

# Modèle Swagger pour mise à jour
booking_update_model = bookings_ns.model(
    "BookingUpdate",
    {
        "pickup_location": fields.String(min_length=1, max_length=500),
        "dropoff_location": fields.String(min_length=1, max_length=500),
        "scheduled_time": fields.String(description="ISO 8601"),
        "amount": fields.Float(min=0),
        "status": fields.String(
            enum=["pending", "confirmed", "in_progress", "completed", "cancelled"]
        ),
        "medical_facility": fields.String(max_length=200),
        "doctor_name": fields.String(max_length=200),
        "is_round_trip": fields.Boolean(),
        "notes_medical": fields.String(max_length=1000),
    },
)

booking_export_pdf_model = bookings_ns.model(
    "BookingExportPdfRequest",
    {
        "client_public_id": fields.String(required=False),
        "period": fields.String(
            required=True,
            enum=["this_month", "previous_month", "this_year", "custom"],
        ),
        "from": fields.String(
            required=False, description="ISO 8601 pour période custom"
        ),
        "to": fields.String(required=False, description="ISO 8601 pour période custom"),
    },
)

# -----------------------------------------------------
# Helper: déclenche le moteur de dispatch de manière sûre


def _queue_trigger(company_id: int | None, action: str) -> None:
    if not company_id:
        return
    # Déclenchement AUTOMATIQUE : on précise l'origine pour que le garde-fou
    # d'automatisation (queue.trigger) s'applique en mode MANUAL.
    from models import DispatchTriggerOrigin

    origin = (
        DispatchTriggerOrigin.CANCELLATION
        if action == "cancel"
        else DispatchTriggerOrigin.BOOKING_CHANGE
    )
    try:
        # API moderne
        t1 = getattr(queue, "trigger_on_booking_change", None)
        if callable(t1):
            t1(company_id, reason=f"booking_{action}", origin=origin)
            return
        # API alternative
        t2 = getattr(queue, "trigger", None)
        if callable(t2):
            t2(company_id, reason=f"booking_{action}", mode="auto", origin=origin)
            return
    except Exception as e:
        logger.warning("⚠️ _queue_trigger failed: %s", e)


# -----------------------------------------------------
# Helper: construit les liens de pagination RFC 5988


def _build_pagination_links(
    page: int, per_page: int, total: int, endpoint: str, **kwargs
):
    """Construit les liens de pagination conformes RFC 5988.

    Returns:
        dict avec 'Link' header + metadata pagination

    """
    from flask import current_app

    total_pages = (total + per_page - 1) // per_page
    links = []
    # Sécuriser l'URL externe pour éviter Host header injection
    # Utiliser SERVER_NAME de la config Flask (pas request.host)
    server_name = current_app.config.get("SERVER_NAME")
    if not server_name:
        # Fallback sécurisé: utiliser localhost si SERVER_NAME non configuré
        server_name = "localhost"
    scheme = current_app.config.get("PREFERRED_URL_SCHEME", "https")
    base_url = f"{scheme}://{server_name}"

    if page > PAGE_ONE:
        url = url_for(endpoint, page=page - 1, per_page=per_page, **kwargs)
        links.append(f'<{base_url}{url}>; rel="prev"')
    if page < total_pages:
        url = url_for(endpoint, page=page + 1, per_page=per_page, **kwargs)
        links.append(f'<{base_url}{url}>; rel="next"')

    url = url_for(endpoint, page=1, per_page=per_page, **kwargs)
    links.append(f'<{base_url}{url}>; rel="first"')
    url = url_for(endpoint, page=total_pages, per_page=per_page, **kwargs)
    links.append(f'<{base_url}{url}>; rel="last"')

    return {
        "Link": ", ".join(links),
        "X-Total-Count": str(total),
        "X-Page": str(page),
        "X-Per-Page": str(per_page),
        "X-Total-Pages": str(total_pages),
    }


# =====================================================
# 🔐 SECURITY: Ownership Check Helper (CWE-284)
# =====================================================
def _check_booking_ownership(
    booking,  # Booking model from repository
    user,  # User model from repository
    action: str = "access",
) -> tuple[bool, tuple[dict[str, str], int] | None]:
    """Vérifie si l'utilisateur a le droit d'accéder/modifier ce booking.

    Args:
        booking: Le booking à vérifier
        user: L'utilisateur authentifié
        action: Type d'action ("read", "modify", "delete")

    Returns:
        (has_access: bool, error_response_tuple_or_none)

    Exemple:
        has_access, error = _check_booking_ownership(booking, user, "modify")
        if not has_access:
            return error  # ({"error": "..."}, 403)

    """
    user_role_value = str(getattr(user.role, "value", user.role))
    error_response = ({"error": f"Accès non autorisé ({action})"}, 403)

    # Admin a tous les droits
    if user_role_value == UserRole.admin.value:
        return True, None

    # Company a accès à tous ses bookings
    if user_role_value == UserRole.company.value:
        from repositories.company_repository import CompanyRepository

        company_repo = CompanyRepository()
        company = company_repo.find_by_user_id(user.id)
        has_access = company is not None and company.id == booking.company_id
        return (True, None) if has_access else (False, error_response)

    # Client propriétaire
    if user_role_value == UserRole.client.value:
        client = client_repo.find_by_user_id(user.id)
        if not client:
            logger.warning(
                "⚠️ User %s has client role but no Client record", user.public_id
            )
        elif client.id == booking.client_id:
            return True, None
        else:
            # IDOR attempt détecté
            warning_msg = (
                "🚨 IDOR blocked: user=%s (client_id=%s) tried to %s "
                + "booking_id=%s (owner_client_id=%s)"
            )
            logger.warning(
                warning_msg,
                user.public_id,
                client.id,
                action,
                booking.id,
                booking.client_id,
            )
            error_response = ({"error": "Accès non autorisé à cette réservation"}, 403)
        return False, error_response

    # Driver assigné (read-only access)
    if user_role_value == UserRole.driver.value and action == "read":
        driver = driver_repo.find_model_by_user_id(user.id)
        has_access = driver is not None and booking.driver_id == driver.id
        return (True, None) if has_access else (False, error_response)

    # Aucun droit
    return False, error_response


# =====================================================
# 🔐 SECURITY: Décorateur pour forcer vérification ownership (CWE-284)
# =====================================================
def require_booking_ownership(action: str = "access"):
    """Décorateur pour forcer vérification ownership booking.

    Args:
        action: Type d'action ("read", "modify", "delete")

    Usage:
        @bookings_ns.route("/<int:booking_id>")
        class BookingResource(Resource):
            @require_booking_ownership("read")
            def get(self, booking_id, booking, user):  # booking et user injectés
                return booking.serialize, 200
    """

    def decorator(f):
        @wraps(f)
        @jwt_required()
        def decorated_function(*args, **kwargs):
            # Récupérer booking_id depuis kwargs ou args
            booking_id = kwargs.get("booking_id")
            if booking_id is None and args:
                # Si booking_id est dans les arguments positionnels
                booking_id = args[0] if args else None

            if booking_id is None:
                return APIErrorHandler.handle_validation_error(
                    "booking_id_required",
                    field="booking_id",
                    logger_instance=logger,
                )

            # Charger le booking avec eager loading pour éviter N+1
            booking = booking_repo.find_model_by_id_with_eager_loading(booking_id)

            if not booking:
                return APIErrorHandler.handle_not_found(
                    "Réservation", booking_id, logger
                )

            # ✅ DDD: Utilise use-case au lieu de service directement
            from shared.infrastructure.adapters.auth_adapter import (
                get_current_user_via_use_case,
            )

            user = get_current_user_via_use_case()
            if not user:
                return APIErrorHandler.handle_permission_error(
                    "Utilisateur non authentifié",
                    logger_instance=logger,
                )

            # Vérifier ownership
            has_access, error_response = _check_booking_ownership(booking, user, action)
            if not has_access:
                return error_response

            # Injecter booking et user dans kwargs pour éviter requête supplémentaire
            kwargs["booking"] = booking
            kwargs["user"] = user
            return f(*args, **kwargs)

        return decorated_function

    return decorator


# =====================================================
# Création d'une réservation pour un client
# =====================================================
def _validate_user_and_client(
    public_id: str,
) -> tuple[
    Any | None, Any | None, tuple[dict[str, str], int] | None
]:  # Returns User and Client models from repositories
    """Valide utilisateur et client. Retourne (user, client, error_response).

    ✅ DDD: Utilise use-cases au lieu de service directement.
    """
    from shared.infrastructure.adapters.auth_adapter import (
        get_current_user_via_use_case,
    )

    # Récupérer l'utilisateur via use-case
    user = get_current_user_via_use_case()
    if not user:
        return None, None, ({"message": "Utilisateur non authentifié"}, 401)

    # Récupérer le client et vérifier ownership
    client = client_repo.find_by_public_id(public_id)
    if not client or client.user_id != user.id:
        return (
            None,
            None,
            ({"message": "Client non trouvé ou non associé à cet utilisateur"}, 403),
        )

    return user, client, None


# Helper pour gérer les erreurs de géocodage
def _handle_geocoding_error(error: RuntimeError) -> tuple[dict[str, str], int]:
    """Gère les erreurs de géocodage et retourne la réponse HTTP appropriée.

    Args:
        error: Exception RuntimeError du service

    Returns:
        Tuple (response_dict, status_code)
    """
    error_msg = str(error)
    if "temporairement indisponible" in error_msg.lower():
        return {
            "error": "erreur_geocodage",
            "message": error_msg,
        }, 400
    return {
        "error": "impossible_de_geocoder",
        "message": error_msg,
    }, 400


def _parse_iso_or_none(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _period_bounds(
    period: str, custom_from: datetime | None, custom_to: datetime | None
) -> tuple[datetime, datetime] | None:
    now = datetime.now(UTC)
    if period == "this_month":
        start = datetime(now.year, now.month, 1, tzinfo=UTC)
        if now.month == _MONTH_DECEMBER:
            end = datetime(now.year + 1, 1, 1, tzinfo=UTC)
        else:
            end = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
        return start, end
    if period == "previous_month":
        first_this_month = datetime(now.year, now.month, 1, tzinfo=UTC)
        end = first_this_month
        start = datetime(end.year, end.month, 1, tzinfo=UTC) - timedelta(days=1)
        start = datetime(start.year, start.month, 1, tzinfo=UTC)
        return start, end
    if period == "this_year":
        start = datetime(now.year, 1, 1, tzinfo=UTC)
        end = datetime(now.year + 1, 1, 1, tzinfo=UTC)
        return start, end
    if period == "custom" and custom_from and custom_to and custom_from <= custom_to:
        return custom_from, custom_to
    return None


_STATUS_LABEL_RULES: list[tuple[frozenset[str], str]] = [
    (frozenset({"awaiting_client_payment"}), "En attente de paiement"),
    (frozenset({"pending", "requested", "awaiting", "waiting"}), "En attente"),
    (frozenset({"confirmed", "accepted", "validated", "assigned"}), "Confirmée"),
    (
        frozenset(
            {
                "driver_on_the_way",
                "driver_en_route",
                "driver_arriving",
                "en_route",
            }
        ),
        "Chauffeur en route",
    ),
    (frozenset({"in_progress", "ongoing", "started"}), "En cours"),
    (frozenset({"completed", "done", "finished", "terminated"}), "Terminée"),
    (frozenset({"cancelled", "canceled", "rejected"}), "Annulée"),
]


def _status_label(status_value: Any) -> str:
    s = str(getattr(status_value, "value", status_value) or "").lower()
    for keys, label in _STATUS_LABEL_RULES:
        if s in keys:
            return label
    return "En attente"


def _build_client_bookings_pdf(bookings: list[Any], period_label: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4
    y = height - 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, f"Export courses client - {period_label}")
    y -= 24

    headers = [
        "Date",
        "Heure",
        "Trajet",
        "Transporteur",
        "Montant",
        "Statut",
        "Référence",
        "Adresse départ",
        "Adresse arrivée",
    ]
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(40, y, " | ".join(headers))
    y -= 14
    pdf.setFont("Helvetica", 8)

    total_amount = 0.0
    for booking in bookings:
        scheduled = getattr(booking, "scheduled_time", None)
        if isinstance(scheduled, datetime):
            date_str = scheduled.strftime("%d/%m/%Y")
            time_str = scheduled.strftime("%H:%M")
        else:
            date_str = "-"
            time_str = "-"
        amount = float(getattr(booking, "amount", 0) or 0)
        total_amount += amount
        reference = f"BK-{getattr(booking, 'id', '-')}"
        row = [
            date_str,
            time_str,
            f"{getattr(booking, 'pickup_location', '')} -> {getattr(booking, 'dropoff_location', '')}",
            getattr(getattr(booking, "company", None), "name", None) or "Non assignée",
            f"{amount:.2f} CHF",
            _status_label(getattr(booking, "status", None)),
            reference,
            str(getattr(booking, "pickup_location", "") or ""),
            str(getattr(booking, "dropoff_location", "") or ""),
        ]
        pdf.drawString(40, y, " | ".join(row)[:220])
        y -= 12
        if y < _PDF_NEW_PAGE_Y_THRESHOLD:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 8)

    y -= 8
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, y, f"Total période: {total_amount:.2f} CHF")
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _validate_booking_request(
    data: dict[str, Any], public_id: str
) -> tuple[Any | None, Any | None, tuple[dict[str, str], int] | None]:
    """Valide la requête de création de booking.

    Args:
        data: Données JSON de la requête
        public_id: ID public du client

    Returns:
        Tuple (user, client, error_response) où error_response est None si OK
    """
    # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
    from marshmallow import ValidationError

    try:
        validated = validate_request(BookingCreateSchema(), data)
    except ValidationError as e:
        # Retourner None pour user/client et l'erreur de validation
        # La route gérera le return
        return None, None, handle_validation_error(e)

    # Données normalisées (ex. scheduled_time rempli si asap=true)
    data.clear()
    data.update(validated)

    # Validation utilisateur et client
    user, client, auth_error = _validate_user_and_client(public_id)
    if auth_error:
        return None, None, auth_error

    # Défense en profondeur : vérification explicite
    if user is None or client is None:
        logger.error(
            "[Bookings] user or client is None after validation (should not happen)"
        )
        error_response = APIErrorHandler.handle_exception(
            Exception("Erreur interne d'authentification"),
            logger,
        )
        return None, None, error_response

    return user, client, None


def _booking_creation_client_brief(booking: Any) -> dict[str, Any]:
    """Résumé minimal pour le portail client (POST création, sans sérialisation complète)."""
    status_val = getattr(booking.status, "value", booking.status)
    preview_amount = getattr(booking, "price_amount", None)
    raw_breakdown = getattr(booking, "price_breakdown_json", None)
    breakdown = raw_breakdown if isinstance(raw_breakdown, dict) else {}
    recalculation_reason = None
    pricing_status = "confirmed"
    if breakdown.get("overridden_by_preferential"):
        source = str(breakdown.get("preferential_source") or "policy").strip().lower()
        if source == "clinic":
            recalculation_reason = (
                "Tarif recalculé selon la convention clinique active."
            )
        elif source == "client":
            recalculation_reason = "Tarif recalculé selon le tarif préférentiel client."
        else:
            recalculation_reason = (
                "Tarif recalculé selon une règle métier de tarification serveur."
            )
        pricing_status = "adjusted"
    elif breakdown.get("pricing_amount_applied"):
        recalculation_reason = (
            "Tarif recalculé à la création selon le moteur de pricing backend."
        )
        pricing_status = "adjusted"
    if pricing_status not in PRICING_STATUS_VALUES:
        pricing_status = "confirmed"
    return {
        "id": booking.id,
        "amount": float(getattr(booking, "amount", 0) or 0),
        "price_amount": float(preview_amount or 0)
        if preview_amount is not None
        else None,
        "pricing_adjustment_reason": recalculation_reason,
        "pricing_status": pricing_status,
        "pricing_contract_version": PRICING_CONTRACT_VERSION,
        "billed_to_type": (str(getattr(booking, "billed_to_type", None) or "patient"))
        .strip()
        .lower(),
        "status": str(status_val).lower() if status_val is not None else "pending",
    }


def execute_client_booking_creation(public_id: str) -> Any:
    """Création réservation pour `public_id` utilisateur — partagé entre
    `POST /bookings/clients/<id>/bookings` et `POST /clients/<id>/bookings`."""
    try:
        idempotency_key = IdempotencyService.get_idempotency_key_from_request()
        if idempotency_key:
            cached_response = IdempotencyService.check_key(idempotency_key)
            if cached_response[0]:
                logger.info(
                    "Idempotency key found, returning cached response",
                    extra={
                        "trace_id": get_trace_id(),
                        "idempotency_key": idempotency_key,
                    },
                )
                return cached_response[1], 201

        data = request.get_json(silent=True) or {}

        user, client, validation_error = _validate_booking_request(data, public_id)
        if validation_error:
            return validation_error

        if user is None or client is None:
            return APIErrorHandler.handle_exception(
                Exception("Erreur interne d'authentification"),
                logger,
            )

        from bookings.infrastructure.adapters.booking_service_adapter import (
            create_booking_via_use_case,
        )

        try:
            from application.bookings.create_booking import InvalidClientBookingCommand

            new_booking = create_booking_via_use_case(
                user_id=user.id, client_id=client.id, data=data
            )
        except InvalidClientBookingCommand as e:
            return {
                "error": str(e),
                "error_code": e.code,
                "fields": e.fields,
            }, 400
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e),
                logger_instance=logger,
            )
        except PlatformTenantSuspended as e:
            trace_id = get_trace_id()
            return {
                "error": e.message,
                "error_code": ERROR_TENANT_PLATFORM_SUSPENDED,
                "reason_code": ERROR_TENANT_PLATFORM_SUSPENDED,
                "human_reason": e.message,
                "retryable": False,
                "trace_id": trace_id,
            }, 403
        except RuntimeError as e:
            return _handle_geocoding_error(e)

        booking_id = getattr(new_booking, "id", None)
        if new_booking is not None:
            from models.enums import BookingStatus

            billed = (
                str(getattr(new_booking, "billed_to_type", None) or "patient")
                .strip()
                .lower()
            )
            if billed == "patient":
                new_booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
                db.session.commit()

        if new_booking is not None and getattr(new_booking, "company_id", None) is None:
            try:
                from services.dispatch.open_booking_offers import (
                    seed_dispatch_offers_for_unassigned_booking,
                )

                seed_dispatch_offers_for_unassigned_booking(int(new_booking.id))
            except Exception:
                logger.exception(
                    "[Bookings] Échec seed offres dispatch pour booking ouvert id=%s",
                    getattr(new_booking, "id", None),
                )

        trace_id = get_trace_id()
        logger.info(
            "✅ Réservation créée avec succès: booking_id=%s, client_id=%s",
            booking_id,
            client.id,
            extra={
                "trace_id": trace_id,
                "booking_id": booking_id,
                "client_id": client.id,
            },
        )

        booking_brief = (
            _booking_creation_client_brief(new_booking)
            if new_booking is not None
            else None
        )
        requested_amount = data.get("amount")
        preview_amount = data.get("preview_amount")
        if booking_brief is not None:
            final_amount = booking_brief.get("amount")
            provided_reference_amount = (
                preview_amount
                if isinstance(preview_amount, (float, int))
                else requested_amount
            )
            if (
                isinstance(provided_reference_amount, (float, int))
                and isinstance(final_amount, (float, int))
                and abs(float(final_amount) - float(provided_reference_amount))
                >= _PRICING_ADJUSTMENT_EPSILON
                and not booking_brief.get("pricing_adjustment_reason")
            ):
                booking_brief["pricing_adjustment_reason"] = (
                    "Le prix a été recalculé à la création en raison d'une mise à jour "
                    "des règles de tarification ou du contexte opérationnel."
                )
                booking_brief["pricing_status"] = "adjusted"
        response_data, status_code = created_response(
            data={
                "booking_id": booking_id,
                "trace_id": trace_id,
                "booking": booking_brief,
            },
            location=f"/api/bookings/{booking_id}" if booking_id else None,
            message="Réservation créée avec succès",
        )

        if idempotency_key:
            IdempotencyService.store_response(idempotency_key, response_data, 201)

        return response_data, status_code

    except Exception as e:
        db.session.rollback()
        return APIErrorHandler.handle_exception(e, logger)


@bookings_ns.route("/clients/<string:public_id>/bookings")
class CreateBooking(Resource):
    """✅ DDD: Route simplifiée - orchestration uniquement.

    La logique métier est déléguée aux use cases dans `application/bookings/`.
    """

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("50 per hour")  # ✅ 2.8: Rate limiting création réservations
    @bookings_ns.expect(booking_create_model)
    @bookings_ns.response(200, "Réservation créée avec succès (idempotency)")
    @bookings_ns.response(201, "Réservation créée avec succès")
    @bookings_ns.response(400, "Erreur de validation", validation_error_model)
    @bookings_ns.response(401, "Non authentifié", permission_error_model)
    @bookings_ns.response(403, "Non autorisé", permission_error_model)
    @bookings_ns.response(
        409, "Réservation déjà existante (idempotency)", api_error_model
    )
    @bookings_ns.response(500, "Erreur serveur", api_error_model)
    def post(self, public_id):
        """Créer une réservation pour un client (statut PENDING).

        ✅ P0: Support idempotency-key pour éviter les doublons.
        """
        return execute_client_booking_creation(public_id)


@bookings_ns.route("/clients/me/bookings/export-pdf")
class ClientBookingsExportPdf(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    @bookings_ns.expect(booking_export_pdf_model, validate=False)
    @limiter.limit("30 per hour")
    def post(self):
        try:
            from models import Booking
            from shared.infrastructure.adapters.auth_adapter import (
                get_current_user_via_use_case,
            )

            user = get_current_user_via_use_case()
            if not user:
                return {"message": "Utilisateur non authentifié"}, 401

            client = client_repo.find_by_user_id(user.id)
            if not client:
                return {"message": "Profil client introuvable"}, 404

            data = request.get_json(silent=True) or {}
            period = str(data.get("period") or "").strip().lower()
            if period not in {"this_month", "previous_month", "this_year", "custom"}:
                return {"message": "Période invalide"}, 400

            custom_from = _parse_iso_or_none(data.get("from"))
            custom_to = _parse_iso_or_none(data.get("to"))
            bounds = _period_bounds(period, custom_from, custom_to)
            if bounds is None:
                return {"message": "Période personnalisée invalide"}, 400
            start_dt, end_dt = bounds

            bookings = (
                Booking.query.filter(Booking.client_id == client.id)
                .filter(Booking.scheduled_time >= start_dt)
                .filter(Booking.scheduled_time < end_dt)
                .order_by(Booking.scheduled_time.asc())
                .all()
            )
            if not bookings:
                return {
                    "message": "Aucune donnée exportable pour la période sélectionnée."
                }, 404

            period_label_map = {
                "this_month": "Ce mois",
                "previous_month": "Mois précédent",
                "this_year": "Cette année",
                "custom": "Période personnalisée",
            }
            period_label = period_label_map.get(period, "Période")
            pdf_bytes = _build_client_bookings_pdf(bookings, period_label)

            total_amount = sum(
                float(getattr(item, "amount", 0) or 0) for item in bookings
            )
            filename = f"courses_{period}.pdf"
            response = send_file(
                BytesIO(pdf_bytes),
                mimetype="application/pdf",
                as_attachment=True,
                download_name=filename,
            )
            response.headers["X-Period-Label"] = period_label
            response.headers["X-Total-Amount"] = f"{total_amount:.2f}"
            response.headers["X-Rows-Count"] = str(len(bookings))
            return response
        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@bookings_ns.route("/<int:booking_id>/saferpay/initialize")
class BookingSaferpayInitialize(Resource):
    """Démarre un paiement Saferpay (Payment Page) pour une réservation client."""

    @require_booking_ownership("read")
    @limiter.limit("30 per hour")
    def post(self, booking_id, booking, user):
        _ = booking_id
        from services.saferpay.config import saferpay_configured

        role_val = str(getattr(user.role, "value", user.role))
        if role_val != UserRole.client.value:
            return APIErrorHandler.handle_permission_error(
                "Le paiement en ligne est réservé aux comptes client",
                logger_instance=logger,
            )

        client = client_repo.find_by_user_id(user.id)
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Profil client introuvable",
                logger_instance=logger,
            )

        if not saferpay_configured():
            return {
                "error": "payment_unavailable",
                "message": "Paiement Saferpay non configuré sur ce serveur",
            }, 503

        payload = request.get_json(silent=True) or {}
        return_url = payload.get("return_url")

        try:
            out = create_saferpay_payment_page_initialize(
                booking=booking,
                user=user,
                client=client,
                return_url_override=return_url if isinstance(return_url, str) else None,
            )
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e),
                logger_instance=logger,
            )
        except Exception as e:
            db.session.rollback()
            if isinstance(e, RuntimeError):
                return api_error(
                    "saferpay_initialize_failed",
                    str(e),
                    503,
                )
            return APIErrorHandler.handle_exception(e, logger)

        return success_response(data=out)


@bookings_ns.route("/<int:booking_id>/saferpay/assert")
class BookingSaferpayAssert(Resource):
    """Finalise le paiement (Assert + Capture) après retour depuis Saferpay."""

    @require_booking_ownership("read")
    @limiter.limit("60 per hour")
    def post(self, booking_id, booking, user):
        _ = booking_id
        role_val = str(getattr(user.role, "value", user.role))
        if role_val != UserRole.client.value:
            return APIErrorHandler.handle_permission_error(
                "Le paiement en ligne est réservé aux comptes client",
                logger_instance=logger,
            )

        client = client_repo.find_by_user_id(user.id)
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Profil client introuvable",
                logger_instance=logger,
            )

        if not saferpay_configured():
            return {
                "error": "payment_unavailable",
                "message": "Paiement Saferpay non configuré sur ce serveur",
            }, 503

        payload = request.get_json(silent=True) or {}
        raw_payment_id = payload.get("payment_id")
        if raw_payment_id is None:
            return APIErrorHandler.handle_validation_error(
                "payment_id requis (entier)",
                logger_instance=logger,
            )
        try:
            payment_id_int = int(raw_payment_id)
        except (TypeError, ValueError):
            return APIErrorHandler.handle_validation_error(
                "payment_id requis (entier)",
                logger_instance=logger,
            )

        payment_row = Payment.query.filter_by(
            id=payment_id_int,
            booking_id=booking.id,
            client_id=client.id,
            payment_provider="saferpay",
        ).first()
        if not payment_row:
            return APIErrorHandler.handle_not_found("Payment", payment_id_int, logger)

        try:
            out = finalize_saferpay_payment(payment_row)
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)

        assert_status = str(out.get("status", "")).strip().lower()
        payment_status = "pending_verification"
        if assert_status in {"completed", "already_completed"}:
            payment_status = "paid"
        elif assert_status in {
            "payment_failed",
            "assert_failed",
            "unexpected_tx_status",
        }:
            payment_status = "failed"
        else:
            persisted_payment_status = (
                str(getattr(payment_row.status, "value", payment_row.status))
                .strip()
                .upper()
            )
            if persisted_payment_status == "FAILED":
                payment_status = "failed"
            elif persisted_payment_status == "COMPLETED":
                payment_status = "paid"

        booking_status = (
            str(getattr(booking.status, "value", booking.status)).strip().lower()
        )
        payload = {
            **out,
            "booking_id": booking.id,
            "payment_id": payment_row.id,
            "payment_provider": "saferpay",
            "payment_status": payment_status,
            "booking_status": booking_status,
            "pending_verification": payment_status == "pending_verification",
        }
        return success_response(data=payload)


# =====================================================
# Récupération, mise à jour et annulation d'une réservation
# =====================================================


@bookings_ns.route("/<int:booking_id>")
class BookingResource(Resource):
    @require_booking_ownership("read")
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting lecture réservation
    def get(self, booking_id, booking, user):  # booking et user injectés par décorateur
        """Récupère une réservation (contrôle d'accès par rôle)."""
        try:
            # ✅ Clean Architecture: Délègue au use-case GetBookingUseCase
            # Note: Le booking est déjà chargé par le décorateur, mais on utilise
            # le use-case pour la cohérence architecturale
            from application.bookings import GetBookingInput
            from application.bookings.get_booking import GetBookingUseCase

            _ = user, booking  # Marquer comme utilisé pour éviter warning
            uc = GetBookingUseCase(booking_repo=booking_repo)
            input_data = GetBookingInput(booking_id=booking_id)
            result = uc.execute(input_data)
            if not result.found or not result.booking:
                return APIErrorHandler.handle_not_found(
                    "Réservation", booking_id, logger
                )
            return success_response(data=result.booking.serialize)

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)

    @require_booking_ownership("modify")
    @limiter.limit("100 per hour")  # ✅ 2.8: Rate limiting modification réservation
    @bookings_ns.expect(booking_update_model, validate=False)
    def put(self, booking_id, booking, user):  # booking et user injectés par décorateur
        """Met à jour une réservation côté client (en attente, acceptée ou assignée, hors en route)."""
        try:
            # booking et user sont déjà chargés et vérifiés par le décorateur
            # booking_id est requis par Flask mais non utilisé ici
            _ = booking_id, user  # Marquer comme utilisé pour éviter warning

            data = request.get_json(silent=True) or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.booking_schemas import BookingUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    BookingUpdateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            # Normaliser scheduled_time (UTC) dans la route (parsing/validation)
            if "scheduled_time" in validated_data:
                try:
                    validated_data["scheduled_time"] = to_utc(
                        validated_data["scheduled_time"]
                    )
                except Exception:
                    return APIErrorHandler.handle_validation_error(
                        "Format de date invalide",
                        field="scheduled_time",
                        logger_instance=logger,
                    )

            # ✅ Clean step: règles métier dans le use-case Application
            from application.bookings import UpdatePendingBookingInput
            from application.bookings.update_pending_booking import (
                UpdatePendingBookingUseCase,
            )

            uc = UpdatePendingBookingUseCase()
            input_data = UpdatePendingBookingInput(
                booking=booking, validated_data=validated_data
            )
            uc_result = uc.execute(input_data)
            if not uc_result.success:
                return APIErrorHandler.handle_validation_error(
                    uc_result.error.get("message", "Erreur de validation")
                    if uc_result.error
                    else "Erreur de validation",
                    logger_instance=logger,
                )

            addresses_changed = bool(uc_result.addresses_changed)
            old_pickup = uc_result.old_pickup
            old_dropoff = uc_result.old_dropoff

            db.session.commit()

            # ✅ P1: Invalider cache géocodage et OSRM si adresses changées
            if addresses_changed:
                try:
                    from infrastructure.bookings.cache_invalidation_adapter import (
                        invalidate_geocoding_cache_adapter,
                        invalidate_osrm_matrix_cache_adapter,
                    )

                    # Invalider cache géocodage pour les anciennes et nouvelles adresses
                    if old_pickup:
                        invalidate_geocoding_cache_adapter(
                            old_pickup, country="CH", provider="both"
                        )
                    if old_dropoff:
                        invalidate_geocoding_cache_adapter(
                            old_dropoff, country="CH", provider="both"
                        )
                    if booking.pickup_location:
                        invalidate_geocoding_cache_adapter(
                            booking.pickup_location, country="CH", provider="both"
                        )
                    if booking.dropoff_location:
                        invalidate_geocoding_cache_adapter(
                            booking.dropoff_location, country="CH", provider="both"
                        )

                    # Invalider cache OSRM si coordonnées disponibles
                    if (
                        booking.pickup_lat
                        and booking.pickup_lon
                        and booking.dropoff_lat
                        and booking.dropoff_lon
                    ):
                        coords = [
                            (float(booking.pickup_lat), float(booking.pickup_lon)),
                            (float(booking.dropoff_lat), float(booking.dropoff_lon)),
                        ]
                        invalidate_osrm_matrix_cache_adapter(coords=coords)

                    logger.info(
                        "[Cache] ✅ Invalidated geocoding and OSRM cache for booking #%s (addresses changed)",
                        booking.id,
                    )
                except Exception as e:
                    logger.warning(
                        "[Cache] Failed to invalidate cache for booking #%s: %s",
                        booking.id,
                        e,
                    )

            # Pas de trigger si PENDING (non pris par l'engine). On log juste.
            return success_response(message="Réservation mise à jour avec succès")

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)

    @require_booking_ownership("delete")
    @limiter.limit("50 per hour")  # ✅ 2.8: Rate limiting suppression réservation
    def delete(
        self, booking_id, booking, user
    ):  # booking et user injectés par décorateur
        """Annule une réservation (PENDING ou ASSIGNED).
        Déclenche queue si nécessaire.
        """
        try:
            # booking et user sont déjà chargés et vérifiés par le décorateur
            # booking_id est requis par Flask mais non utilisé ici
            _ = booking_id, user  # Marquer comme utilisé pour éviter warning

            from application.bookings import CancelBookingInput
            from application.bookings.cancel_booking import CancelBookingUseCase

            st_before = getattr(booking.status, "value", booking.status)
            previous_status = str(st_before or "")

            uc = CancelBookingUseCase()
            input_data = CancelBookingInput(booking=booking)
            uc_result = uc.execute(input_data)
            if not uc_result.success:
                return APIErrorHandler.handle_validation_error(
                    uc_result.error.get("message", "Erreur de validation")
                    if uc_result.error
                    else "Erreur de validation",
                    logger_instance=logger,
                )

            db.session.commit()

            try:
                from middleware.trace_id import get_trace_id
                from security.audit_log import AuditLogger

                role = getattr(user, "role", None)
                user_type_str = str(
                    getattr(role, "value", role) if role is not None else "user"
                ).lower()
                AuditLogger.log_action(
                    action_type="booking_cancelled",
                    action_category="booking",
                    user_id=getattr(user, "id", None),
                    user_type=user_type_str,
                    company_id=getattr(booking, "company_id", None),
                    booking_id=getattr(booking, "id", None),
                    correlation_id=get_trace_id(),
                    action_details={
                        "source": "routes.bookings.delete",
                        "previous_status": previous_status,
                        "new_status": "canceled",
                        "trigger": "user",
                    },
                )
            except Exception as audit_exc:
                logger.critical(
                    "[booking] audit booking_cancelled failed booking_id=%s: %s",
                    getattr(booking, "id", None),
                    audit_exc,
                    exc_info=True,
                )
                try:
                    from services.monitoring.prometheus import (
                        inc_booking_audit_write_failed,
                    )

                    inc_booking_audit_write_failed(action_type="booking_cancelled")
                except Exception:
                    pass

            if uc_result.should_trigger_dispatch:
                _queue_trigger(uc_result.company_id, "cancel")

            return success_response(message="Réservation annulée avec succès")

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


# =====================================================
# Liste selon le rôle (admin / client)
# =====================================================


def _get_admin_bookings(
    page: int, per_page: int, status_filter: str | None
) -> tuple[dict[str, Any], int, dict[str, str]]:
    """Helper pour récupérer les réservations pour un admin.

    ✅ Clean Architecture: Délègue au use-case ListBookingsUseCase.
    """
    from application.bookings import ListBookingsInput
    from application.bookings.list_bookings import ListBookingsUseCase

    uc = ListBookingsUseCase(booking_repo=booking_repo)
    input_data = ListBookingsInput(
        user_role=UserRole.admin,
        user_id=0,  # Non utilisé pour admin
        page=page,
        per_page=per_page,
        status_filter=status_filter,
    )
    result = uc.execute(input_data)
    if not result or not result.success:
        return {"bookings": [], "total": 0}, 200, {}
    if not result.bookings or result.total is None:
        return {"bookings": [], "total": 0}, 200, {}

    headers = _build_pagination_links(
        page, per_page, result.total, "bookings.list_bookings"
    )
    bookings_serialized = [b.serialize for b in result.bookings]
    return {"bookings": bookings_serialized, "total": result.total}, 200, headers


def _get_client_bookings(
    user,
    page: int,
    per_page: int,
    status_filter: str | None,  # user is User model from repository
) -> tuple[dict[str, Any], int, dict[str, str]] | None:
    """Helper pour récupérer les réservations pour un client.
    Retourne None si erreur.

    ✅ Clean Architecture: Délègue au use-case ListBookingsUseCase.
    """
    from application.bookings import ListBookingsInput, ListBookingsUseCase

    expire_awaiting_client_payment_bookings()

    uc = ListBookingsUseCase(booking_repo=booking_repo, client_repo=client_repo)
    input_data = ListBookingsInput(
        user_role=user.role,
        user_id=user.id,
        page=page,
        per_page=per_page,
        status_filter=status_filter,
    )
    result = uc.execute(input_data)
    if not result or not result.success:
        return None
    # Une liste vide est un résultat valide (client sans réservation).
    if result.bookings is None or result.total is None:
        return None

    headers = _build_pagination_links(
        page, per_page, result.total, "bookings.list_bookings"
    )
    bookings_serialized = [b.serialize for b in result.bookings]
    return {"bookings": bookings_serialized, "total": result.total}, 200, headers


@bookings_ns.route("/")
class ListBookings(Resource):
    @jwt_required()
    @limiter.limit(_RATELIMIT_BOOKINGS_LIST)  # ✅ 2.8 + surcharge env
    @bookings_ns.param(
        "page",
        "Numéro de page (défaut: 1, min: 1)",
        type="integer",
        default=1,
        minimum=1,
    )
    @bookings_ns.param(
        "per_page",
        "Résultats par page (défaut: 100, min: 1, max: 500)",
        type="integer",
        default=100,
        minimum=1,
        maximum=500,
    )
    @bookings_ns.param(
        "status",
        "Filtre par statut (pending|confirmed|in_progress|completed|cancelled)",
        type="string",
        enum=["pending", "confirmed", "in_progress", "completed", "cancelled"],
    )
    @bookings_ns.param(
        "from_date",
        "Date de début (YYYY-MM-DD)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    @bookings_ns.param(
        "to_date",
        "Date de fin (YYYY-MM-DD)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    def get(self):
        """Retourne les réservations (paginées).

        Query params:
            - page: numéro de page (défaut: 1, min: 1)
            - per_page: résultats par page (défaut: 100, min: 1, max: 500)
            - status: filtre par statut
              (pending|confirmed|in_progress|completed|cancelled), optionnel
            - from_date: filtre par date de début (YYYY-MM-DD), optionnel
            - to_date: filtre par date de fin (YYYY-MM-DD), optionnel
        """
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from shared.infrastructure.adapters.auth_adapter import (
                get_current_user_via_use_case,
            )

            user = get_current_user_via_use_case()
            if not user:
                return APIErrorHandler.handle_permission_error(
                    "User not found",
                    logger_instance=logger,
                )

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import ValidationError

            from schemas.booking_schemas import BookingListSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            args_dict = dict(request.args)
            try:
                validated_args = validate_request(
                    BookingListSchema(), args_dict, strict=False
                )
                page = validated_args.get("page", 1)
                per_page = validated_args.get("per_page", 100)
                status_filter = validated_args.get("status")
            except ValidationError as e:
                return handle_validation_error(e)

            # Traitement selon le rôle
            result = None
            if user.role == UserRole.admin:
                result = _get_admin_bookings(page, per_page, status_filter)
            elif user.role == UserRole.client:
                client_result = _get_client_bookings(
                    user, page, per_page, status_filter
                )
                if client_result is not None:
                    result = client_result
                else:
                    result = APIErrorHandler.handle_permission_error(
                        "Unauthorized: No client profile found",
                        logger_instance=logger,
                    )
            else:
                result = APIErrorHandler.handle_permission_error(
                    "Unauthorized: You don't have permission",
                    logger_instance=logger,
                )

            return result

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)
