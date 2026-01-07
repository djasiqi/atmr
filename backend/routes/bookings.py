from __future__ import annotations

import logging
from functools import wraps
from typing import Any

import sentry_sdk  # pyright: ignore[reportMissingImports]
from flask import (  # pyright: ignore[reportMissingImports]
    request,
    url_for,
)
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)

from ext import db, limiter, role_required
from infrastructure.dispatch import queue_adapter as queue
from middleware.trace_id import get_trace_id
from models.enums import UserRole
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
from schemas.booking_schemas import BookingCreateSchema
from schemas.validation_utils import handle_validation_error, validate_request
from services.security.idempotency import IdempotencyService
from shared.constants import PaginationConstants
from shared.error_handlers import APIErrorHandler
from shared.response_helpers import created_response, success_response
from shared.time_utils import to_utc

# ✅ REFACTORING: Utilisation de constantes centralisées
# Alias pour compatibilité avec le code existant
PAGE_ONE = PaginationConstants.PAGE_ONE

logger = logging.getLogger(__name__)

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
                "Note: Les dates passées sont normalement rejetées, sauf si time_confirmed=False "
                "(permis pour imports historiques)."
            ),
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

# -----------------------------------------------------
# Helper: déclenche le moteur de dispatch de manière sûre


def _queue_trigger(company_id: int | None, action: str) -> None:
    if not company_id:
        return
    try:
        # API moderne
        t1 = getattr(queue, "trigger_on_booking_change", None)
        if callable(t1):
            t1(company_id, action=action)
            return
        # API alternative
        t2 = getattr(queue, "trigger", None)
        if callable(t2):
            t2(company_id, reason=f"booking_{action}", mode="auto")
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
    from flask import current_app  # pyright: ignore[reportMissingImports]

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
    from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

    try:
        validate_request(BookingCreateSchema(), data)
    except ValidationError as e:
        # Retourner None pour user/client et l'erreur de validation
        # La route gérera le return
        return None, None, handle_validation_error(e)

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
    def post(self, public_id):  # noqa: PLR0911
        """Créer une réservation pour un client (statut PENDING).

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

            # Validation de la requête
            user, client, validation_error = _validate_booking_request(data, public_id)
            if validation_error:
                return validation_error

            # Défense en profondeur : vérification explicite pour le type checker
            if user is None or client is None:
                return APIErrorHandler.handle_exception(
                    Exception("Erreur interne d'authentification"),
                    logger,
                )

            # ✅ DDD: Utiliser directement le use-case au lieu du service
            from bookings.infrastructure.adapters.booking_service_adapter import (
                create_booking_via_use_case,
            )

            try:
                new_booking = create_booking_via_use_case(
                    user_id=user.id, client_id=client.id, data=data
                )
            except ValueError as e:
                # Erreur de validation (date, adresse, etc.)
                return APIErrorHandler.handle_validation_error(
                    str(e),
                    logger_instance=logger,
                )
            except RuntimeError as e:
                # Erreur de géocodage (adresse invalide, service indisponible, etc.)
                return _handle_geocoding_error(e)

            # ⚠️ Pas de dispatch ici (PENDING seulement).
            # L'entreprise acceptera -> ACCEPTED.
            booking_id = getattr(new_booking, "id", None)

            # ✅ P0: Ajouter trace_id dans la réponse
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

            response_data, status_code = created_response(
                data={"booking_id": booking_id, "trace_id": trace_id},
                location=f"/api/bookings/{booking_id}" if booking_id else None,
                message="Réservation créée avec succès",
            )

            # ✅ P0: Stocker la réponse pour idempotency
            if idempotency_key:
                IdempotencyService.store_response(idempotency_key, response_data, 201)

            return response_data, status_code

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


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
        """Met à jour une réservation (si PENDING). Déclenche queue si utile."""
        try:
            # booking et user sont déjà chargés et vérifiés par le décorateur
            # booking_id est requis par Flask mais non utilisé ici
            _ = booking_id, user  # Marquer comme utilisé pour éviter warning

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

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
    if not result.bookings or result.total is None:
        return None

    headers = _build_pagination_links(
        page, per_page, result.total, "bookings.list_bookings"
    )
    bookings_serialized = [b.serialize for b in result.bookings]
    return {"bookings": bookings_serialized, "total": result.total}, 200, headers


@bookings_ns.route("/")
class ListBookings(Resource):
    @jwt_required()
    @limiter.limit("300 per hour")  # ✅ 2.8: Rate limiting liste réservations
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
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

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
