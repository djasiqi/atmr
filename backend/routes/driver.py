from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from typing import cast as tcast

from flask import current_app, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (
    Namespace,
    Resource,
    fields,
)
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from ext import db, role_required, socketio
from models import DelayEvent, Driver
from models.enums import BookingStatus, DriverType, UserRole
from routes.db_error_utils import format_integrity_error
from shared.error_handlers import APIErrorHandler
from shared.notifications import notify_booking_update

# Note: Modèles (DelayEvent, Driver) utilisés pour types/annotations
# TODO: Migrer vers repositories quand les méthodes nécessaires seront disponibles

logger = logging.getLogger(__name__)

# Import conditionnel pour éviter les dépendances circulaires
try:
    from routes.companies import _maybe_trigger_dispatch
except ImportError:
    _maybe_trigger_dispatch = None

# Constantes pour éviter les valeurs magiques
LAT_THRESHOLD = 90
LON_THRESHOLD = 180
MIN_POINTS_FOR_MATCHING = 3
MIN_TOKEN_LENGTH = 10

# sentry (si initialisé dans app.py, on garde try/except pour éviter
# ImportError en tests)
try:
    from app import sentry_sdk
except Exception:

    class _S:
        def capture_exception(*a, **k): ...

    sentry_sdk = _S()

driver_ns = Namespace("driver", description="Gestion des chauffeurs")

# ---------------------------
# Modèles Swagger
# ---------------------------
driver_profile_model = driver_ns.model(
    "DriverProfileUpdate",
    {
        "first_name": fields.String(description="Prénom", min_length=1, max_length=100),
        "last_name": fields.String(description="Nom", min_length=1, max_length=100),
        "phone": fields.String(description="Téléphone", max_length=20),
        "status": fields.String(
            description="Statut", enum=["disponible", "hors service"]
        ),
        # HR fields
        "contract_type": fields.String(
            description="Type de contrat (max 50 caractères)", max_length=50
        ),
        "weekly_hours": fields.Integer(
            description="Heures contrat / semaine (0-168)", minimum=0, maximum=168
        ),
        "hourly_rate_cents": fields.Integer(
            description="Taux horaire (centimes, >= 0)", minimum=0
        ),
        "employment_start_date": fields.String(
            description="Date de début (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"
        ),
        "employment_end_date": fields.String(
            description="Date de fin (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"
        ),
        "license_categories": fields.List(
            fields.String(description="Catégorie de permis", max_length=10),
            description="Ex: ['B','C1'] (max 10 catégories)",
        ),
        "license_valid_until": fields.String(
            description="Validité permis (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"
        ),
        "trainings": fields.List(
            fields.Raw(description="Formation: {name, valid_until}"),
            description="Formations (max 50)",
        ),
        "medical_valid_until": fields.String(
            description="Validité médicale (YYYY-MM-DD)",
            pattern="^\\d{4}-\\d{2}-\\d{2}$",
        ),
    },
)

photo_model = driver_ns.model(
    "DriverPhoto",
    {"photo": fields.String(required=True, description="Photo en Base64 ou URL")},
)

location_model = driver_ns.model(
    "DriverLocation",
    {
        "latitude": fields.Float(required=True, description="Latitude"),
        "longitude": fields.Float(required=True, description="Longitude"),
        "speed": fields.Float(required=False, description="Vitesse m/s"),
        "heading": fields.Float(required=False, description="Cap en degrés"),
        "accuracy": fields.Float(required=False, description="Précision en mètres"),
        "ts": fields.String(required=False, description="Horodatage ISO8601"),
    },
)

booking_status_model = driver_ns.model(
    "BookingStatusUpdate",
    {
        "status": fields.String(
            required=True,
            description=(
                "Nouveau statut (en_route, in_progress, completed, return_completed, canceled)"
            ),
        ),
        "cancel_reason": fields.String(description="CANCEL ou RELEASE (si status=canceled)"),
        "reason_code": fields.String(
            description="Motif facturation (NO_SHOW, COMPANY_ISSUE, LAST_MINUTE, etc.)"
        ),
        "scope": fields.String(
            description="Si 'reservation', annule toute la réservation (aller + retour)"
        ),
    },
)

availability_model = driver_ns.model(
    "DriverAvailability",
    {
        "is_available": fields.Boolean(
            required=True, description="Disponibilité du chauffeur"
        )
    },
)

# ---------------------------
# Helpers
# ---------------------------


def get_driver_from_token() -> tuple[Driver | None, dict[str, Any] | None, int | None]:
    """Récupère le chauffeur associé à l'utilisateur connecté via le token JWT.
    Retourne (driver, None, None) si trouvé, sinon (None, error_response, status_code).
    """
    from repositories.driver_repository import DriverRepository
    from repositories.user_repository import UserRepository

    user_public_id = get_jwt_identity()
    logger.info("JWT Identity récupérée: %s", user_public_id)

    user_repo = UserRepository()
    user = user_repo.find_by_public_id(user_public_id)
    if not user:
        logger.error("User not found for public_id: %s", user_public_id)
        error_response, status_code = APIErrorHandler.handle_not_found(
            "User",
            user_public_id,
            logger,
        )
        return None, error_response, status_code

    logger.info("User details: id=%s, role=%s", user.id, user.role)

    if user.role != UserRole.driver:
        logger.error(
            "User %s n'a pas le rôle 'driver'",
            getattr(user, "username", user.id),
        )
        error_response, status_code = APIErrorHandler.handle_not_found(
            "Driver",
            user_public_id,
            logger,
        )
        return None, error_response, status_code

    driver_repo = DriverRepository()
    driver = driver_repo.find_model_by_user_id(user.id)
    if not driver:
        logger.error("Driver not found for user ID: {user.id}")
        error_response, status_code = APIErrorHandler.handle_not_found(
            "Driver",
            user.id,
            logger,
        )
        return None, error_response, status_code

    logger.info(
        "Driver found: %s for user %s",
        driver.id,
        getattr(user, "username", user.id),
    )
    return driver, None, None


def notify_booking_cancelled(driver_id: int, booking_id: int) -> None:
    room = f"driver_{driver_id}"
    socketio.emit("booking_cancelled", {"id": booking_id}, to=room)


# ---------------------------
# Routes
# ---------------------------


@driver_ns.route("/me/profile")
class DriverProfile(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère le profil du chauffeur."""
        # #region agent log
        try:
            import json
            from datetime import UTC, datetime
            from pathlib import Path

            from flask import (
                current_app,
                request,
            )

            log_path = Path(r"c:\Users\jasiq\atmr\.cursor\debug.log")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "location": "driver.py:DriverProfile.get",
                            "message": "GET /driver/me/profile entry",
                            "data": {
                                "has_access_token_cookie": bool(
                                    request.cookies.get(
                                        current_app.config.get(
                                            "COOKIE_ACCESS_TOKEN_NAME", "access_token"
                                        )
                                    )
                                ),
                                "has_refresh_token_cookie": bool(
                                    request.cookies.get(
                                        current_app.config.get(
                                            "COOKIE_REFRESH_TOKEN_NAME",
                                            "refresh_token",
                                        )
                                    )
                                ),
                                "has_authorization_header": bool(
                                    request.headers.get("Authorization")
                                ),
                                "cookie_names": list(request.cookies.keys()),
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "G",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        result = None
        status_code = 200
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)
                from application.drivers import (
                    GetDriverProfileInput,
                    GetDriverProfileUseCase,
                )

                uc = GetDriverProfileUseCase()
                input_data = GetDriverProfileInput(driver=driver)
                uc_res = uc.execute(input_data)
                if not uc_res.success or not uc_res.response:
                    result = {"error": uc_res.error or "Erreur inconnue"}
                    status_code = uc_res.status_code or 500
                else:
                    result = uc_res.response
                    status_code = uc_res.status_code

                    # ✅ Implémentation : Synchroniser le profil via notification silencieuse
                    # Permet de synchroniser le profil en arrière-plan pour optimisation
                    try:
                        from services.events.fanout import send_profile_sync

                        # Extraire le profil et les stats si disponibles
                        profile_data = result if result else {}
                        stats_data = profile_data.get("stats")  # Stats optionnelles

                        # Envoyer la notification silencieuse (pas de son/vibration)
                        send_profile_sync(
                            driver_id=driver.id,
                            profile=profile_data,
                            stats=stats_data,
                        )
                        logger.debug(
                            "[Driver Profile] Profil synchronisé via notification silencieuse pour driver %s",
                            driver.id,
                        )
                    except Exception as e:
                        # Ne pas faire échouer l'endpoint si la sync échoue
                        logger.debug(
                            "[Driver Profile] Échec sync profil (non-critique): %s",
                            e,
                        )

                    # ✅ Implémentation : Synchroniser la configuration via notification silencieuse
                    # Permet de synchroniser la config de l'app en arrière-plan pour optimisation
                    try:
                        from services.events.fanout import send_config_update
                        from services.events.night_mode import get_night_mode_status

                        # Récupérer le statut du mode nuit
                        night_mode_status = get_night_mode_status()

                        # Construire la configuration de l'app
                        app_config = {
                            "night_mode": {
                                "is_night": night_mode_status.get("is_night", False),
                                "current_time": night_mode_status.get("current_time"),
                                "night_start": night_mode_status.get(
                                    "night_start", "22:00"
                                ),
                                "night_end": night_mode_status.get(
                                    "night_end", "06:00"
                                ),
                            },
                            "notifications": {
                                "enabled": True,  # Les notifications sont toujours activées si l'app est utilisée
                                "critical_alerts": True,  # Alertes critiques toujours activées
                                "silent_updates": True,  # Mises à jour silencieuses activées
                            },
                            "app_preferences": {
                                "language": "fr",  # Langue par défaut
                                "units": "metric",  # Unités métriques
                            },
                        }

                        # Ajouter la config de l'entreprise si disponible
                        # Note: company_id est nullable=False selon le modèle Driver
                        company_id_value = getattr(driver, "company_id", None)
                        if company_id_value:
                            try:
                                from models.company import Company

                                company = Company.query.get(company_id_value)
                                if company:
                                    # Ajouter des infos sur le mode dispatch si pertinent
                                    dispatch_mode = getattr(
                                        company, "dispatch_mode", None
                                    )
                                    if dispatch_mode:
                                        app_config["dispatch"] = {
                                            "mode": dispatch_mode,
                                        }
                            except Exception:
                                # Ne pas bloquer si on ne peut pas récupérer la config entreprise
                                pass

                        # Envoyer la notification silencieuse (pas de son/vibration)
                        send_config_update(driver_id=driver.id, config=app_config)
                        logger.debug(
                            "[Driver Profile] Configuration synchronisée via notification silencieuse pour driver %s",
                            driver.id,
                        )
                    except Exception as e:
                        # Ne pas faire échouer l'endpoint si la sync échoue
                        logger.debug(
                            "[Driver Profile] Échec sync configuration (non-critique): %s",
                            e,
                        )
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation/récupération profil driver: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur lors de la récupération du profil.",
            }
            status_code = 400
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors récupération profil driver: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la récupération du profil.",
            }
            status_code = 500
        except Exception as e:
            logger.exception("❌ Erreur inattendue get_driver_profile")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue lors de la récupération du profil.",
            }
            status_code = 500
        return result, status_code

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(driver_profile_model)
    def put(self):
        """Met à jour le profil du chauffeur."""
        result = None
        status_code = 200
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            result = error_response
        else:
            driver = cast("Driver", driver)

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.driver_schemas import DriverProfileUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    DriverProfileUpdateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            logger.info("Payload reçu pour mise à jour du profil: %s", validated_data)
            from application.drivers.update_driver_profile import (
                UpdateDriverProfileUseCase,
            )

            uc = UpdateDriverProfileUseCase()
            try:
                uc_res = uc.execute(driver=driver, validated_data=validated_data)
                result = uc_res.response
                status_code = uc_res.status_code

                if uc_res.should_commit:
                    db.session.commit()
            except (ValueError, TypeError, AttributeError) as e:
                db.session.rollback()
                logger.warning(
                    "❌ Erreur validation lors mise à jour profil driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                result = {
                    "error": "validation_error",
                    "message": "Erreur de validation lors de la mise à jour du profil.",
                }
                status_code = 400
            except IntegrityError as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur contrainte DB lors mise à jour profil driver %s: %s",
                    driver.id,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result, status_code = format_integrity_error(e)
            except (OperationalError, SQLAlchemyError) as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur DB lors mise à jour profil driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "database_error",
                    "message": "Erreur de base de données lors de la mise à jour du profil.",
                }
                status_code = 500
            except Exception as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur inattendue update_driver_profile (driver_id=%s)",
                    driver.id,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "internal_error",
                    "message": "Une erreur interne est survenue.",
                }
                status_code = 500
        return result, status_code


@driver_ns.route("/me/photo")
class DriverPhoto(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(photo_model)
    def put(self):
        """Met à jour la photo du chauffeur."""
        result = None
        status_code = 200
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            result = error_response
        else:
            driver = cast("Driver", driver)

            payload = cast("dict[str, Any] | None", request.get_json())
            logger.info("Payload reçu pour mise à jour de la photo: %s", payload)

            from application.drivers.update_driver_photo import UpdateDriverPhotoUseCase
            from infrastructure.files.file_validation import validate_uploaded_image

            uc = UpdateDriverPhotoUseCase(
                validate_uploaded_image_fn=validate_uploaded_image
            )

            try:
                uc_res = uc.execute(driver=driver, payload=payload)
                result = uc_res.response
                status_code = uc_res.status_code
                if uc_res.should_commit:
                    db.session.commit()
            except (ValueError, TypeError, AttributeError) as e:
                db.session.rollback()
                logger.warning(
                    "❌ Erreur validation lors mise à jour photo driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                result = {
                    "error": "validation_error",
                    "message": "Erreur de validation lors de la mise à jour de la photo.",
                }
                status_code = 400
            except IntegrityError as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur contrainte DB lors mise à jour photo driver %s: %s",
                    driver.id,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "database_constraint_error",
                    "message": "Erreur de contrainte de base de données.",
                }
                status_code = 500
            except (OperationalError, SQLAlchemyError) as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur DB lors mise à jour photo driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "database_error",
                    "message": "Erreur de base de données lors de la mise à jour de la photo.",
                }
                status_code = 500
            except Exception as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur inattendue update_driver_photo (driver_id=%s)",
                    driver.id,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "internal_error",
                    "message": "Une erreur interne est survenue.",
                }
                status_code = 500
        return result, status_code


@driver_ns.route("/me/bookings")
class DriverUpcomingBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        # 🔍 LOG : Vérifier quel chauffeur charge ses missions
        driver_name = (
            f"{driver.user.first_name} {driver.user.last_name}"
            if driver.user
            else f"#{driver.id}"
        )
        logger.info(
            "📱 [Driver Bookings] Driver %s (ID: %s) loading bookings",
            driver_name,
            driver.id,
        )

        from application.drivers.get_driver_upcoming_bookings import (
            GetDriverUpcomingBookingsUseCase,
        )
        from repositories.booking_repository import BookingRepository
        from shared.time_utils import day_local_bounds, now_local

        uc = GetDriverUpcomingBookingsUseCase(
            booking_repo=BookingRepository(),
            day_local_bounds_fn=day_local_bounds,
            now_local_fn=now_local,
        )
        bookings = uc.execute(driver_id=driver.id).bookings

        # 🔍 LOG : Afficher les courses trouvées
        logger.info(
            "📱 [Driver Bookings] Found %s bookings for driver %s (ID: %s)",
            len(bookings),
            driver_name,
            driver.id,
        )
        for b in bookings:
            logger.info(
                "   - Booking #%s: driver_id=%s, client=%s, time=%s",
                b.id,
                b.driver_id,
                b.customer_name,
                b.scheduled_time,
            )

        # ✅ Implémentation : Précharger les missions via notification silencieuse
        # Permet de synchroniser les missions en arrière-plan pour optimisation
        try:
            from services.events.fanout import send_missions_preload

            # Sérialiser les bookings pour la notification silencieuse
            missions_data = [
                b.serialize if hasattr(b, "serialize") else {"id": b.id}
                for b in bookings
            ]

            # Envoyer la notification silencieuse (pas de son/vibration)
            send_missions_preload(driver_id=driver.id, missions=missions_data)
            logger.debug(
                "[Driver Bookings] Missions préchargées via notification silencieuse pour driver %s",
                driver.id,
            )
        except Exception as e:
            # Ne pas faire échouer l'endpoint si le préchargement échoue
            logger.debug(
                "[Driver Bookings] Échec préchargement missions (non-critique): %s",
                e,
            )

        # ✅ Implémentation : Précharger les cartes (routes) via notification silencieuse
        # Permet de précharger les itinéraires pour optimisation navigation
        try:
            import os

            from services.events.fanout import send_maps_precache
            from services.geolocation.osrm import route_info

            routes_data = []
            osrm_base_url = os.getenv("UD_OSRM_BASE_URL", "http://osrm:5000")

            # Calculer les routes pour chaque booking avec pickup et dropoff
            for booking in bookings:
                pickup_lat = getattr(booking, "pickup_lat", None)
                pickup_lon = getattr(booking, "pickup_lon", None)
                dropoff_lat = getattr(booking, "dropoff_lat", None)
                dropoff_lon = getattr(booking, "dropoff_lon", None)

                # Si les coordonnées sont disponibles, calculer la route
                if all(
                    [
                        pickup_lat is not None,
                        pickup_lon is not None,
                        dropoff_lat is not None,
                        dropoff_lon is not None,
                    ]
                ):
                    try:
                        # Convertir en float une seule fois avec vérification explicite
                        pickup_lat_float = float(pickup_lat)  # type: ignore[arg-type]
                        pickup_lon_float = float(pickup_lon)  # type: ignore[arg-type]
                        dropoff_lat_float = float(dropoff_lat)  # type: ignore[arg-type]
                        dropoff_lon_float = float(dropoff_lon)  # type: ignore[arg-type]

                        # Calculer la route (avec timeout court pour ne pas ralentir l'endpoint)
                        route_result = route_info(
                            origin=(pickup_lat_float, pickup_lon_float),
                            destination=(dropoff_lat_float, dropoff_lon_float),
                            base_url=osrm_base_url,
                            profile="driving",
                            timeout=2,  # Timeout court pour ne pas bloquer
                            overview="simplified",  # Geometry simplifiée pour préchargement
                            geometries="geojson",
                        )

                        # Extraire les données essentielles de la route
                        routes_data.append(
                            {
                                "booking_id": booking.id,
                                "pickup": {
                                    "lat": pickup_lat_float,
                                    "lon": pickup_lon_float,
                                },
                                "dropoff": {
                                    "lat": dropoff_lat_float,
                                    "lon": dropoff_lon_float,
                                },
                                "duration_seconds": route_result.get("duration", 0),
                                "distance_meters": route_result.get("distance", 0),
                                "geometry": route_result.get(
                                    "geometry"
                                ),  # GeoJSON geometry
                            }
                        )
                    except Exception as route_error:
                        # Ne pas bloquer si une route échoue
                        logger.debug(
                            "[Driver Bookings] Échec calcul route pour booking %s (non-critique): %s",
                            booking.id,
                            route_error,
                        )

            # Envoyer la notification silencieuse si on a des routes
            if routes_data:
                send_maps_precache(driver_id=driver.id, routes=routes_data)
                logger.debug(
                    "[Driver Bookings] Cartes préchargées via notification silencieuse pour driver %s (%s routes)",
                    driver.id,
                    len(routes_data),
                )
        except Exception as e:
            # Ne pas faire échouer l'endpoint si le préchargement échoue
            logger.debug(
                "[Driver Bookings] Échec préchargement cartes (non-critique): %s",
                e,
            )

        return [b.serialize for b in bookings], 200


@driver_ns.route("/me/bookings/since")
class DriverBookingsSince(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupérer les bookings du chauffeur modifiés depuis un timestamp donné.

        Args (query params):
            since: Timestamp ISO 8601 (optionnel). Si fourni, retourne uniquement
                   les bookings avec updated_at >= since. Sinon, retourne tous les bookings.

        Returns:
            Liste des bookings (sérialisées)
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from application.drivers.get_driver_upcoming_bookings import (
            GetDriverUpcomingBookingsUseCase,
        )
        from repositories.booking_repository import BookingRepository
        from shared.time_utils import day_local_bounds, now_local

        # Récupérer le paramètre since (optionnel)
        since_str = request.args.get("since")

        if since_str:
            # Parser le timestamp ISO
            try:
                # Gérer les formats avec et sans Z
                if since_str.endswith("Z"):
                    since_str = since_str.replace("Z", "+00:00")
                since_dt = datetime.fromisoformat(since_str)
                # S'assurer que c'est en UTC
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=UTC)
                else:
                    since_dt = since_dt.astimezone(UTC)
            except (ValueError, AttributeError) as e:
                logger.warning(
                    "Invalid 'since' timestamp format: %s, error: %s", since_str, e
                )
                return APIErrorHandler.handle_error(
                    f"Format de timestamp invalide: {since_str}. Utilisez le format ISO 8601.",
                    400,
                )

            # Filtrer par updated_at >= since ET statuts appropriés
            from models.booking import Booking
            from models.enums import BookingStatus

            bookings = (
                Booking.query.filter(Booking.driver_id == driver.id)
                .filter(Booking.updated_at >= since_dt)
                .filter(
                    Booking.status.in_(
                        [
                            BookingStatus.ASSIGNED,
                            BookingStatus.EN_ROUTE,
                            BookingStatus.IN_PROGRESS,
                        ]
                    )
                )
                .order_by(Booking.updated_at.asc())
                .all()
            )

            logger.info(
                "📱 [Driver Bookings Since] Driver %s (ID: %s) - Found %s bookings since %s",
                driver.id,
                driver.id,
                len(bookings),
                since_dt.isoformat(),
            )
        else:
            # Pas de filtre since : utiliser le use case existant (comportement par défaut)
            uc = GetDriverUpcomingBookingsUseCase(
                booking_repo=BookingRepository(),
                day_local_bounds_fn=day_local_bounds,
                now_local_fn=now_local,
            )
            bookings = uc.execute(driver_id=driver.id).bookings

            logger.warning(
                "📱 [Driver Bookings Since] Driver %s (ID: %s) - No 'since' param, returning all upcoming bookings (%s)",
                driver.id,
                driver.id,
                len(bookings),
            )
            # 🔍 LOG DÉTAILLÉ : Afficher chaque booking trouvé
            for b in bookings:
                logger.warning(
                    "   - Booking #%s: driver_id=%s, status=%s, scheduled_time=%s",
                    b.id,
                    b.driver_id,
                    b.status,
                    b.scheduled_time,
                )

        return [b.serialize for b in bookings], 200


@driver_ns.route("/me/bookings/eta")
class DriverBookingsETA(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Calcule l'ETA dynamique pour toutes les missions du chauffeur
        basé sur sa position GPS actuelle."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from datetime import date

        from application.drivers.get_driver_bookings_eta import (
            GetDriverBookingsETAUseCase,
        )
        from infrastructure.dispatch.eta_calculator import get_eta_seconds_fn
        from shared.time_utils import day_local_bounds, now_local

        # Récupérer les courses d'aujourd'hui (non terminées)
        today_start, today_end = day_local_bounds(date.today().strftime("%Y-%m-%d"))

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        bookings = booking_repo.find_models_by_driver_with_statuses_and_time_range(
            driver_id=driver.id,
            statuses=[
                BookingStatus.ASSIGNED,
                BookingStatus.EN_ROUTE,
                BookingStatus.IN_PROGRESS,
            ],
            start_time=today_start,
            end_time=today_end,
        )

        # Position actuelle du chauffeur
        driver_lat = getattr(driver, "latitude", None)
        driver_lon = getattr(driver, "longitude", None)

        uc = GetDriverBookingsETAUseCase(
            eta_seconds_fn=get_eta_seconds_fn(),
            now_local_fn=now_local,
        )
        resp = uc.execute(
            driver_lat=float(driver_lat) if driver_lat is not None else None,
            driver_lon=float(driver_lon) if driver_lon is not None else None,
            bookings=bookings,
        )

        if not resp.has_gps:
            return {
                "has_gps": False,
                "bookings": [
                    {
                        "id": b.id,
                        "duration_seconds": b.duration_seconds,
                        "distance_meters": b.distance_meters,
                        "eta_to_dropoff_seconds": None,
                        "estimated_arrival_dropoff": None,
                    }
                    for b in bookings
                ],
            }, 200

        return {
            "has_gps": True,
            "driver_position": resp.driver_position,
            "bookings": [
                {
                    "id": item.id,
                    "eta_to_pickup_seconds": item.eta_to_pickup_seconds,
                    "eta_to_dropoff_seconds": item.eta_to_dropoff_seconds,
                    "duration_seconds": item.duration_seconds,
                    "distance_meters": item.distance_meters,
                    "estimated_arrival": item.estimated_arrival,
                    "estimated_arrival_dropoff": item.estimated_arrival_dropoff,
                }
                for item in resp.bookings
            ],
        }, 200


@driver_ns.route("/me/location")
class DriverLocation(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(location_model)
    def put(self):
        """Tracking temps réel : enregistre la dernière position."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        # Variables pour stocker le résultat
        result = None
        status_code = 200

        try:
            p = request.get_json(force=True)
            logger.debug("📍 Received location data: %s (type=%s)", p, type(p))

            if not p:
                result = {"error": "No data provided"}
                status_code = 400
            elif "latitude" not in p or "longitude" not in p:
                result = {"error": "Latitude and longitude are required"}
                status_code = 400
            else:
                # Validation et conversion
                try:
                    lat = float(p["latitude"])
                    lon = float(p["longitude"])

                    if result is None and (
                        (not (-LAT_THRESHOLD <= lat <= LAT_THRESHOLD))
                        or not (-LON_THRESHOLD <= lon <= LON_THRESHOLD)
                    ):
                        result = {"error": "Coordinates out of valid range"}
                        status_code = 400

                    if result is None:
                        speed = float(p.get("speed", 0.0) or 0.0)
                        heading = float(p.get("heading", 0.0) or 0.0)
                        accuracy = float(p.get("accuracy", 0.0) or 0.0)
                        ts = p.get("ts") or datetime.now(UTC).isoformat()

                        from application.drivers.update_driver_location import (
                            UpdateDriverLocationCommand,
                            UpdateDriverLocationUseCase,
                        )
                        from drivers.infrastructure.adapters.location_adapter import (
                            create_location_update_fn,
                        )

                        try:
                            # ✅ DDD: Utilise adapter au lieu de service directement
                            uc = UpdateDriverLocationUseCase(
                                update_location_fn=create_location_update_fn()
                            )
                            uc_result = uc.execute(
                                UpdateDriverLocationCommand(
                                    driver_id=driver.id,
                                    latitude=lat,
                                    longitude=lon,
                                    speed=speed if speed > 0 else None,
                                    heading=heading if heading >= 0 else None,
                                    accuracy=accuracy if accuracy > 0 else None,
                                    ts=ts,
                                )
                            )

                            # Utiliser position snapée
                            lat = uc_result.snapped_lat
                            lon = uc_result.snapped_lon
                            source = uc_result.source

                            # Émettre events geofencing si détectés
                            for event in uc_result.geofence_events:
                                if event == "arrived_at_pickup":
                                    socketio.emit(
                                        "driver:arrived_at_pickup",
                                        {"driver_id": driver.id},
                                        to=f"company_{driver.company_id}",
                                    )
                                elif event == "arrived_at_dropoff":
                                    socketio.emit(
                                        "driver:arrived_at_dropoff",
                                        {"driver_id": driver.id},
                                        to=f"company_{driver.company_id}",
                                    )

                        except Exception as e_loc:
                            logger.warning(
                                "[LocationService] HTTP location update failed: %s",
                                str(e_loc),
                            )
                            source = "raw"  # Fallback

                        # 5) Diffusion temps réel à la room entreprise
                        try:
                            room = f"company_{driver.company_id}"

                            # Extraire first_name et last_name depuis driver.user
                            first_name = None
                            last_name = None
                            if hasattr(driver, "user") and driver.user is not None:
                                first_name = getattr(driver.user, "first_name", None)
                                last_name = getattr(driver.user, "last_name", None)

                            # ✅ FIX: Émettre "driver_location_update" pour correspondre au frontend
                            socketio.emit(
                                "driver_location_update",
                                {
                                    "driver_id": driver.id,
                                    "company_id": driver.company_id,
                                    "lat": lat,
                                    "lon": lon,
                                    "speed": speed,
                                    "heading": heading,
                                    "accuracy": accuracy,
                                    "ts": ts,
                                    "source": source,
                                    "first_name": first_name,
                                    "last_name": last_name,
                                },
                                to=room,
                            )
                        except Exception:
                            pass

                        result = {
                            "ok": True,
                            "source": source,
                            "message": "Location updated",
                        }
                except (ValueError, TypeError):
                    result = {"error": "Invalid coordinate format"}
                    status_code = 400

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Unexpected error in location update: %s", e)
            logger.error("❌ Request data: %s", request.get_data())
            result = {"error": f"Internal error: {e!s}"}
            status_code = 500

        return result, status_code


@driver_ns.route("/me/accident")
class ReportAccident(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Signale un accident pour le chauffeur connecté.

        Le chauffeur peut signaler manuellement un accident depuis l'application mobile.
        Cela déclenche une alerte critique à l'entreprise et une notification push.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        try:
            p = request.get_json(force=True) or {}

            # Données optionnelles sur l'accident
            accident_details = {
                "driver_id": driver.id,
                "company_id": driver.company_id,
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
                "timestamp": datetime.now(UTC).isoformat(),
                "manual_report": True,  # Signalement manuel
                "description": p.get("description"),
                "severity": p.get("severity", "unknown"),
            }

            # ✅ Implémentation : Envoyer l'alerte d'accident
            from services.events.fanout import send_accident_alert

            success = send_accident_alert(
                driver_id=driver.id,
                accident_details=accident_details,
            )

            if success:
                logger.warning(
                    "[ReportAccident] 🚨 Accident signalé par driver %s (company %s)",
                    driver.id,
                    driver.company_id,
                )
                return {"message": "Accident signalé avec succès", "ok": True}, 200

            logger.error(
                "[ReportAccident] Échec envoi alerte accident pour driver %s",
                driver.id,
            )
            return {
                "error": "Erreur lors de l'envoi de l'alerte",
                "ok": False,
            }, 500

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Erreur lors du signalement d'accident: %s", e)
            return {"error": f"Erreur interne: {e!s}", "ok": False}, 500


@driver_ns.route("/me/medical-emergency")
class ReportMedicalEmergency(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Signale une urgence médicale pour le chauffeur connecté.

        Le chauffeur peut signaler manuellement une urgence médicale (passager ou lui-même)
        depuis l'application mobile. Cela déclenche une alerte critique à l'entreprise
        et une notification push.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        try:
            p = request.get_json(force=True) or {}

            # Données optionnelles sur l'urgence médicale
            emergency_details = {
                "driver_id": driver.id,
                "company_id": driver.company_id,
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
                "timestamp": datetime.now(UTC).isoformat(),
                "manual_report": True,  # Signalement manuel
                "description": p.get("description"),
                "severity": p.get("severity", "high"),
                "affected_person": p.get(
                    "affected_person", "passenger"
                ),  # "driver" ou "passenger"
                "symptoms": p.get("symptoms"),  # Liste de symptômes
            }

            # ✅ Implémentation : Envoyer l'alerte d'urgence médicale
            from services.events.fanout import send_medical_emergency_alert

            success = send_medical_emergency_alert(
                driver_id=driver.id,
                emergency_details=emergency_details,
            )

            if success:
                logger.warning(
                    "[ReportMedicalEmergency] 🚨 Urgence médicale signalée par driver %s (company %s)",
                    driver.id,
                    driver.company_id,
                )
                return {
                    "message": "Urgence médicale signalée avec succès",
                    "ok": True,
                }, 200

            logger.error(
                "[ReportMedicalEmergency] Échec envoi alerte urgence médicale pour driver %s",
                driver.id,
            )
            return {
                "error": "Erreur lors de l'envoi de l'alerte",
                "ok": False,
            }, 500

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Erreur lors du signalement d'urgence médicale: %s", e)
            return {"error": f"Erreur interne: {e!s}", "ok": False}, 500


@driver_ns.route("/me/security-zone")
class ReportSecurityZone(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Signale une entrée dans une zone dangereuse pour le chauffeur connecté.

        Le chauffeur peut signaler manuellement qu'il entre dans une zone à risque
        depuis l'application mobile. Cela déclenche une alerte critique à l'entreprise
        et une notification push.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        try:
            p = request.get_json(force=True) or {}

            # Données optionnelles sur la zone dangereuse
            zone_details = {
                "driver_id": driver.id,
                "company_id": driver.company_id,
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
                "timestamp": datetime.now(UTC).isoformat(),
                "manual_report": True,  # Signalement manuel
                "zone_name": p.get("zone_name", "Zone dangereuse"),
                "zone_type": p.get(
                    "zone_type", "unknown"
                ),  # "high_crime", "no_service", etc.
                "risk_level": p.get(
                    "risk_level", "high"
                ),  # "low", "medium", "high", "critical"
                "description": p.get("description"),
            }

            # ✅ Implémentation : Envoyer l'alerte zone dangereuse
            from services.events.fanout import send_security_zone_alert

            success = send_security_zone_alert(
                driver_id=driver.id,
                zone_details=zone_details,
            )

            if success:
                logger.warning(
                    "[ReportSecurityZone] 🚨 Zone dangereuse signalée par driver %s (company %s) - zone: %s",
                    driver.id,
                    driver.company_id,
                    zone_details.get("zone_name"),
                )
                return {
                    "message": "Alerte zone dangereuse signalée avec succès",
                    "ok": True,
                }, 200

            logger.error(
                "[ReportSecurityZone] Échec envoi alerte zone dangereuse pour driver %s",
                driver.id,
            )
            return {
                "error": "Erreur lors de l'envoi de l'alerte",
                "ok": False,
            }, 500

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Erreur lors du signalement de zone dangereuse: %s", e)
            return {"error": f"Erreur interne: {e!s}", "ok": False}, 500


@driver_ns.route("/me/bookings/<int:booking_id>")
class BookingDetails(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                return error_response, status_code
            driver = cast("Driver", driver)

            from application.drivers.get_driver_booking_details import (
                GetDriverBookingDetailsUseCase,
            )
            from repositories.booking_repository import BookingRepository

            uc = GetDriverBookingDetailsUseCase(booking_repo=BookingRepository())  # type: ignore[reportArgumentType]
            result = uc.execute(booking_id=booking_id, driver_id=driver.id)
            if result is None:
                return APIErrorHandler.handle_not_found(
                    "Booking",
                    booking_id,
                    logger,
                )

            return result.payload, 200
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors récupération détails booking %s: %s - %s",
                booking_id,
                type(e).__name__,
                e,
            )
            return {
                "error": "validation_error",
                "message": "Erreur de validation lors de la récupération des détails de la réservation.",
            }, 400
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors récupération détails booking %s: %s - %s",
                booking_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            return {
                "error": "database_error",
                "message": "Erreur de base de données lors de la récupération des détails.",
            }, 500
        except Exception as e:
            logger.exception(
                "❌ Erreur inattendue get_booking_details (booking_id=%s)", booking_id
            )
            sentry_sdk.capture_exception(e)
            return {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }, 500


@driver_ns.route("/company/<int:company_id>/live-locations")
class CompanyLiveLocations(Resource):
    @jwt_required()
    def get(self, company_id: int):
        """Retourne la dernière position connue
        de tous les chauffeurs de l'entreprise."""
        try:
            from application.drivers.get_company_drivers_live_locations import (
                GetCompanyDriversLiveLocationsUseCase,
            )
            from infrastructure.persistence.drivers.redis_driver_location_store import (
                get_driver_last_location,
            )
            from repositories.driver_repository import DriverRepository

            uc = GetCompanyDriversLiveLocationsUseCase(
                driver_repo=DriverRepository(),  # type: ignore[reportArgumentType]
                get_last_location_fn=get_driver_last_location,
            )
            uc_res = uc.execute(company_id=company_id)
            return uc_res.response, uc_res.status_code
        except (ValueError, TypeError) as e:
            logger.warning(
                "❌ Erreur validation lors récupération locations company %s: %s - %s",
                company_id,
                type(e).__name__,
                e,
            )
            return {"items": []}, 200
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors récupération locations company %s: %s - %s",
                company_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            return {"items": []}, 200
        except Exception as e:
            logger.exception(
                "❌ Erreur inattendue get_location_history (company_id=%s)", company_id
            )
            sentry_sdk.capture_exception(e)
            return {"items": []}, 200


@driver_ns.route("/me/bookings/<int:booking_id>/status", methods=["PUT", "OPTIONS"])
class UpdateBookingStatus(Resource):
    def options(self, booking_id: int):  # noqa: ARG002
        """Gère les requêtes CORS preflight (OPTIONS)."""
        return {}, 200

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(booking_status_model)
    def put(self, booking_id: int):
        # Variables pour stocker le résultat
        result = None
        status_code = 200

        data = request.get_json()
        logger.info("📦 Body reçu pour status update booking %s: %s", booking_id, data)
        logger.info("📦 Type de data: %s", type(data))

        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                logger.error("Driver not found for token: %s", get_jwt_identity())
                result = error_response
            else:
                driver = cast("Driver", driver)

                from repositories.booking_repository import BookingRepository

                booking_repo = BookingRepository()
                booking = booking_repo.find_model_by_id(booking_id=booking_id)

                # ✅ Debug logs pour diagnostiquer le problème
                if booking:
                    logger.info(
                        "📦 Booking trouvé: id=%s, company_id=%s, executing_company_id=%s, driver_id=%s, status=%s",
                        booking.id,
                        booking.company_id,
                        booking.executing_company_id,
                        booking.driver_id,
                        booking.status,
                    )
                    logger.info(
                        "🚗 Driver: id=%s, company_id=%s", driver.id, driver.company_id
                    )

                if not booking:
                    logger.error("Booking with id %s not found", booking_id)
                    result = {"error": "Booking not found"}
                    status_code = 404
                else:
                    # ✅ Extraire les valeurs pour éviter les problèmes de type SQLAlchemy
                    # Ces valeurs sont déjà des entiers Python une fois le modèle chargé
                    executing_company_id = booking.executing_company_id
                    booking_company_id = cast(int, booking.company_id)
                    driver_company_id = cast(int, driver.company_id)

                    # ✅ Vérifier que le chauffeur appartient à l'entreprise qui exécute (originale OU transférée)
                    has_executing_company = executing_company_id is not None
                    if (
                        has_executing_company
                        and driver_company_id != executing_company_id
                    ):
                        logger.error(
                            "❌ Chauffeur (company_id=%s) non autorisé pour booking exécuté par company_id=%s",
                            driver_company_id,
                            executing_company_id,
                        )
                        result = {
                            "error": "Ce chauffeur n'appartient pas à l'entreprise qui exécute cette course"
                        }
                        status_code = 403
                    elif (
                        not has_executing_company
                        and driver_company_id != booking_company_id
                    ):
                        logger.error(
                            "❌ Chauffeur (company_id=%s) non autorisé pour booking créé par company_id=%s",
                            driver_company_id,
                            booking_company_id,
                        )
                        result = {
                            "error": "Ce chauffeur n'appartient pas à l'entreprise de cette course"
                        }
                        status_code = 403
                    elif (
                        booking.driver_id is None
                        and booking.status == BookingStatus.PENDING
                    ):
                        booking.driver_id = driver.id
                    elif booking.driver_id != driver.id:
                        logger.error(
                            "❌ Chauffeur %s (id=%s) essaie de modifier booking assigné à driver_id=%s",
                            driver.user.username if driver.user else "Unknown",
                            driver.id,
                            booking.driver_id,
                        )
                        result = {
                            "error": "Cette course est assignée à un autre chauffeur"
                        }
                        status_code = 403
                    elif not data:
                        result = {"error": "Missing JSON payload"}
                        status_code = 400
                    else:
                        from application.drivers.update_driver_booking_status import (
                            UpdateDriverBookingStatusCommand,
                            UpdateDriverBookingStatusUseCase,
                        )
                        from repositories.assignment_repository import (
                            AssignmentRepository,
                        )
                        # ✅ DDD: Plus besoin d'importer emit_assignment_cancelled, le use-case publie un événement

                        # Helper pour fallback si événement échoue
                        def _emit_assignment_cancelled_fallback(
                            company_id: int,
                            assignment_id: str,
                            booking_id: int,
                            driver_id: int,
                        ) -> None:
                            """Fallback pour notification directe si événement échoue."""
                            from services.realtime.socketio import (
                                emit_assignment_cancelled,
                            )

                            emit_assignment_cancelled(
                                company_id=company_id,
                                assignment_id=assignment_id,
                                booking_id=booking_id,
                                driver_id=driver_id,
                            )

                        uc = UpdateDriverBookingStatusUseCase(
                            booking_repo=booking_repo,
                            assignment_repo=AssignmentRepository(),
                            db_session=db.session,
                            notify_booking_update_fn=notify_booking_update,
                            resolve_delays_fn=DelayEvent.resolve_delays_for_booking,
                            emit_assignment_cancelled_fn=_emit_assignment_cancelled_fallback,
                            maybe_trigger_dispatch_fn=_maybe_trigger_dispatch,
                        )
                        uc_res = uc.execute(
                            UpdateDriverBookingStatusCommand(
                                booking_id=booking_id,
                                driver_id=driver.id,
                                payload=cast("dict[str, Any] | None", data),
                            )
                        )
                        result = uc_res.response
                        status_code = uc_res.status_code

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors mise à jour statut booking: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de la mise à jour du statut.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur contrainte DB lors mise à jour statut booking: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            logger.exception(
                "❌ Erreur DB lors mise à jour statut booking: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la mise à jour du statut.",
            }
            status_code = 500
        except Exception as e:
            logger.exception("❌ Erreur inattendue update_booking_status")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500

        return result, status_code


@driver_ns.route("/me/bookings/<int:booking_id>")
class RejectBooking(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def delete(self, booking_id: int):
        """Réjette une réservation assignée."""
        result = None
        status_code = 200
        driver = None
        try:
            from repositories.driver_repository import DriverRepository

            current_user_id = get_jwt_identity()
            driver_repo = DriverRepository()
            driver = driver_repo.find_model_by_user_id(user_id=current_user_id)
            if not driver:
                result = {"error": "Unauthorized: Driver not found"}
                status_code = 403
            else:
                from http import HTTPStatus

                from application.drivers.reject_driver_booking import (
                    RejectDriverBookingUseCase,
                )
                from repositories.booking_repository import BookingRepository

                uc = RejectDriverBookingUseCase(booking_repo=BookingRepository())
                uc_res = uc.execute(booking_id=booking_id, driver_id=int(driver.id))
                result = uc_res.response
                status_code = uc_res.status_code

                if uc_res.should_commit and status_code == HTTPStatus.OK:
                    db.session.commit()
                    if uc_res.booking is not None:
                        # ✅ Clean Architecture: Publier événement au lieu d'appel direct
                        try:
                            from application.events.event_bus import publish_event
                            from domain.events.events import BookingCancelledEvent

                            publish_event(
                                BookingCancelledEvent(
                                    booking_id=uc_res.booking.id,
                                    driver_id=driver.id,
                                    company_id=uc_res.booking.company_id,
                                )
                            )
                        except Exception as e:
                            # Fallback vers notification directe si événement échoue
                            logger.warning(
                                "[RejectBooking] Event publish failed, using direct notification: %s",
                                e,
                            )
                            notify_booking_cancelled(driver.id, uc_res.booking.id)
        except (ValueError, TypeError, AttributeError) as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.warning(
                "❌ Erreur validation lors rejet booking %s par driver %s: %s - %s",
                booking_id,
                driver_id,
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors du rejet de la réservation.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.exception(
                "❌ Erreur contrainte DB lors rejet booking %s par driver %s: %s",
                booking_id,
                driver_id,
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.exception(
                "❌ Erreur DB lors rejet booking %s par driver %s: %s - %s",
                booking_id,
                driver_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors du rejet de la réservation.",
            }
            status_code = 500
        except Exception as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.exception(
                "❌ Erreur inattendue reject_booking (booking_id=%s, driver_id=%s)",
                booking_id,
                driver_id,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/me/availability")
class UpdateAvailability(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(availability_model)
    def put(self):
        """Met à jour la disponibilité du chauffeur."""
        result = None
        status_code = 200
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)

                from application.drivers.update_driver_availability import (
                    UpdateDriverAvailabilityUseCase,
                )

                uc = UpdateDriverAvailabilityUseCase()
                uc_res = uc.execute(
                    driver=driver,
                    payload=cast("dict[str, Any] | None", request.get_json()),
                )
                result = uc_res.response
                status_code = uc_res.status_code

                if uc_res.should_commit:
                    db.session.commit()
        except (ValueError, TypeError, AttributeError) as e:
            db.session.rollback()
            logger.warning(
                "❌ Erreur validation lors mise à jour disponibilité driver: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de la mise à jour de la disponibilité.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur contrainte DB lors mise à jour disponibilité driver: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur DB lors mise à jour disponibilité driver: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la mise à jour de la disponibilité.",
            }
            status_code = 500
        except Exception as e:
            db.session.rollback()
            logger.exception("❌ Erreur inattendue update_availability")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/me/bookings/all")
class DriverAllBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère toutes les réservations assignées au chauffeur."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from application.drivers.get_driver_all_bookings import (
            GetDriverAllBookingsUseCase,
        )
        from repositories.booking_repository import BookingRepository

        uc = GetDriverAllBookingsUseCase(booking_repo=BookingRepository())
        bookings = uc.execute(driver_id=driver.id).bookings
        # ✅ Retourner une liste vide au lieu d'une erreur 404
        return [b.serialize for b in bookings], 200


@driver_ns.route("/me/bookings/<int:booking_id>/report")
class ReportBookingIssue(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        result = None
        status_code = 200
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            result = error_response
        else:
            driver = cast("Driver", driver)

            from application.drivers.report_driver_booking_issue import (
                ReportDriverBookingIssueUseCase,
            )
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            uc = ReportDriverBookingIssueUseCase(booking_repo=booking_repo)
            uc_res = uc.execute(
                booking_id=booking_id,
                driver_id=driver.id,
                payload=cast("dict[str, Any] | None", request.get_json()),
            )
            result = uc_res.response
            status_code = uc_res.status_code

            from http import HTTPStatus

            if status_code == HTTPStatus.OK:
                try:
                    db.session.commit()
                except (ValueError, TypeError, AttributeError) as e:
                    db.session.rollback()
                    logger.warning(
                        "❌ Erreur validation lors report issue booking %s: %s - %s",
                        booking_id,
                        type(e).__name__,
                        e,
                    )
                    result = {
                        "error": "validation_error",
                        "message": "Erreur de validation lors du signalement du problème.",
                    }
                    status_code = 400
                except IntegrityError as e:
                    db.session.rollback()
                    logger.exception(
                        "❌ Erreur contrainte DB lors report issue booking %s: %s",
                        booking_id,
                        e,
                    )
                    sentry_sdk.capture_exception(e)
                    result, status_code = format_integrity_error(e)
                except (OperationalError, SQLAlchemyError) as e:
                    db.session.rollback()
                    logger.exception(
                        "❌ Erreur DB lors report issue booking %s: %s - %s",
                        booking_id,
                        type(e).__name__,
                        e,
                    )
                    sentry_sdk.capture_exception(e)
                    result = {
                        "error": "database_error",
                        "message": "Erreur de base de données lors du signalement du problème.",
                    }
                    status_code = 500
                except Exception as e:
                    db.session.rollback()
                    logger.exception(
                        "❌ Erreur inattendue report_issue (booking_id=%s)",
                        booking_id,
                    )
                    sentry_sdk.capture_exception(e)
                    result = {
                        "error": "internal_error",
                        "message": "Une erreur interne est survenue.",
                    }
                    status_code = 500
        return result, status_code


@driver_ns.route("/save-push-token")
class SavePushToken(Resource):
    @jwt_required()
    def post(self):
        # Variables pour stocker le résultat
        result = None
        status_code = 200

        try:
            # Log & typage strict
            payload_raw = request.get_json(force=True) or {}
            logger.info("[push-token] payload=%s", payload_raw)
            payload: dict[str, Any] = tcast("dict[str, Any]", payload_raw)

            from http import HTTPStatus

            from application.drivers.save_driver_push_token import (
                SaveDriverPushTokenUseCase,
            )
            from repositories.driver_repository import DriverRepository
            from repositories.user_repository import UserRepository

            uc = SaveDriverPushTokenUseCase(
                user_repo=UserRepository(),
                driver_repo=DriverRepository(),
            )
            uc_res = uc.execute(payload=payload, jwt_identity=get_jwt_identity())
            result = uc_res.response
            status_code = uc_res.status_code

            if uc_res.should_commit and status_code == HTTPStatus.OK:
                db.session.commit()

        except (ValueError, TypeError, AttributeError) as e:
            db.session.rollback()
            logger.warning(
                "[push-token] ❌ Erreur validation: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de l'enregistrement du token.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "[push-token] ❌ Erreur contrainte DB: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            db.session.rollback()
            logger.exception(
                "[push-token] ❌ Erreur DB: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de l'enregistrement du token.",
            }
            status_code = 500
        except Exception as e:
            db.session.rollback()
            logger.exception("[push-token] ❌ Erreur inattendue")
            sentry_sdk.capture_exception(e)
            traceback.print_exc()
            result = {
                "error": "internal_error",
                "message": f"Erreur serveur : {e!s}",
            }
            status_code = 500

        return result, status_code


@driver_ns.route("/me/push-privacy")
class DriverPushPrivacy(Resource):
    """Réglage mode discret push pour le chauffeur (pas de nom client sur lockscreen).

    Contrôle d'accès / multi-tenant :
    - @jwt_required() + @role_required(UserRole.driver).
    - get_driver_from_token() détermine le driver (donc le user = driver.user_id).
    - user = User.query.get(driver.user_id) : aucun user_id ni driver_id dans le body.
    - Seules valeurs acceptées en PATCH : "detailed" | "discreet" (après .strip().lower()), sinon 400.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupérer le mode push (detailed | discreet)."""
        from http import HTTPStatus

        from models import User

        driver, err, code = get_driver_from_token()
        if err:
            return err, code or HTTPStatus.UNAUTHORIZED
        user = User.query.get(driver.user_id) if driver else None
        if not user:
            return {"error": "User not found"}, HTTPStatus.NOT_FOUND
        mode = getattr(user, "push_privacy_mode", None) or "detailed"
        return {"push_privacy_mode": mode}, 200

    @jwt_required()
    @role_required(UserRole.driver)
    def patch(self):
        """Mettre à jour le mode push (detailed | discreet)."""
        from http import HTTPStatus

        from models import User

        driver, err, code = get_driver_from_token()
        if err:
            return err, code or HTTPStatus.UNAUTHORIZED
        user = User.query.get(driver.user_id) if driver else None
        if not user:
            return {"error": "User not found"}, HTTPStatus.NOT_FOUND
        data = request.get_json(silent=True) or {}
        mode = (data.get("push_privacy_mode") or "").strip().lower()
        if mode not in ("detailed", "discreet"):
            return (
                {"error": "push_privacy_mode doit être 'detailed' ou 'discreet'"},
                HTTPStatus.BAD_REQUEST,
            )
        if hasattr(user, "push_privacy_mode"):
            user.push_privacy_mode = mode
            db.session.commit()
        return {"push_privacy_mode": mode}, 200


@driver_ns.route("/<int:driver_id>/update-profile")
class UpdateDriverProfile(Resource):
    @jwt_required()
    def post(self, driver_id: int):
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id(driver_id=driver_id)
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Chauffeur",
                driver_id,
                logger,
            )

        from application.drivers.update_driver_admin_profile import (
            UpdateDriverAdminProfileUseCase,
        )

        payload = request.get_json() or {}
        uc = UpdateDriverAdminProfileUseCase()
        uc_res = uc.execute(driver=driver, payload=payload)
        if uc_res.should_commit:
            db.session.commit()
        return uc_res.response, uc_res.status_code


@driver_ns.route("/<int:driver_id>/completed-trips")
class CompletedTrips(Resource):
    @jwt_required()
    def get(self, driver_id: int):
        from application.drivers.get_driver_completed_trips import (
            GetDriverCompletedTripsUseCase,
        )
        from infrastructure.bookings.completed_trips_query import (
            get_completed_trips_for_driver,
        )

        uc = GetDriverCompletedTripsUseCase(
            get_completed_trips_fn=get_completed_trips_for_driver
        )
        uc_res = uc.execute(driver_id=driver_id)
        return uc_res.response, uc_res.status_code


@driver_ns.route("/trips/<int:assignment_id>/tracking")
class TripTrackingReplay(Resource):
    """✅ 3.3.3: Route pour replay trajet complet avec analytics."""

    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, assignment_id: int):
        """Replay trajet complet avec positions GPS et analytics.

        Retourne:
        - Liste des positions GPS pendant le trajet
        - Analytics (vitesse moyenne, arrêts, détours)
        - Timestamps pour replay temporel
        """
        result = None
        status_code = 200
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)
                from application.drivers.get_trip_tracking_replay import (
                    GetTripTrackingReplayUseCase,
                )
                from repositories.assignment_repository import AssignmentRepository
                from repositories.trip_tracking_repository import TripTrackingRepository
                from shared.geo_utils import haversine_distance

                uc = GetTripTrackingReplayUseCase(
                    get_assignment_fn=AssignmentRepository().find_model_by_id,  # type: ignore[reportArgumentType]
                    get_positions_fn=TripTrackingRepository().find_models_by_assignment_id,  # type: ignore[reportArgumentType]
                    haversine_distance_fn=haversine_distance,
                )
                uc_res = uc.execute(assignment_id=assignment_id, driver_id=driver.id)
                result = uc_res.response
                status_code = uc_res.status_code

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors trip tracking replay (assignment_id=%s): %s - %s",
                assignment_id,
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de la récupération du replay.",
            }
            status_code = 400
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors trip tracking replay (assignment_id=%s): %s - %s",
                assignment_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la récupération du replay.",
            }
            status_code = 500
        except Exception as e:
            logger.exception(
                "❌ Erreur inattendue trip_tracking_replay (assignment_id=%s)",
                assignment_id,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": f"Internal error: {e!s}",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/me/switch-to-enterprise")
class SwitchToEnterprise(Resource):
    """Endpoint pour basculer d'un compte chauffeur vers un compte entreprise.

    Permet à un chauffeur d'urgence (ou un chauffeur lié à une entreprise)
    de basculer vers son compte entreprise.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Génère un token entreprise à partir d'un token chauffeur."""
        # #region agent log
        log_path = Path(r"c:\Users\jasiq\atmr\.cursor\debug.log")
        try:
            user_public_id = get_jwt_identity()
            from repositories.user_repository import UserRepository

            user_repo = UserRepository()
            user = user_repo.find_by_public_id(user_public_id)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "location": "driver.py:SwitchToEnterprise.post",
                            "message": "POST /switch-to-enterprise entry",
                            "data": {
                                "headers": {
                                    k: v
                                    for k, v in request.headers
                                    if k.lower()
                                    in [
                                        "authorization",
                                        "x-device-id",
                                        "x-requested-with",
                                        "content-type",
                                    ]
                                },
                                "has_authorization": "Authorization" in request.headers,
                                "user_public_id": user_public_id,
                                "user_role": user.role.value
                                if user and user.role
                                else None,
                                "user_id": user.id if user else None,
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "driver.py:SwitchToEnterprise.post",
                                "message": "POST /switch-to-enterprise entry ERROR",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion
        result = None
        status_code = 200
        try:
            # 1. Récupérer le driver depuis le token
            driver, error_response, status_code = get_driver_from_token()
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "driver.py:SwitchToEnterprise.post",
                                "message": "get_driver_from_token result",
                                "data": {
                                    "has_driver": driver is not None,
                                    "has_error": error_response is not None,
                                    "status_code": status_code,
                                    "driver_id": driver.id if driver else None,
                                    "driver_type": str(driver.driver_type)
                                    if driver
                                    else None,
                                    "company_id": driver.company_id if driver else None,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "B",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)
                from application.drivers.switch_to_enterprise import (
                    SwitchToEnterpriseCommand,
                    SwitchToEnterpriseUseCase,
                )
                from infrastructure.drivers.company_user_lookup import (
                    find_company_user_for_driver,
                )
                from repositories.company_repository import CompanyRepository

                def _store_refresh_token_fn(
                    *,
                    token: str,
                    user_id: int,
                    expires_at: datetime,
                    device_id: str | None,
                    device_name: str | None,
                ) -> None:
                    from routes.auth import store_refresh_token

                    store_refresh_token(
                        token=token,
                        user_id=user_id,
                        expires_at=expires_at,
                        device_id=device_id,
                        device_name=device_name,
                    )

                access_delta = current_app.config.get(
                    "JWT_ACCESS_TOKEN_EXPIRES", timedelta(hours=1)
                )
                # ✅ PHASE 4 : Augmentation de la durée du refresh token à 90 jours
                refresh_delta = current_app.config.get(
                    "JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=90)
                )

                # #region agent log
                try:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "location": "driver.py:SwitchToEnterprise.post",
                                    "message": "before use case execute",
                                    "data": {
                                        "driver_id": driver.id,
                                        "driver_type": str(driver.driver_type),
                                        "driver_type_value": driver.driver_type.value
                                        if hasattr(driver.driver_type, "value")
                                        else str(driver.driver_type),
                                        "expected_emergency": str(DriverType.EMERGENCY),
                                        "is_emergency": driver.driver_type
                                        == DriverType.EMERGENCY,
                                        "company_id": driver.company_id,
                                    },
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "C",
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
                uc = SwitchToEnterpriseUseCase(
                    find_company_fn=CompanyRepository().find_model_by_id,  # type: ignore[reportArgumentType]
                    find_company_user_fn=find_company_user_for_driver,  # type: ignore[reportArgumentType]
                    create_access_token_fn=create_access_token,
                    create_refresh_token_fn=create_refresh_token,
                    store_refresh_token_fn=_store_refresh_token_fn,
                    now_utc_fn=lambda: datetime.now(UTC),
                    driver_type_emergency=DriverType.EMERGENCY,
                )
                uc_res = uc.execute(
                    SwitchToEnterpriseCommand(
                        driver=driver,  # type: ignore[reportArgumentType]
                        access_expires_delta=access_delta,
                        refresh_expires_delta=refresh_delta,
                        device_id=request.headers.get("X-Device-ID"),
                        device_name=request.headers.get("X-Device-Name"),
                    )
                )
                # #region agent log
                try:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "location": "driver.py:SwitchToEnterprise.post",
                                    "message": "use case execute result",
                                    "data": {
                                        "status_code": uc_res.status_code,
                                        "has_error": "error" in uc_res.response,
                                        "error_message": uc_res.response.get("error"),
                                        "has_token": "token" in uc_res.response,
                                    },
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "D",
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
                result = uc_res.response
                status_code = uc_res.status_code

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors switch-to-enterprise: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors du basculement vers l'entreprise.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur contrainte DB lors switch-to-enterprise: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            logger.exception(
                "❌ Erreur DB lors switch-to-enterprise: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors du basculement vers l'entreprise.",
            }
            status_code = 500
        except Exception as e:
            logger.exception("❌ Erreur inattendue switch-to-enterprise")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/night-mode-status")
class NightModeStatus(Resource):
    """Endpoint pour vérifier le statut du mode nuit.

    Utile pour debugging et monitoring.
    """

    def get(self):
        """Récupère le statut actuel du mode nuit.

        Returns:
            {
                "is_night": bool,
                "current_time": "HH:MM",
                "night_start": "22:00",
                "night_end": "06:00"
            }
        """
        try:
            from services.events.night_mode import get_night_mode_status

            status = get_night_mode_status()
            return status, 200
        except Exception as e:
            logger.exception("❌ Erreur récupération statut mode nuit")
            return {"error": str(e)}, 500


@driver_ns.route("/me/bookings/<int:booking_id>/quick-accept")
class QuickAcceptBooking(Resource):
    """Endpoint pour accepter rapidement une mission depuis une notification.

    Phase 2 - Actions directes.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        """Accepte une mission rapidement (depuis notification).

        Args:
            booking_id: ID de la mission

        Returns:
            {
                "ok": true,
                "message": "Mission acceptée",
                "booking_id": int
            }
        """
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                logger.error("Driver not found for token: %s", get_jwt_identity())
                return error_response, status_code

            driver = cast("Driver", driver)

            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            booking = booking_repo.find_model_by_id(booking_id=booking_id)

            if not booking:
                logger.error("Booking %s not found for quick-accept", booking_id)
                return {"error": "Booking not found"}, 404

            # Vérifier que la mission est assignée au chauffeur
            if booking.driver_id != driver.id:
                logger.warning(
                    "Driver %s tried to accept booking %s not assigned to them",
                    driver.id,
                    booking_id,
                )
                return {"error": "Cette mission ne vous est pas assignée"}, 403

            # Vérifier que la mission est dans un statut acceptable
            if booking.status not in [
                BookingStatus.assigned,
                BookingStatus.pending,
            ]:
                logger.warning(
                    "Booking %s cannot be accepted from status %s",
                    booking_id,
                    booking.status,
                )
                return {
                    "error": f"Mission dans un statut non accepté: {booking.status}"
                }, 400

            # Mettre à jour le statut
            booking.status = BookingStatus.accepted
            db.session.commit()

            logger.info(
                "✅ Mission %s quickly accepted by driver %s",
                booking_id,
                driver.id,
            )

            # Notifier les autres parties prenantes
            try:
                from shared.notifications import notify_booking_update

                notify_booking_update(driver_id=driver.id, booking=booking)
            except Exception:
                logger.exception("Erreur notification après quick-accept")

            return {
                "ok": True,
                "message": "Mission acceptée",
                "booking_id": booking_id,
            }, 200

        except Exception as e:
            logger.exception("❌ Erreur quick-accept booking %s", booking_id)
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500


@driver_ns.route("/me/bookings/<int:booking_id>/quick-reject")
class QuickRejectBooking(Resource):
    """Endpoint pour refuser rapidement une mission depuis une notification.

    Phase 2 - Actions directes.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        """Refuse une mission rapidement (depuis notification).

        Args:
            booking_id: ID de la mission

        Returns:
            {
                "ok": true,
                "message": "Mission refusée",
                "booking_id": int
            }
        """
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                logger.error("Driver not found for token: %s", get_jwt_identity())
                return error_response, status_code

            driver = cast("Driver", driver)

            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            booking = booking_repo.find_model_by_id(booking_id=booking_id)

            if not booking:
                logger.error("Booking %s not found for quick-reject", booking_id)
                return {"error": "Booking not found"}, 404

            # Vérifier que la mission est assignée au chauffeur
            if booking.driver_id != driver.id:
                logger.warning(
                    "Driver %s tried to reject booking %s not assigned to them",
                    driver.id,
                    booking_id,
                )
                return {"error": "Cette mission ne vous est pas assignée"}, 403

            # Vérifier que la mission est dans un statut refusable
            if booking.status not in [
                BookingStatus.assigned,
                BookingStatus.pending,
            ]:
                logger.warning(
                    "Booking %s cannot be rejected from status %s",
                    booking_id,
                    booking.status,
                )
                return {
                    "error": f"Mission dans un statut non refusable: {booking.status}"
                }, 400

            # Mettre à jour le statut
            booking.status = BookingStatus.cancelled
            booking.driver_id = None  # Libérer le chauffeur
            db.session.commit()

            logger.info(
                "❌ Mission %s quickly rejected by driver %s",
                booking_id,
                driver.id,
            )

            # Notifier les autres parties prenantes
            try:
                from shared.notifications import notify_booking_update

                notify_booking_update(driver_id=driver.id, booking=booking)
            except Exception:
                logger.exception("Erreur notification après quick-reject")

            return {
                "ok": True,
                "message": "Mission refusée",
                "booking_id": booking_id,
            }, 200

        except Exception as e:
            logger.exception("❌ Erreur quick-reject booking %s", booking_id)
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500
