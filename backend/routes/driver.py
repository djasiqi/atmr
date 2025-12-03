from __future__ import annotations

import contextlib
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from typing import cast as tcast

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields
from sqlalchemy import or_

from ext import app_logger, db, redis_client, role_required, socketio
from models import (
    Assignment,
    Booking,
    BookingStatus,
    DelayEvent,
    Driver,
    TripTracking,
    User,
    UserRole,
)
from models.enums import CancelReason
from shared.notifications import notify_booking_update

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

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import ColumnElement

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
                "Nouveau statut (en_route, in_progress, completed, return_completed)"
            ),
        )
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
    user_public_id = get_jwt_identity()
    app_logger.info(f"JWT Identity récupérée: {user_public_id}")

    user = User.query.filter_by(public_id=user_public_id).one_or_none()
    if not user:
        app_logger.error("User not found for public_id: {user_public_id}")
        return None, {"error": "User not found"}, 404

    app_logger.info(f"User details: id={user.id}, role={user.role}")

    if user.role != UserRole.driver:
        app_logger.error(
            f"User {getattr(user, 'username', user.id)} n'a pas le rôle 'driver'"
        )
        return None, {"error": "Driver not found"}, 404

    driver = Driver.query.filter_by(user_id=user.id).one_or_none()
    if not driver:
        app_logger.error("Driver not found for user ID: {user.id}")
        return None, {"error": "Driver not found"}, 404

    app_logger.info(
        f"Driver found: {driver.id} for user {getattr(user, 'username', user.id)}"
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
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                return error_response, status_code
            driver = cast("Driver", driver)
            return {"profile": driver.serialize}, 200
        except Exception as e:
            app_logger.error(
                f"❌ ERREUR get_driver_profile: {type(e).__name__} - {e!s}",
                exc_info=True,
            )
            sentry_sdk.capture_exception(e)
            return {
                "error": (
                    "Une erreur interne est survenue lors de la récupération du profil."
                )
            }, 500

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(driver_profile_model)
    def put(self):
        """Met à jour le profil du chauffeur."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.driver_schemas import DriverProfileUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(
                DriverProfileUpdateSchema(), data, strict=False
            )
        except ValidationError as e:
            return handle_validation_error(e)

        app_logger.info(f"Payload reçu pour mise à jour du profil: {validated_data}")
        if not driver.user:
            return {"error": "Aucun utilisateur associé au driver"}, 500

        # Mise à jour champs utilisateur - utilise données validées
        if validated_data.get("first_name"):
            driver.user.first_name = validated_data["first_name"]
        if validated_data.get("last_name"):
            driver.user.last_name = validated_data["last_name"]
        if validated_data.get("phone"):
            driver.user.phone = validated_data["phone"]

        # Statut
        if validated_data.get("status"):
            status_val = validated_data["status"].strip().lower()
            if status_val == "disponible":
                driver.is_active = True
            elif status_val == "hors service":
                driver.is_active = False

        try:
            # HR optional updates - utilise données validées
            if validated_data.get("contract_type"):
                driver.contract_type = validated_data["contract_type"].upper()
            if validated_data.get("weekly_hours") is not None:
                driver.weekly_hours = validated_data["weekly_hours"]
            if validated_data.get("hourly_rate_cents") is not None:
                driver.hourly_rate_cents = validated_data["hourly_rate_cents"]

            # Dates (déjà validées par Marshmallow comme Date)
            if validated_data.get("employment_start_date"):
                driver.employment_start_date = validated_data["employment_start_date"]
            if validated_data.get("employment_end_date"):
                driver.employment_end_date = validated_data["employment_end_date"]
            if validated_data.get("license_valid_until"):
                driver.license_valid_until = validated_data["license_valid_until"]
            if validated_data.get("medical_valid_until"):
                driver.medical_valid_until = validated_data["medical_valid_until"]

            # Listes
            if validated_data.get("license_categories"):
                driver.license_categories = [
                    str(cat) for cat in validated_data["license_categories"]
                ]
            if validated_data.get("trainings"):
                driver.trainings = validated_data["trainings"]

            db.session.commit()
            app_logger.info(f"Profil du driver {driver.id} mis à jour avec succès")
            return {
                "profile": driver.serialize,
                "message": "Profil mis à jour avec succès",
            }, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(
                f"❌ ERREUR update_driver_profile: {type(e).__name__} - {e!s}",
                exc_info=True,
            )
            return {"error": "Une erreur interne est survenue."}, 500


@driver_ns.route("/me/photo")
class DriverPhoto(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(photo_model)
    def put(self):
        """Met à jour la photo du chauffeur."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        data = request.get_json()
        app_logger.info(f"Payload reçu pour mise à jour de la photo: {data}")
        if not data or "photo" not in data:
            return {"error": "Donnée photo non fournie"}, 400

        photo_data = data.get("photo")
        if not photo_data:
            return {"error": "Photo invalide"}, 400

        driver.driver_photo = photo_data
        try:
            db.session.commit()
            app_logger.info(f"Photo du driver {driver.id} mise à jour avec succès")
            return {
                "profile": driver.serialize,
                "message": "Photo mise à jour avec succès",
            }, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(
                f"❌ ERREUR update_driver_photo: {type(e).__name__} - {e!s}",
                exc_info=True,
            )
            return {"error": "Une erreur interne est survenue."}, 500


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
        app_logger.info(
            (
                f"📱 [Driver Bookings] Driver {driver_name} "
                f"(ID: {driver.id}) loading bookings"
            )
        )

        from datetime import date, timedelta

        from shared.time_utils import day_local_bounds, now_local

        # ✅ Récupérer les courses d'AUJOURD'HUI (passées et futures)
        # tant qu'elles ne sont pas terminées
        today_start, today_end = day_local_bounds(date.today().strftime("%Y-%m-%d"))

        # S'assurer que ce sont des objets datetime pour SQLAlchemy
        from datetime import datetime

        today_start = datetime.fromisoformat(str(today_start))
        today_end = datetime.fromisoformat(str(today_end))

        # 🆕 LOGIQUE : À partir de 19h00, inclure aussi les courses du lendemain
        # pour permettre aux chauffeurs de voir leur planning la veille
        now = now_local()
        cutoff_hour = 19  # 19h00

        # Si on est après 19h00, inclure aussi le lendemain
        if now.hour >= cutoff_hour:
            tomorrow = date.today() + timedelta(days=1)
            tomorrow_start, tomorrow_end = day_local_bounds(
                tomorrow.strftime("%Y-%m-%d")
            )
            tomorrow_start = datetime.fromisoformat(str(tomorrow_start))
            tomorrow_end = datetime.fromisoformat(str(tomorrow_end))
            # Étendre la plage jusqu'à la fin du lendemain
            query_end = tomorrow_end
            app_logger.info(
                (
                    f"📱 [Driver Bookings] After 19:00 - including tomorrow's bookings "
                    f"(until {query_end})"
                )
            )
        else:
            # Avant 19h00 : uniquement les courses du jour
            query_end = today_end
            app_logger.info(
                (
                    f"📱 [Driver Bookings] Before 19:00 - today's bookings only "
                    f"(until {query_end})"
                )
            )

        status_pred = Booking.status.in_(
            [BookingStatus.ASSIGNED, BookingStatus.EN_ROUTE, BookingStatus.IN_PROGRESS]
        )

        bookings = (
            Booking.query.filter_by(driver_id=driver.id)
            .filter(
                Booking.scheduled_time >= today_start,
                Booking.scheduled_time < query_end,
            )
            .filter(status_pred)
            .order_by(Booking.scheduled_time.asc())
            .all()
        )

        # 🔍 LOG : Afficher les courses trouvées
        app_logger.info(
            (
                f"📱 [Driver Bookings] Found {len(bookings)} bookings "
                f"for driver {driver_name} (ID: {driver.id})"
            )
        )
        for b in bookings:
            app_logger.info(
                (
                    f"   - Booking #{b.id}: driver_id={b.driver_id}, "
                    f"client={b.customer_name}, time={b.scheduled_time}"
                )
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

        from datetime import date, timedelta

        from services.unified_dispatch.data import calculate_eta as calc_eta
        from shared.time_utils import day_local_bounds, now_local

        # Récupérer les courses d'aujourd'hui (non terminées)
        today_start, today_end = day_local_bounds(date.today().strftime("%Y-%m-%d"))

        status_pred = Booking.status.in_(
            [BookingStatus.ASSIGNED, BookingStatus.EN_ROUTE, BookingStatus.IN_PROGRESS]
        )

        bookings = (
            Booking.query.filter_by(driver_id=driver.id)
            .filter(
                Booking.scheduled_time >= today_start,
                Booking.scheduled_time < today_end,
            )
            .filter(status_pred)
            .order_by(Booking.scheduled_time.asc())
            .all()
        )

        # Position actuelle du chauffeur
        driver_lat = getattr(driver, "latitude", None)
        driver_lon = getattr(driver, "longitude", None)

        if not driver_lat or not driver_lon:
            # Pas de position GPS, retourner les durées statiques
            return {
                "has_gps": False,
                "bookings": [
                    {
                        "id": b.id,
                        "duration_seconds": b.duration_seconds,
                        "distance_meters": b.distance_meters,
                    }
                    for b in bookings
                ],
            }, 200

        driver_pos = (float(driver_lat), float(driver_lon))
        current_time = now_local()

        results = []
        for booking in bookings:
            pickup_lat = getattr(booking, "pickup_lat", None)
            pickup_lon = getattr(booking, "pickup_lon", None)
            dropoff_lat = getattr(booking, "dropoff_lat", None)
            dropoff_lon = getattr(booking, "dropoff_lon", None)

            eta_to_pickup = None
            total_duration = booking.duration_seconds

            # Si on a les coordonnées, calculer l'ETA dynamique
            if pickup_lat and pickup_lon:
                try:
                    pickup_pos = (float(pickup_lat), float(pickup_lon))
                    eta_seconds = calc_eta(driver_pos, pickup_pos)
                    eta_to_pickup = eta_seconds

                    # Si on a aussi les coordonnées de destination, recalculer
                    # la durée totale
                    if (
                        dropoff_lat
                        and dropoff_lon
                        and booking.status != BookingStatus.IN_PROGRESS
                    ):
                        dropoff_pos = (float(dropoff_lat), float(dropoff_lon))
                        pickup_to_dropoff = calc_eta(pickup_pos, dropoff_pos)
                        total_duration = pickup_to_dropoff
                except Exception as e:
                    app_logger.warning(
                        f"ETA calculation failed for booking {booking.id}: {e}"
                    )

            results.append(
                {
                    "id": booking.id,
                    "eta_to_pickup_seconds": eta_to_pickup,
                    "duration_seconds": total_duration,
                    "distance_meters": booking.distance_meters,
                    "estimated_arrival": (
                        current_time + timedelta(seconds=eta_to_pickup)
                    ).isoformat()
                    if eta_to_pickup
                    else None,
                }
            )

        return {
            "has_gps": True,
            "driver_position": {"lat": driver_lat, "lon": driver_lon},
            "bookings": results,
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
            app_logger.debug(f"📍 Received location data: {p} (type={type(p)})")

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

                        # ✅ 3.3.1: Utiliser LocationService pour centraliser la logique
                        from services.location_service import get_location_service

                        timestamp = (
                            datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if ts
                            else datetime.now(UTC)
                        )

                        try:
                            location_service = get_location_service()
                            result = location_service.update_driver_location(
                                driver_id=driver.id,
                                latitude=lat,
                                longitude=lon,
                                speed=speed if speed > 0 else None,
                                heading=heading if heading >= 0 else None,
                                accuracy=accuracy if accuracy > 0 else None,
                                source="gps",
                                timestamp=timestamp,
                            )

                            # Utiliser position snapée
                            lat = result.snapped_lat
                            lon = result.snapped_lon
                            source = result.source

                            # Émettre events geofencing si détectés
                            for event in result.geofence_events:
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
                            app_logger.warning(
                                "[LocationService] HTTP location update failed: %s",
                                str(e_loc),
                            )
                            source = "raw"  # Fallback

                        # 5) Diffusion temps réel à la room entreprise
                        try:
                            room = f"company_{driver.company_id}"
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
            app_logger.error(
                "❌ Unexpected error in location update: %s", e, exc_info=True
            )
            app_logger.error("❌ Request data: %s", request.get_data())
            result = {"error": f"Internal error: {e!s}"}
            status_code = 500

        return result, status_code


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

            booking = Booking.query.filter_by(
                id=booking_id, driver_id=driver.id
            ).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404

            return {
                "id": booking.id,
                "customer_name": booking.customer_name
                or getattr(booking, "customer_full_name", None),
                "client_name": booking.customer_name
                or getattr(booking, "customer_full_name", None),
                "pickup_location": booking.pickup_location,
                "dropoff_location": booking.dropoff_location,
                "scheduled_time": booking.scheduled_time.isoformat()
                if booking.scheduled_time
                else None,
                "amount": booking.amount,
                "status": booking.status.value
                if hasattr(booking.status, "value")
                else str(booking.status),
                # 🏥 Informations médicales
                "medical_facility": booking.medical_facility,
                "doctor_name": booking.doctor_name,
                "hospital_service": booking.hospital_service,
                "notes_medical": booking.notes_medical,
                "wheelchair_client_has": booking.wheelchair_client_has,
                "wheelchair_need": booking.wheelchair_need,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(
                f"❌ ERREUR get_booking_details: {type(e).__name__} - {e!s}",
                exc_info=True,
            )
            return {"error": "Une erreur interne est survenue."}, 500


@driver_ns.route("/company/<int:company_id>/live-locations")
class CompanyLiveLocations(Resource):
    @jwt_required()
    def get(self, company_id: int):
        """Retourne la dernière position connue
        de tous les chauffeurs de l'entreprise."""
        try:
            drivers = Driver.query.filter_by(company_id=company_id).all()
            items: list[dict[str, Any]] = []
            rc: Any = redis_client

            for d in drivers:
                key = f"driver:{d.id}:loc"
                h = rc.hgetall(key)
                if not h:
                    continue
                # redis renvoie bytes -> decode

                def _dec(v: Any) -> Any:
                    try:
                        return v.decode()
                    except Exception:
                        return v

                # h peut être dict[bytes, bytes] ; on force le cast pour
                # Pylance
                rec = {(_dec(k)): _dec(v) for k, v in cast("dict[str, str]", h).items()}

                for kf in ("lat", "lon", "speed", "heading", "accuracy"):
                    if kf in rec:
                        with contextlib.suppress(Exception):
                            rec[kf] = float(rec[kf])
                items.append({"driver_id": d.id, **rec})

            return {"items": items}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return {"items": []}, 200


@driver_ns.route("/me/bookings/<int:booking_id>/status", methods=["PUT", "OPTIONS"])
class UpdateBookingStatus(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(booking_status_model)
    def put(self, booking_id: int):
        # Variables pour stocker le résultat
        result = None
        status_code = 200

        if request.method == "OPTIONS":
            result = {}
            status_code = 200
        else:
            data = request.get_json()
            app_logger.info(f"Body reçu pour status update: {data}")

            try:
                driver, error_response, status_code = get_driver_from_token()
                if error_response:
                    app_logger.error(
                        "Driver not found for token: %s", get_jwt_identity()
                    )
                    result = error_response
                else:
                    driver = cast("Driver", driver)

                    booking = Booking.query.filter_by(id=booking_id).first()
                    if not booking:
                        app_logger.error("Booking with id %s not found", booking_id)
                        result = {"error": "Booking not found"}
                        status_code = 404
                    elif (
                        booking.driver_id is None
                        and booking.status == BookingStatus.PENDING
                    ):
                        booking.driver_id = driver.id
                    elif booking.driver_id != driver.id:
                        result = {"error": "Unauthorized access to this booking"}
                        status_code = 403
                    elif not data:
                        result = {"error": "Missing JSON payload"}
                        status_code = 400
                    else:
                        new_status_str = data.get("status")
                        valid_statuses = [
                            "en_route",
                            "in_progress",
                            "completed",
                            "return_completed",
                            "canceled",  # ✅ Annulation par le chauffeur
                        ]
                        if new_status_str not in valid_statuses:
                            result = {"error": "Invalid status"}
                            status_code = 400
                        else:
                            # EN ROUTE
                            if new_status_str == "en_route":
                                if booking.status == BookingStatus.EN_ROUTE:
                                    result = {"message": "Booking already en route"}
                                elif booking.status != BookingStatus.ASSIGNED:
                                    result = {
                                        "error": (
                                            "Booking must be ASSIGNED "
                                            "before going en_route"
                                        )
                                    }
                                    status_code = 400
                                else:
                                    booking.status = BookingStatus.EN_ROUTE

                            # EN COURS
                            elif new_status_str == "in_progress":
                                if booking.status == BookingStatus.IN_PROGRESS:
                                    result = {"message": "Booking already in progress"}
                                elif booking.status != BookingStatus.EN_ROUTE:
                                    result = {
                                        "error": (
                                            "Booking must be en_route before starting"
                                        )
                                    }
                                    status_code = 400
                                else:
                                    booking.status = BookingStatus.IN_PROGRESS
                                    booking.boarded_at = datetime.now(UTC)

                            # TERMINER (ALLER OU RETOUR SELON is_return)
                            elif new_status_str == "completed":
                                if booking.is_return:
                                    if booking.status == BookingStatus.RETURN_COMPLETED:
                                        result = {
                                            "message": "Return trip already completed"
                                        }
                                    elif booking.status != BookingStatus.IN_PROGRESS:
                                        result = {
                                            "error": (
                                                "Booking must be in_progress "
                                                "before completing return"
                                            )
                                        }
                                        status_code = 400
                                    else:
                                        booking.status = BookingStatus.RETURN_COMPLETED
                                        booking.completed_at = datetime.now(UTC)
                                        # ✅ 3.5.1: Résoudre retards lors complétion
                                        DelayEvent.resolve_delays_for_booking(
                                            booking.id, booking.completed_at
                                        )
                                elif booking.status == BookingStatus.COMPLETED:
                                    result = {"message": "Booking already completed"}
                                elif booking.status != BookingStatus.IN_PROGRESS:
                                    result = {
                                        "error": (
                                            "Booking must be in_progress "
                                            "before completing"
                                        )
                                    }
                                    status_code = 400
                                else:
                                    booking.status = BookingStatus.COMPLETED
                                    booking.completed_at = datetime.now(UTC)
                                    # ✅ 3.5.1: Résoudre retards lors complétion
                                    DelayEvent.resolve_delays_for_booking(
                                        booking.id, booking.completed_at
                                    )

                            # TERMINER RETOUR explicite
                            elif new_status_str == "return_completed":
                                if booking.status == BookingStatus.RETURN_COMPLETED:
                                    result = {
                                        "message": "Return trip already completed"
                                    }
                                elif booking.status != BookingStatus.IN_PROGRESS:
                                    result = {
                                        "error": (
                                            "Booking must be in_progress "
                                            "before completing return"
                                        )
                                    }
                                    status_code = 400
                                elif booking.is_return:
                                    booking.status = BookingStatus.RETURN_COMPLETED
                                    booking.completed_at = datetime.now(UTC)
                                    # ✅ 3.5.1: Résoudre retards lors complétion
                                    DelayEvent.resolve_delays_for_booking(
                                        booking.id, booking.completed_at
                                    )
                                else:
                                    result = {"error": "Not a return trip"}
                                    status_code = 400

                            # ✅ ANNULATION PAR LE CHAUFFEUR (facturable) OU LIBÉRATION (réassignation)
                            elif new_status_str == "canceled":
                                if booking.status == BookingStatus.CANCELED:
                                    result = {"message": "Booking already canceled"}
                                elif booking.status in [
                                    BookingStatus.COMPLETED,
                                    BookingStatus.RETURN_COMPLETED,
                                ]:
                                    result = {
                                        "error": "Impossible d'annuler une course déjà terminée"
                                    }
                                    status_code = 400
                                elif booking.status == BookingStatus.IN_PROGRESS:
                                    result = {
                                        "error": (
                                            "Impossible d'annuler une course en cours : "
                                            "le client est déjà à bord"
                                        )
                                    }
                                    status_code = 400
                                elif booking.status not in [
                                    BookingStatus.ASSIGNED,
                                    BookingStatus.EN_ROUTE,
                                ]:
                                    result = {
                                        "error": (
                                            "Impossible d'annuler une course "
                                            "qui n'est pas assignée ou en route"
                                        )
                                    }
                                    status_code = 400
                                else:
                                    # Récupérer la raison d'annulation (CANCEL ou RELEASE)
                                    cancel_reason_str = data.get(
                                        "cancel_reason", "CANCEL"
                                    )
                                    cancel_reason = CancelReason.CANCEL  # Par défaut
                                    try:
                                        cancel_reason = CancelReason(cancel_reason_str)
                                    except ValueError:
                                        # Si la valeur n'est pas valide, utiliser CANCEL par défaut
                                        app_logger.warning(
                                            (
                                                f"Raison d'annulation invalide: {cancel_reason_str}, "
                                                "utilisation de CANCEL par défaut"
                                            )
                                        )

                                    if cancel_reason == CancelReason.RELEASE:
                                        # ✅ LIBÉRATION : remettre à ACCEPTED pour permettre réassignation
                                        # Ne pas facturer, permettre à l'entreprise de réassigner ou au système de réassigner automatiquement
                                        # ACCEPTED est le statut approprié car le booking est accepté mais pas encore assigné à un chauffeur
                                        # Cela fonctionne même si le statut était EN_ROUTE - le booking revient à ACCEPTED pour être réassigné
                                        previous_status = (
                                            booking.status
                                        )  # Sauvegarder l'ancien statut pour le log
                                        booking.status = BookingStatus.ACCEPTED
                                        booking.driver_id = None

                                        # Supprimer l'assignment pour permettre une nouvelle assignation
                                        assignment = Assignment.query.filter_by(
                                            booking_id=booking_id
                                        ).first()
                                        assignment_id_str = None
                                        if assignment:
                                            assignment_id_str = str(assignment.id)
                                            db.session.delete(assignment)

                                        app_logger.info(
                                            (
                                                f"📱 [Driver Release] Chauffeur {driver.id} "
                                                f"a libéré la course {booking_id} pour réassignation "
                                                f"(statut précédent: {previous_status}, nouveau statut: ACCEPTED)"
                                            )
                                        )

                                        # Émettre événement d'annulation d'assignment pour notifier l'entreprise
                                        from services.socketio_service import (
                                            emit_assignment_cancelled,
                                        )

                                        if assignment_id_str:
                                            emit_assignment_cancelled(
                                                company_id=booking.company_id,
                                                assignment_id=assignment_id_str,
                                                booking_id=booking_id,
                                                driver_id=driver.id,
                                            )

                                        # ✅ Déclencher un dispatch automatique pour réassigner immédiatement
                                        # si le dispatch est activé pour l'entreprise
                                        if _maybe_trigger_dispatch:
                                            try:
                                                _maybe_trigger_dispatch(
                                                    booking.company_id, "reassign"
                                                )
                                            except Exception as e:
                                                app_logger.warning(
                                                    "Erreur lors du déclenchement du dispatch: %s",
                                                    e,
                                                )
                                    else:
                                        # ✅ ANNULATION RÉELLE : marquer comme CANCELED
                                        # Déterminer si facturation selon la raison
                                        client_fault_reasons = {
                                            CancelReason.CLIENT_REQUEST,
                                            CancelReason.CLIENT_NO_SHOW,
                                            CancelReason.CANCEL,  # Par défaut, CANCEL est facturé
                                        }
                                        should_bill = (
                                            cancel_reason in client_fault_reasons
                                        )

                                        # Stocker la raison d'annulation pour la facturation
                                        # Note: On peut ajouter un champ cancel_reason dans le modèle Booking si nécessaire
                                        booking.status = BookingStatus.CANCELED

                                        billing_status = (
                                            "facturée"
                                            if should_bill
                                            else "non facturée"
                                        )
                                        app_logger.info(
                                            (
                                                f"📱 [Driver Cancel] Chauffeur {driver.id} "
                                                f"a annulé la course {booking_id} (raison: {cancel_reason.value}, {billing_status}) "
                                                f"(statut précédent: {booking.status})"
                                            )
                                        )

                                        # TODO: Si nécessaire, stocker cancel_reason dans un champ du booking
                                        # pour la facturation ultérieure (booking.cancel_reason = cancel_reason.value)

                            if result is None:
                                db.session.commit()
                                driver_id = driver.id
                                notify_booking_update(driver_id, booking)
                                result = {
                                    "message": (
                                        f"Booking status updated to {new_status_str}"
                                    )
                                }

            except Exception as e:
                sentry_sdk.capture_exception(e)
                app_logger.error(
                    "❌ ERREUR update_booking_status: %s - %s",
                    type(e).__name__,
                    str(e),
                    exc_info=True,
                )
                result = {"error": "Une erreur interne est survenue."}
                status_code = 500

        return result, status_code


@driver_ns.route("/me/bookings/<int:booking_id>")
class RejectBooking(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def delete(self, booking_id: int):
        """Réjette une réservation assignée."""
        try:
            current_user_id = get_jwt_identity()
            driver = Driver.query.filter_by(user_id=current_user_id).one_or_none()
            if not driver:
                return {"error": "Unauthorized: Driver not found"}, 403

            booking = Booking.query.filter_by(
                id=booking_id, driver_id=driver.id
            ).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404
            if booking.status != BookingStatus.ASSIGNED:
                return {"error": "Only assigned bookings can be rejected"}, 400

            booking.driver_id = None
            booking.status = BookingStatus.PENDING
            db.session.commit()
            notify_booking_cancelled(driver.id, booking.id)

            return {"message": "Booking rejected successfully"}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(
                f"❌ ERREUR reject_booking: {type(e).__name__} - {e!s}", exc_info=True
            )
            return {"error": "Une erreur interne est survenue."}, 500


@driver_ns.route("/me/availability")
class UpdateAvailability(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(availability_model)
    def put(self):
        """Met à jour la disponibilité du chauffeur."""
        try:
            current_user_id = get_jwt_identity()
            driver = Driver.query.filter_by(user_id=current_user_id).one_or_none()
            if not driver:
                return {"error": "Unauthorized: Driver not found"}, 403

            data = request.get_json()
            availability = data.get("is_available") if data else None
            if availability is None:
                return {"error": "Availability status is required"}, 400

            driver.is_available = bool(availability)
            db.session.commit()
            status_str = "available" if availability else "unavailable"
            return {"message": f"Driver is now {status_str}"}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(
                f"❌ ERREUR update_availability: {type(e).__name__} - {e!s}",
                exc_info=True,
            )
            return {"error": "Une erreur interne est survenue."}, 500


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

        bookings = Booking.query.filter_by(driver_id=driver.id).all()
        # ✅ Retourner une liste vide au lieu d'une erreur 404
        return [b.serialize for b in bookings], 200


@driver_ns.route("/me/bookings/<int:booking_id>/report")
class ReportBookingIssue(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        booking = Booking.query.filter_by(
            id=booking_id, driver_id=driver.id
        ).one_or_none()
        if not booking:
            return {"error": "Booking not found"}, 404

        data = request.get_json()
        issue_message = (data or {}).get("issue")
        if not issue_message:
            return {"error": "Issue message is required"}, 400

        # Assure-toi que ce champ existe dans le modèle Booking
        booking.issue_report = issue_message
        try:
            db.session.commit()
            return {"message": "Issue reported successfully"}, 200
        except Exception:
            db.session.rollback()
            return {"error": "Une erreur interne est survenue."}, 500


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
            app_logger.info(f"[push-token] payload={payload_raw}")
            payload: dict[str, Any] = tcast("dict[str, Any]", payload_raw)

            # token (expo/fcm) requis
            token_any: Any = (
                payload.get("token")
                or payload.get("expo_token")
                or payload.get("push_token")
            )
            if not isinstance(token_any, str) or len(token_any) < MIN_TOKEN_LENGTH:
                result = {"error": "Token FCM/Expo invalide ou manquant."}
                status_code = 400
            else:
                token: str = token_any

                # 1) si driverId fourni -> on essaye de le caster
                driver_id: int | None = None
                raw_id: Any = payload.get("driverId") or payload.get("driver_id")
                if raw_id is not None:
                    try:
                        # Convertir en float d'abord pour gérer les nombres
                        # décimaux, puis en int
                        driver_id = int(float(raw_id))
                        app_logger.info(
                            f"[push-token] driver_id extrait du payload: {driver_id}"
                        )
                    except (ValueError, TypeError) as e:
                        app_logger.warning(
                            (
                                f"[push-token] Impossible de convertir "
                                f"driver_id={raw_id}: {e}"
                            )
                        )
                        result = {"error": f"Format de driverId invalide: {raw_id}"}
                        status_code = 400

                # 2) sinon on déduit depuis le JWT (user -> driver)
                if result is None and driver_id is None:
                    app_logger.info(
                        "[push-token] driver_id absent du payload, déduction depuis JWT"
                    )
                    user_pid = get_jwt_identity()
                    if not user_pid:
                        result = {"error": "Token JWT invalide ou expiré."}
                        status_code = 401
                    else:
                        user = User.query.filter_by(public_id=user_pid).one_or_none()
                        if not user:
                            result = {"error": "Utilisateur non trouvé pour le JWT."}
                            status_code = 404
                        else:
                            drv = Driver.query.filter_by(user_id=user.id).one_or_none()
                            if not drv:
                                result = {
                                    "error": (
                                        "Chauffeur introuvable pour cet utilisateur."
                                    )
                                }
                                status_code = 404
                            else:
                                driver_id = int(drv.id)
                                app_logger.info(
                                    f"[push-token] driver_id déduit du JWT: {driver_id}"
                                )

                # 3) Validation finale et enregistrement
                if result is None:
                    driver = Driver.query.get(driver_id)
                    if not driver:
                        app_logger.error(
                            (
                                f"[push-token] Driver introuvable "
                                f"pour driver_id={driver_id}"
                            )
                        )
                        result = {
                            "error": f"Chauffeur introuvable pour l'ID {driver_id}."
                        }
                        status_code = 404
                    else:
                        # Enregistrement du token
                        driver.push_token = token
                        db.session.commit()

                        app_logger.info(
                            (
                                f"[push-token] ✅ Token enregistré avec succès "
                                f"pour driver_id={driver_id}"
                            )
                        )
                        result = {
                            "message": "✅ Push token enregistré avec succès.",
                            "driver_id": driver_id,
                        }

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"[push-token] ❌ Erreur serveur: {e!s}", exc_info=True)
            traceback.print_exc()
            result = {"error": f"Erreur serveur : {e!s}"}
            status_code = 500

        return result, status_code


@driver_ns.route("/<int:driver_id>/update-profile")
class UpdateDriverProfile(Resource):
    @jwt_required()
    def post(self, driver_id: int):
        driver = Driver.query.get(driver_id)
        if not driver:
            return {"error": "Chauffeur non trouvé."}, 404

        data = request.get_json() or {}
        driver.vehicle_assigned = data.get("vehicle_assigned", driver.vehicle_assigned)
        driver.brand = data.get("brand", driver.brand)
        driver.license_plate = data.get("license_plate", driver.license_plate)
        driver.driver_photo = data.get("photo", driver.driver_photo)
        if driver.user:
            driver.user.phone = data.get("phone", driver.user.phone)

        db.session.commit()
        return {"message": "Profil mis à jour avec succès."}, 200


@driver_ns.route("/<int:driver_id>/completed-trips")
class CompletedTrips(Resource):
    @jwt_required()
    def get(self, driver_id: int):
        # Chaque clause est castée en ColumnElement[bool] pour Pylance
        drv_clause: ColumnElement[bool] = tcast(
            "ColumnElement[bool]", Booking.driver_id == driver_id
        )

        st_completed: ColumnElement[bool] = tcast(
            "ColumnElement[bool]", Booking.status == BookingStatus.COMPLETED
        )
        st_return_completed: ColumnElement[bool] = tcast(
            "ColumnElement[bool]", Booking.status == BookingStatus.RETURN_COMPLETED
        )
        status_clause = or_(st_completed, st_return_completed)

        trips = (
            Booking.query.filter(drv_clause)
            .filter(status_clause)
            .order_by(Booking.scheduled_time.desc())
            .all()
        )
        return [trip.serialize for trip in trips], 200


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
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                return error_response, status_code
            driver = cast("Driver", driver)

            # Vérifier que l'assignment appartient au chauffeur
            assignment = Assignment.query.get(assignment_id)
            if not assignment or assignment.driver_id != driver.id:
                return {"error": "Assignment not found or unauthorized"}, 404

            # Récupérer toutes les positions du trajet
            positions = (
                TripTracking.query.filter_by(assignment_id=assignment_id)
                .order_by(TripTracking.timestamp.asc())
                .all()
            )

            if not positions:
                return {
                    "assignment_id": assignment_id,
                    "positions": [],
                    "analytics": {
                        "total_positions": 0,
                        "duration_seconds": 0,
                        "average_speed_kmh": 0,
                        "max_speed_kmh": 0,
                        "total_distance_km": 0,
                        "stops_count": 0,
                    },
                }, 200

            # Calculer analytics
            from shared.geo_utils import haversine_distance

            total_distance_km = 0.0
            speeds = []
            stops_count = 0
            last_position = None
            STOP_THRESHOLD_MS = 1.0  # < 1 m/s = arrêt

            for pos in positions:
                if last_position:
                    # Distance depuis dernière position
                    distance_km = haversine_distance(
                        last_position.latitude,
                        last_position.longitude,
                        pos.latitude,
                        pos.longitude,
                    )
                    total_distance_km += distance_km

                    # Détecter arrêts (vitesse < 1 m/s)
                    if pos.speed is not None and pos.speed < STOP_THRESHOLD_MS:
                        stops_count += 1

                # Collecter vitesses
                if pos.speed is not None and pos.speed > 0:
                    speeds.append(pos.speed * 3.6)  # m/s -> km/h

                last_position = pos

            # Durée du trajet
            duration_seconds = (
                (positions[-1].timestamp - positions[0].timestamp).total_seconds()
                if len(positions) > 1
                else 0
            )

            # Analytics
            average_speed_kmh = sum(speeds) / len(speeds) if speeds else 0.0
            max_speed_kmh = max(speeds) if speeds else 0.0

            return {
                "assignment_id": assignment_id,
                "booking_id": assignment.booking_id,
                "positions": [pos.to_dict() for pos in positions],
                "analytics": {
                    "total_positions": len(positions),
                    "duration_seconds": int(duration_seconds),
                    "average_speed_kmh": round(average_speed_kmh, 2),
                    "max_speed_kmh": round(max_speed_kmh, 2),
                    "total_distance_km": round(total_distance_km, 2),
                    "stops_count": stops_count,
                },
            }, 200

        except Exception as e:
            app_logger.exception("❌ Erreur trip tracking replay: %s", e)
            return {"error": f"Internal error: {e!s}"}, 500
