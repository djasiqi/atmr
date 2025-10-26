from __future__ import annotations

import contextlib
import json
import os
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from typing import cast as tcast

import requests
from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields
from sqlalchemy import or_

from ext import app_logger, db, redis_client, role_required, socketio
from models import Booking, BookingStatus, Driver, User, UserRole

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
driver_profile_model = driver_ns.model("DriverProfileUpdate", {
    "first_name": fields.String(description="Prénom"),
    "last_name": fields.String(description="Nom"),
    "phone": fields.String(description="Téléphone"),
    "status": fields.String(description="disponible | hors service"),
    # HR fields
    "contract_type": fields.String(description="CDI | CDD | HOURLY"),
    "weekly_hours": fields.Integer(description="Heures contrat / semaine"),
    "hourly_rate_cents": fields.Integer(description="Taux horaire (centimes)"),
    "employment_start_date": fields.String(description="YYYY-MM-DD"),
    "employment_end_date": fields.String(description="YYYY-MM-DD"),
    "license_categories": fields.List(fields.String, description="Ex: ['B','C1']"),
    "license_valid_until": fields.String(description="YYYY-MM-DD"),
    "trainings": fields.List(fields.Raw, description="[{name, valid_until}]"),
    "medical_valid_until": fields.String(description="YYYY-MM-DD"),
})

photo_model = driver_ns.model("DriverPhoto", {
    "photo": fields.String(required=True, description="Photo en Base64 ou URL")
})

location_model = driver_ns.model("DriverLocation", {
    "latitude": fields.Float(required=True, description="Latitude"),
    "longitude": fields.Float(required=True, description="Longitude"),
    "speed": fields.Float(required=False, description="Vitesse m/s"),
    "heading": fields.Float(required=False, description="Cap en degrés"),
    "accuracy": fields.Float(required=False, description="Précision en mètres"),
    "ts": fields.String(required=False, description="Horodatage ISO8601")
})

booking_status_model = driver_ns.model("BookingStatusUpdate", {
    "status": fields.String(
        required=True,
        description="Nouveau statut (en_route, in_progress, completed, return_completed)"
    )
})

availability_model = driver_ns.model(
    "DriverAvailability", {
        "is_available": fields.Boolean(
            required=True, description="Disponibilité du chauffeur")})

# ---------------------------
# Helpers
# ---------------------------


def get_driver_from_token() -> tuple[Driver | None, dict[str, Any] | None, int | None]:
    """Récupère le chauffeur associé à l'utilisateur connecté via le token JWT.
    Retourne (driver, None, None) si trouvé, sinon (None, error_response, status_code).
    """
    user_public_id = get_jwt_identity()
    app_logger.info("JWT Identity récupérée: {user_public_id}")

    user = User.query.filter_by(public_id=user_public_id).one_or_none()
    if not user:
        app_logger.error("User not found for public_id: {user_public_id}")
        return None, {"error": "User not found"}, 404

    app_logger.info("User details: id={user.id}, role={user.role}")

    if user.role != UserRole.driver:
        app_logger.error(
            f"User {getattr(user, 'username', user.id)} n'a pas le rôle 'driver'")
        return None, {"error": "Driver not found"}, 404

    driver = Driver.query.filter_by(user_id=user.id).one_or_none()
    if not driver:
        app_logger.error("Driver not found for user ID: {user.id}")
        return None, {"error": "Driver not found"}, 404

    app_logger.info(
        f"Driver found: {driver.id} for user {getattr(user, 'username', user.id)}")
    return driver, None, None


def notify_driver_new_booking(driver_id: int, booking: Booking) -> None:
    """Notifie le chauffeur d'une nouvelle mission assignée."""
    room = f"driver_{driver_id}"
    socketio.emit("new_booking", booking.to_dict(), to=room)
    app_logger.info(
        f"📤 new_booking émis vers {room} pour booking_id={booking.id}")


def notify_booking_update(driver_id: int, booking: Booking) -> None:
    """Notifie le chauffeur d'une mise à jour de mission."""
    room = f"driver_{driver_id}"
    # ✅ FIX: Émettre "new_booking" au lieu de "booking_updated" pour cohérence avec le mobile
    socketio.emit("new_booking", booking.to_dict(), to=room)
    app_logger.info(
        f"📤 new_booking (update) émis vers {room} pour booking_id={booking.id}")


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
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)
        return {"profile": driver.serialize}, 200

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(driver_profile_model)
    def put(self):
        """Met à jour le profil du chauffeur."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        data = request.get_json()
        app_logger.info("Payload reçu pour mise à jour du profil: {data}")
        if not data:
            return {"error": "Aucune donnée fournie"}, 400
        if not driver.user:
            return {"error": "Aucun utilisateur associé au driver"}, 500

        driver.user.first_name = data.get("first_name", driver.user.first_name)
        driver.user.last_name = data.get("last_name", driver.user.last_name)
        driver.user.phone = data.get("phone", driver.user.phone)

        status_val = data.get("status")
        if isinstance(status_val, str):
            val = status_val.strip().lower()
            if val == "disponible":
                driver.is_active = True
            elif val == "hors service":
                driver.is_active = False

        try:
            # HR optional updates
            ct = data.get("contract_type")
            if isinstance(ct, str) and ct:
                driver.contract_type = ct.upper()
            if "weekly_hours" in data:
                with contextlib.suppress(Exception):
                    driver.weekly_hours = int(data.get("weekly_hours")) if data.get(
                        "weekly_hours") is not None else None
            if "hourly_rate_cents" in data:
                with contextlib.suppress(Exception):
                    driver.hourly_rate_cents = int(data.get("hourly_rate_cents")) if data.get(
                        "hourly_rate_cents") is not None else None
            from datetime import date as _date

            def _parse_d(s):
                try:
                    return _date.fromisoformat(
                        s) if isinstance(s, str) and s else None
                except Exception:
                    return None
            if "employment_start_date" in data:
                driver.employment_start_date = _parse_d(
                    data.get("employment_start_date"))
            if "employment_end_date" in data:
                driver.employment_end_date = _parse_d(
                    data.get("employment_end_date"))
            if "license_categories" in data and isinstance(
                    data.get("license_categories"), list):
                driver.license_categories = list(
                    map(str, data.get("license_categories")))
            if "license_valid_until" in data:
                driver.license_valid_until = _parse_d(
                    data.get("license_valid_until"))
            if "trainings" in data and isinstance(data.get("trainings"), list):
                driver.trainings = data.get("trainings")
            if "medical_valid_until" in data:
                driver.medical_valid_until = _parse_d(
                    data.get("medical_valid_until"))

            db.session.commit()
            app_logger.info(
                f"Profil du driver {driver.id} mis à jour avec succès")
            return {"profile": driver.serialize,
                    "message": "Profil mis à jour avec succès"}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(
                f"❌ ERREUR update_driver_profile: {type(e).__name__} - {e!s}",
                exc_info=True)
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
        app_logger.info("Payload reçu pour mise à jour de la photo: {data}")
        if not data or "photo" not in data:
            return {"error": "Donnée photo non fournie"}, 400

        photo_data = data.get("photo")
        if not photo_data:
            return {"error": "Photo invalide"}, 400

        driver.driver_photo = photo_data
        try:
            db.session.commit()
            app_logger.info(
                f"Photo du driver {driver.id} mise à jour avec succès")
            return {"profile": driver.serialize,
                    "message": "Photo mise à jour avec succès"}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(
                f"❌ ERREUR update_driver_photo: {type(e).__name__} - {e!s}",
                exc_info=True)
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
        driver_name = f"{driver.user.first_name} {driver.user.last_name}" if driver.user else f"#{driver.id}"
        app_logger.info(
            f"📱 [Driver Bookings] Driver {driver_name} (ID: {driver.id}) loading bookings")

        from datetime import date

        from shared.time_utils import day_local_bounds

        # ✅ Récupérer les courses d'AUJOURD'HUI (passées et futures) tant qu'elles ne sont pas terminées
        today_start, today_end = day_local_bounds(
            date.today().strftime("%Y-%m-%d"))
        
        # S'assurer que ce sont des objets datetime pour SQLAlchemy
        from datetime import datetime
        today_start = datetime.fromisoformat(str(today_start))
        today_end = datetime.fromisoformat(str(today_end))

        status_pred = Booking.status.in_([
            BookingStatus.ASSIGNED,
            BookingStatus.EN_ROUTE,
            BookingStatus.IN_PROGRESS
        ])

            
        bookings = Booking.query.filter_by(driver_id=driver.id).filter(
            Booking.scheduled_time >= today_start,
            Booking.scheduled_time < today_end
        ).filter(status_pred).order_by(Booking.scheduled_time.asc()).all()

        # 🔍 LOG : Afficher les courses trouvées
        app_logger.info(
            f"📱 [Driver Bookings] Found {len(bookings)} bookings for driver {driver_name} (ID: {driver.id})")
        for b in bookings:
            app_logger.info(
                f"   - Booking #{b.id}: driver_id={b.driver_id}, client={b.customer_name}, time={b.scheduled_time}")

        return [b.serialize for b in bookings], 200


@driver_ns.route("/me/bookings/eta")
class DriverBookingsETA(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Calcule l'ETA dynamique pour toutes les missions du chauffeur basé sur sa position GPS actuelle."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from datetime import date, timedelta

        from services.unified_dispatch.data import calculate_eta as calc_eta
        from shared.time_utils import day_local_bounds, now_local

        # Récupérer les courses d'aujourd'hui (non terminées)
        today_start, today_end = day_local_bounds(
            date.today().strftime("%Y-%m-%d"))

        status_pred = Booking.status.in_([
            BookingStatus.ASSIGNED,
            BookingStatus.EN_ROUTE,
            BookingStatus.IN_PROGRESS
        ])

            
        bookings = Booking.query.filter_by(driver_id=driver.id).filter(
            Booking.scheduled_time >= today_start,
            Booking.scheduled_time < today_end
        ).filter(status_pred).order_by(Booking.scheduled_time.asc()).all()

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
                        "distance_meters": b.distance_meters
                    } for b in bookings
                ]
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
                    if dropoff_lat and dropoff_lon and booking.status != BookingStatus.IN_PROGRESS:
                        dropoff_pos = (float(dropoff_lat), float(dropoff_lon))
                        pickup_to_dropoff = calc_eta(pickup_pos, dropoff_pos)
                        total_duration = pickup_to_dropoff
                except Exception as e:
                    app_logger.warning(
                        f"ETA calculation failed for booking {booking.id}: {e}")

            results.append(
                {
                    "id": booking.id,
                    "eta_to_pickup_seconds": eta_to_pickup,
                    "duration_seconds": total_duration,
                    "distance_meters": booking.distance_meters,
                    "estimated_arrival": (
                        current_time +
                        timedelta(
                            seconds=eta_to_pickup)).isoformat() if eta_to_pickup else None})

        return {
            "has_gps": True,
            "driver_position": {"lat": driver_lat, "lon": driver_lon},
            "bookings": results
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
            app_logger.debug("📍 Received location data: {p} (type={type(p)})")

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
                    
                    if result is None and ((not (-LAT_THRESHOLD <= lat <= LAT_THRESHOLD)) or not (-LON_THRESHOLD <= lon <= LON_THRESHOLD)):
                        result = {"error": "Coordinates out of valid range"}
                        status_code = 400
                    
                    if result is None:
                        speed = float(p.get("speed", 0.0) or 0.0)
                        heading = float(p.get("heading", 0.0) or 0.0)
                        accuracy = float(p.get("accuracy", 0.0) or 0.0)
                        ts = p.get("ts") or datetime.now(UTC).isoformat()

                        OSRM = os.getenv("UD_OSRM_BASE_URL", "http://localhost:5001")
                        TTL = int(os.getenv("DRIVER_LOC_TTL_SEC", "600"))
                        MATCH_WINDOW = int(os.getenv("DRIVER_LOC_MATCH_WINDOW", "5"))

                        source = "raw"

                        # 1) Snap sur chaussée la plus proche
                        try:
                            r = requests.get(
                                f"{OSRM}/nearest/v1/driving/{lon},{lat}",
                                params={"number": 1}, timeout=2
                            )
                            if r.ok:
                                loc = r.json()["waypoints"][0]["location"]
                                lon, lat = float(loc[0]), float(loc[1])
                                source = "osrm_nearest"
                        except Exception:
                            pass

                        point = {
                            "ts": ts,
                            "lat": lat,
                            "lon": lon,
                            "speed": speed,
                            "heading": heading}

                        # 2) Ring buffer pour /match
                        try:
                            redis_ring: Any = redis_client
                            ring_key = f"driver:{driver.id}:ring"
                            redis_ring.lpush(ring_key, json.dumps(point))
                            redis_ring.ltrim(ring_key, 0, MATCH_WINDOW - 1)
                            redis_ring.expire(ring_key, TTL)
                        except Exception:
                            pass

                        # 3) Lissage map-matching si on a assez de points
                        try:
                            redis_match: Any = redis_client
                            pts_raw = redis_match.lrange(
                                f"driver:{driver.id}:ring", 0, MATCH_WINDOW - 1)
                            pts = [json.loads(x) for x in pts_raw] if pts_raw else []
                            if len(pts) >= MIN_POINTS_FOR_MATCHING:
                                coords = ";".join(
                                    f'{pp["lon"]:.6f},{pp["lat"]:.6f}' for pp in reversed(pts))
                                r2 = requests.get(
                                    f"{OSRM}/match/v1/driving/{coords}",
                                    params={"tidy": "true", "overview": "false"}, timeout=3
                                )
                                if r2.ok and r2.json().get("matchings"):
                                    tp = r2.json()["tracepoints"][-1]
                                    if tp and tp.get("location"):
                                        lon, lat = float(
                                            tp["location"][0]), float(
                                            tp["location"][1])
                                        source = "osrm_match"
                        except Exception:
                            pass

                        # 4) Sauvegarde DB + Redis
                        try:
                            driver.latitude = lat
                            driver.longitude = lon
                            db.session.commit()
                        except Exception as e:
                            db.session.rollback()
                            sentry_sdk.capture_exception(e)

                        try:
                            redis_loc: Any = redis_client
                            key = f"driver:{driver.id}:loc"
                            redis_loc.hset(key, mapping={
                                "company_id": driver.company_id,
                                "lat": lat, "lon": lon,
                                "speed": speed, "heading": heading,
                                "accuracy": accuracy, "ts": ts, "source": source
                            })
                            redis_loc.expire(key, TTL)
                        except Exception:
                            pass

                        # 5) Diffusion temps réel à la room entreprise
                        try:
                            room = f"company_{driver.company_id}"
                            socketio.emit("driver_location", {
                                "driver_id": driver.id,
                                "company_id": driver.company_id,
                                "lat": lat, "lon": lon,
                                "speed": speed, "heading": heading,
                                "accuracy": accuracy, "ts": ts, "source": source,
                            }, to=room)
                        except Exception:
                            pass

                        result = {"ok": True, "source": source,
                                 "message": "Location updated"}
                except (ValueError, TypeError):
                    result = {"error": "Invalid coordinate format"}
                    status_code = 400

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(
                "❌ Unexpected error in location update: %s",
                e,
                exc_info=True)
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
                id=booking_id, driver_id=driver.id).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404

            return {
                "id": booking.id,
                "customer_name": booking.customer_name or getattr(booking, "customer_full_name", None),
                "client_name": booking.customer_name or getattr(booking, "customer_full_name", None),
                "pickup_location": booking.pickup_location,
                "dropoff_location": booking.dropoff_location,
                "scheduled_time": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
                "amount": booking.amount,
                "status": booking.status.value if hasattr(booking.status, "value") else str(booking.status),
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
                exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


@driver_ns.route("/company/<int:company_id>/live-locations")
class CompanyLiveLocations(Resource):
    @jwt_required()
    def get(self, company_id: int):
        """Retourne la dernière position connue de tous les chauffeurs de l'entreprise."""
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


@driver_ns.route("/me/bookings/<int:booking_id>/status",
                 methods=["PUT", "OPTIONS"])
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
            app_logger.info("Body reçu pour status update: %s", data)

            try:
                driver, error_response, status_code = get_driver_from_token()
                if error_response:
                    app_logger.error(
                        "Driver not found for token: %s",
                        get_jwt_identity())
                    result = error_response
                else:
                    driver = cast("Driver", driver)

                    booking = Booking.query.filter_by(id=booking_id).first()
                    if not booking:
                        app_logger.error("Booking with id %s not found", booking_id)
                        result = {"error": "Booking not found"}
                        status_code = 404
                    elif booking.driver_id is None and booking.status == BookingStatus.PENDING:
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
                            "return_completed"]
                        if new_status_str not in valid_statuses:
                            result = {"error": "Invalid status"}
                            status_code = 400
                        else:
                            # EN ROUTE
                            if new_status_str == "en_route":
                                if booking.status == BookingStatus.EN_ROUTE:
                                    result = {"message": "Booking already en route"}
                                elif booking.status != BookingStatus.ASSIGNED:
                                    result = {"error": "Booking must be ASSIGNED before going en_route"}
                                    status_code = 400
                                else:
                                    booking.status = BookingStatus.EN_ROUTE

                            # EN COURS
                            elif new_status_str == "in_progress":
                                if booking.status == BookingStatus.IN_PROGRESS:
                                    result = {"message": "Booking already in progress"}
                                elif booking.status != BookingStatus.EN_ROUTE:
                                    result = {"error": "Booking must be en_route before starting"}
                                    status_code = 400
                                else:
                                    booking.status = BookingStatus.IN_PROGRESS
                                    booking.boarded_at = datetime.now(UTC)

                            # TERMINER (ALLER OU RETOUR SELON is_return)
                            elif new_status_str == "completed":
                                if booking.is_return:
                                    if booking.status == BookingStatus.RETURN_COMPLETED:
                                        result = {"message": "Return trip already completed"}
                                    elif booking.status != BookingStatus.IN_PROGRESS:
                                        result = {"error": "Booking must be in_progress before completing return"}
                                        status_code = 400
                                    else:
                                        booking.status = BookingStatus.RETURN_COMPLETED
                                        booking.completed_at = datetime.now(UTC)
                                elif booking.status == BookingStatus.COMPLETED:
                                    result = {"message": "Booking already completed"}
                                elif booking.status != BookingStatus.IN_PROGRESS:
                                    result = {"error": "Booking must be in_progress before completing"}
                                    status_code = 400
                                else:
                                    booking.status = BookingStatus.COMPLETED
                                    booking.completed_at = datetime.now(UTC)

                            # TERMINER RETOUR explicite
                            elif new_status_str == "return_completed":
                                if booking.status == BookingStatus.RETURN_COMPLETED:
                                    result = {"message": "Return trip already completed"}
                                elif booking.status != BookingStatus.IN_PROGRESS:
                                    result = {"error": "Booking must be in_progress before completing return"}
                                    status_code = 400
                                elif booking.is_return:
                                    booking.status = BookingStatus.RETURN_COMPLETED
                                    booking.completed_at = datetime.now(UTC)
                                else:
                                    result = {"error": "Not a return trip"}
                                    status_code = 400

                            if result is None:
                                db.session.commit()
                                driver_id = driver.id
                                notify_booking_update(driver_id, booking)
                                result = {"message": f"Booking status updated to {new_status_str}"}

            except Exception as e:
                sentry_sdk.capture_exception(e)
                app_logger.error(
                    "❌ ERREUR update_booking_status: %s - %s",
                    type(e).__name__,
                    str(e),
                    exc_info=True)
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
            driver = Driver.query.filter_by(
                user_id=current_user_id).one_or_none()
            if not driver:
                return {"error": "Unauthorized: Driver not found"}, 403

            booking = Booking.query.filter_by(
                id=booking_id, driver_id=driver.id).one_or_none()
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
                f"❌ ERREUR reject_booking: {type(e).__name__} - {e!s}",
                exc_info=True)
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
            driver = Driver.query.filter_by(
                user_id=current_user_id).one_or_none()
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
                exc_info=True)
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
            id=booking_id, driver_id=driver.id).one_or_none()
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
            app_logger.info("[push-token] payload={payload_raw}")
            payload: dict[str, Any] = tcast("dict[str, Any]", payload_raw)

            # token (expo/fcm) requis
            token_any: Any = payload.get("token") or payload.get(
                "expo_token") or payload.get("push_token")
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
                            f"[push-token] driver_id extrait du payload: {driver_id}")
                    except (ValueError, TypeError) as e:
                        app_logger.warning(
                            f"[push-token] Impossible de convertir driver_id={raw_id}: {e}")
                        result = {"error": f"Format de driverId invalide: {raw_id}"}
                        status_code = 400

                # 2) sinon on déduit depuis le JWT (user -> driver)
                if result is None and driver_id is None:
                    app_logger.info(
                        "[push-token] driver_id absent du payload, déduction depuis JWT")
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
                                result = {"error": "Chauffeur introuvable pour cet utilisateur."}
                                status_code = 404
                            else:
                                driver_id = int(drv.id)
                                app_logger.info(
                                    f"[push-token] driver_id déduit du JWT: {driver_id}")
                
                # 3) Validation finale et enregistrement
                if result is None:
                    driver = Driver.query.get(driver_id)
                    if not driver:
                        app_logger.error(
                            f"[push-token] Driver introuvable pour driver_id={driver_id}")
                        result = {"error": f"Chauffeur introuvable pour l'ID {driver_id}."}
                        status_code = 404
                    else:
                        # Enregistrement du token
                        driver.push_token = token
                        db.session.commit()

                        app_logger.info(
                            f"[push-token] ✅ Token enregistré avec succès pour driver_id={driver_id}")
                        result = {"message": "✅ Push token enregistré avec succès.",
                                "driver_id": driver_id}

        except Exception as e:
            db.session.rollback()
            app_logger.error(
                f"[push-token] ❌ Erreur serveur: {e!s}",
                exc_info=True)
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
        driver.vehicle_assigned = data.get(
            "vehicle_assigned", driver.vehicle_assigned)
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
            "ColumnElement[bool]", Booking.driver_id == driver_id)

        st_completed: ColumnElement[bool] = tcast(
            "ColumnElement[bool]", Booking.status == BookingStatus.COMPLETED
        )
        st_return_completed: ColumnElement[bool] = tcast(
            "ColumnElement[bool]", Booking.status == BookingStatus.RETURN_COMPLETED)
        status_clause = or_(st_completed, st_return_completed)

        trips = (
            Booking.query
            .filter(drv_clause)
            .filter(status_clause)
            .order_by(Booking.scheduled_time.desc())
            .all()
        )
        return [trip.serialize for trip in trips], 200
