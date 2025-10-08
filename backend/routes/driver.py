from __future__ import annotations

from flask import request
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Any, cast
from typing import cast as tcast
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy import or_
from datetime import datetime, timezone
import os
import json
import traceback
import requests

from models import Driver, User, Booking, UserRole, BookingStatus
from ext import role_required, app_logger, db, redis_client, socketio

# sentry (si initialisé dans app.py, on garde try/except pour éviter ImportError en tests)
try:
    from app import sentry_sdk
except Exception:
    class _S:
        def capture_exception(*a, **k): ...
    sentry_sdk = _S()

driver_ns = Namespace('driver', description='Gestion des chauffeurs')

# ---------------------------
# Modèles Swagger
# ---------------------------
driver_profile_model = driver_ns.model('DriverProfileUpdate', {
    'first_name': fields.String(description="Prénom"),
    'last_name': fields.String(description="Nom"),
    'phone': fields.String(description="Téléphone"),
    'status': fields.String(description="disponible | hors service"),
})

photo_model = driver_ns.model('DriverPhoto', {
    'photo': fields.String(required=True, description="Photo en Base64 ou URL")
})

location_model = driver_ns.model('DriverLocation', {
    'latitude':  fields.Float(required=True, description="Latitude"),
    'longitude': fields.Float(required=True, description="Longitude"),
    'speed':     fields.Float(required=False, description="Vitesse m/s"),
    'heading':   fields.Float(required=False, description="Cap en degrés"),
    'accuracy':  fields.Float(required=False, description="Précision en mètres"),
    'ts':        fields.String(required=False, description="Horodatage ISO8601")
})

booking_status_model = driver_ns.model('BookingStatusUpdate', {
    'status': fields.String(
        required=True,
        description="Nouveau statut (en_route, in_progress, completed, return_completed)"
    )
})

availability_model = driver_ns.model('DriverAvailability', {
    'is_available': fields.Boolean(required=True, description="Disponibilité du chauffeur")
})

# ---------------------------
# Helpers
# ---------------------------
def get_driver_from_token() -> tuple[Driver | None, dict | None, int | None]:
    """
    Récupère le chauffeur associé à l'utilisateur connecté via le token JWT.
    Retourne (driver, None, None) si trouvé, sinon (None, error_response, status_code).
    """
    user_public_id = get_jwt_identity()
    app_logger.info(f"JWT Identity récupérée: {user_public_id}")

    user = User.query.filter_by(public_id=user_public_id).one_or_none()
    if not user:
        app_logger.error(f"User not found for public_id: {user_public_id}")
        return None, {"error": "User not found"}, 404

    app_logger.info(f"User details: id={user.id}, role={user.role}")

    if user.role != UserRole.driver:
        app_logger.error(f"User {getattr(user, 'username', user.id)} n'a pas le rôle 'driver'")
        return None, {"error": "Driver not found"}, 404

    driver = Driver.query.filter_by(user_id=user.id).one_or_none()
    if not driver:
        app_logger.error(f"Driver not found for user ID: {user.id}")
        return None, {"error": "Driver not found"}, 404

    app_logger.info(f"Driver found: {driver.id} for user {getattr(user, 'username', user.id)}")
    return driver, None, None

def notify_driver_new_booking(driver_id: int, booking: Booking) -> None:
    """Notifie le chauffeur d'une nouvelle mission assignée."""
    room = f"driver_{driver_id}"
    socketio.emit("new_booking", booking.to_dict(), to=room)
    app_logger.info(f"📤 new_booking émis vers {room} pour booking_id={booking.id}")

def notify_booking_update(driver_id: int, booking: Booking) -> None:
    """Notifie le chauffeur d'une mise à jour de mission."""
    room = f"driver_{driver_id}"
    # ✅ FIX: Émettre "new_booking" au lieu de "booking_updated" pour cohérence avec le mobile
    socketio.emit("new_booking", booking.to_dict(), to=room)
    app_logger.info(f"📤 new_booking (update) émis vers {room} pour booking_id={booking.id}")

def notify_booking_cancelled(driver_id: int, booking_id: int) -> None:
    room = f"driver_{driver_id}"
    socketio.emit("booking_cancelled", {"id": booking_id}, to=room)

# ---------------------------
# Routes
# ---------------------------
@driver_ns.route('/me/profile')
class DriverProfile(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère le profil du chauffeur"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast(Driver, driver)
        return {"profile": driver.serialize}, 200

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(driver_profile_model)
    def put(self):
        """Met à jour le profil du chauffeur"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast(Driver, driver)

        data = request.get_json()
        app_logger.info(f"Payload reçu pour mise à jour du profil: {data}")
        if not data:
            return {"error": "Aucune donnée fournie"}, 400
        if not driver.user:
            return {"error": "Aucun utilisateur associé au driver"}, 500

        driver.user.first_name = data.get('first_name', driver.user.first_name)
        driver.user.last_name  = data.get('last_name',  driver.user.last_name)
        driver.user.phone      = data.get('phone',      driver.user.phone)

        status_val = data.get('status')
        if isinstance(status_val, str):
            val = status_val.strip().lower()
            if val == "disponible":
                driver.is_active = True
            elif val == "hors service":
                driver.is_active = False

        try:
            db.session.commit()
            app_logger.info(f"Profil du driver {driver.id} mis à jour avec succès")
            return {"profile": driver.serialize, "message": "Profil mis à jour avec succès"}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR update_driver_profile: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/photo')
class DriverPhoto(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(photo_model)
    def put(self):
        """Met à jour la photo du chauffeur"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast(Driver, driver)

        data = request.get_json()
        app_logger.info(f"Payload reçu pour mise à jour de la photo: {data}")
        if not data or 'photo' not in data:
            return {"error": "Donnée photo non fournie"}, 400

        photo_data = data.get('photo')
        if not photo_data:
            return {"error": "Photo invalide"}, 400

        driver.driver_photo = photo_data
        try:
            db.session.commit()
            app_logger.info(f"Photo du driver {driver.id} mise à jour avec succès")
            return {"profile": driver.serialize, "message": "Photo mise à jour avec succès"}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR update_driver_photo: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/bookings')
class DriverUpcomingBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast(Driver, driver)

        now = datetime.now(timezone.utc)

        status_pred: ColumnElement[bool] = or_(
            tcast(ColumnElement[bool], Booking.status == BookingStatus.ASSIGNED),
            tcast(ColumnElement[bool], Booking.status == BookingStatus.EN_ROUTE),
            tcast(ColumnElement[bool], Booking.status == BookingStatus.IN_PROGRESS),
        )

        bookings = (
            Booking.query
            .filter(tcast(ColumnElement[bool], Booking.driver_id == driver.id))
            .filter(tcast(ColumnElement[bool], Booking.scheduled_time >= now))
            .filter(status_pred)
            .order_by(Booking.scheduled_time.asc())
            .all()
        )
        return [b.serialize for b in bookings], 200

@driver_ns.route('/me/location')
class DriverLocation(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(location_model)
    def put(self):
        """Tracking temps réel : enregistre la dernière position"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast(Driver, driver)

        try:
            p = request.get_json(force=True)
            app_logger.debug(f"📍 Received location data: {p} (type={type(p)})")

            if not p:
                return {"error": "No data provided"}, 400
            if "latitude" not in p or "longitude" not in p:
                return {"error": "Latitude and longitude are required"}, 400

            # Validation et conversion
            try:
                lat = float(p["latitude"])
                lon = float(p["longitude"])
            except (ValueError, TypeError):
                return {"error": "Invalid coordinate format"}, 400

            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return {"error": "Coordinates out of valid range"}, 400

            speed   = float(p.get("speed",   0.0) or 0.0)
            heading = float(p.get("heading", 0.0) or 0.0)
            accuracy= float(p.get("accuracy",0.0) or 0.0)
            ts = p.get("ts") or datetime.now(timezone.utc).isoformat()

            OSRM = os.getenv("UD_OSRM_BASE_URL", "http://localhost:5001")
            TTL = int(os.getenv("DRIVER_LOC_TTL_SEC", "600"))
            MATCH_WINDOW = int(os.getenv("DRIVER_LOC_MATCH_WINDOW", "5"))

            source = "raw"

            # 1) Snap sur chaussée la plus proche
            try:
                r = requests.get(
                    f"{OSRM}/nearest/v1/driving/{lon:.6f},{lat:.6f}",
                    params={"number": 1}, timeout=2
                )
                if r.ok:
                    loc = r.json()["waypoints"][0]["location"]
                    lon, lat = float(loc[0]), float(loc[1])
                    source = "osrm_nearest"
            except Exception:
                pass

            point = {"ts": ts, "lat": lat, "lon": lon, "speed": speed, "heading": heading}

            # 2) Ring buffer pour /match
            try:
                rc: Any = redis_client
                ring_key = f"driver:{driver.id}:ring"
                rc.lpush(ring_key, json.dumps(point))
                rc.ltrim(ring_key, 0, MATCH_WINDOW - 1)
                rc.expire(ring_key, TTL)
            except Exception:
                pass

            # 3) Lissage map-matching si on a assez de points
            try:
                rc: Any = redis_client
                pts_raw = rc.lrange(f"driver:{driver.id}:ring", 0, MATCH_WINDOW - 1)
                pts = [json.loads(x) for x in pts_raw] if pts_raw else []
                if len(pts) >= 3:
                    coords = ";".join(f'{pp["lon"]:.6f},{pp["lat"]:.6f}' for pp in reversed(pts))
                    r2 = requests.get(
                        f"{OSRM}/match/v1/driving/{coords}",
                        params={"tidy": "true", "overview": "false"}, timeout=3
                    )
                    if r2.ok and r2.json().get("matchings"):
                        tp = r2.json()["tracepoints"][-1]
                        if tp and tp.get("location"):
                            lon, lat = float(tp["location"][0]), float(tp["location"][1])
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
                rc: Any = redis_client
                key = f"driver:{driver.id}:loc"
                rc.hset(key, mapping={
                    "company_id": driver.company_id,
                    "lat": lat, "lon": lon,
                    "speed": speed, "heading": heading,
                    "accuracy": accuracy, "ts": ts, "source": source
                })
                rc.expire(key, TTL)
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

            return {"ok": True, "source": source, "message": "Location updated"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ Unexpected error in location update: %s", e, exc_info=True)
            app_logger.error("❌ Request data: %s", request.get_data())
            return {"error": f"Internal error: {str(e)}"}, 500

@driver_ns.route('/me/bookings/<int:booking_id>')
class BookingDetails(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                return error_response, status_code
            driver = cast(Driver, driver)

            booking = Booking.query.filter_by(id=booking_id, driver_id=driver.id).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404

            return {
                "id": booking.id,
                "customer_name": booking.customer_name or getattr(booking, "customer_full_name", None),
                "client_name":   booking.customer_name or getattr(booking, "customer_full_name", None),
                "pickup_location": booking.pickup_location,
                "dropoff_location": booking.dropoff_location,
                "scheduled_time": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
                "amount": booking.amount,
                "status": booking.status.value if hasattr(booking.status, "value") else str(booking.status),
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_booking_details: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


@driver_ns.route('/company/<int:company_id>/live-locations')
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
                # h peut être dict[bytes, bytes] ; on force le cast pour Pylance
                rec = {(_dec(k)): _dec(v) for k, v in cast(dict, h).items()}

                for kf in ("lat", "lon", "speed", "heading", "accuracy"):
                    if kf in rec:
                        try:
                            rec[kf] = float(rec[kf])
                        except Exception:
                            pass
                items.append({"driver_id": d.id, **rec})

            return {"items": items}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return {"items": []}, 200

@driver_ns.route('/me/bookings/<int:booking_id>/status', methods=['PUT', 'OPTIONS'])
class UpdateBookingStatus(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(booking_status_model)
    def put(self, booking_id: int):
        if request.method == 'OPTIONS':
            return {}, 200

        data = request.get_json()
        app_logger.info("Body reçu pour status update: %s", data)

        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                app_logger.error("Driver not found for token: %s", get_jwt_identity())
                return error_response, status_code
            driver = cast(Driver, driver)

            booking = Booking.query.filter_by(id=booking_id).first()
            if not booking:
                app_logger.error("Booking with id %s not found", booking_id)
                return {"error": "Booking not found"}, 404

            if booking.driver_id is None and booking.status == BookingStatus.PENDING:
                booking.driver_id = driver.id
            elif booking.driver_id != driver.id:
                return {"error": "Unauthorized access to this booking"}, 403

            if not data:
                return {"error": "Missing JSON payload"}, 400

            new_status_str = data.get("status")
            valid_statuses = ["en_route", "in_progress", "completed", "return_completed"]
            if new_status_str not in valid_statuses:
                return {"error": "Invalid status"}, 400

            # EN ROUTE
            if new_status_str == "en_route":
                if booking.status == BookingStatus.EN_ROUTE:
                    return {"message": "Booking already en route"}, 200
                if booking.status != BookingStatus.ASSIGNED:
                    return {"error": "Booking must be ASSIGNED before going en_route"}, 400
                booking.status = BookingStatus.EN_ROUTE

            # EN COURS
            elif new_status_str == "in_progress":
                if booking.status == BookingStatus.IN_PROGRESS:
                    return {"message": "Booking already in progress"}, 200
                if booking.status != BookingStatus.EN_ROUTE:
                    return {"error": "Booking must be en_route before starting"}, 400
                booking.status = BookingStatus.IN_PROGRESS
                booking.boarded_at = datetime.now(timezone.utc)

            # TERMINER (ALLER OU RETOUR SELON is_return)
            elif new_status_str == "completed":
                if booking.is_return:
                    if booking.status == BookingStatus.RETURN_COMPLETED:
                        return {"message": "Return trip already completed"}, 200
                    if booking.status != BookingStatus.IN_PROGRESS:
                        return {"error": "Booking must be in_progress before completing return"}, 400
                    booking.status = BookingStatus.RETURN_COMPLETED
                    booking.completed_at = datetime.now(timezone.utc)
                else:
                    if booking.status == BookingStatus.COMPLETED:
                        return {"message": "Booking already completed"}, 200
                    if booking.status != BookingStatus.IN_PROGRESS:
                        return {"error": "Booking must be in_progress before completing"}, 400
                    booking.status = BookingStatus.COMPLETED
                    booking.completed_at = datetime.now(timezone.utc)

            # TERMINER RETOUR explicite
            elif new_status_str == "return_completed":
                if booking.status == BookingStatus.RETURN_COMPLETED:
                    return {"message": "Return trip already completed"}, 200
                if booking.status != BookingStatus.IN_PROGRESS:
                    return {"error": "Booking must be in_progress before completing return"}, 400
                if booking.is_return:
                    booking.status = BookingStatus.RETURN_COMPLETED
                    booking.completed_at = datetime.now(timezone.utc)
                else:
                    return {"error": "Not a return trip"}, 400

            db.session.commit()
            driver_id: int = tcast(int, driver.id)
            notify_booking_update(driver_id, booking)
            return {"message": f"Booking status updated to {new_status_str}"}, 200


        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR update_booking_status: %s - %s", type(e).__name__, str(e), exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/bookings/<int:booking_id>')
class RejectBooking(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def delete(self, booking_id: int):
        """Réjette une réservation assignée"""
        try:
            current_user_id = get_jwt_identity()
            driver = Driver.query.filter_by(user_id=current_user_id).one_or_none()
            if not driver:
                return {"error": "Unauthorized: Driver not found"}, 403

            booking = Booking.query.filter_by(id=booking_id, driver_id=driver.id).one_or_none()
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
            app_logger.error(f"❌ ERREUR reject_booking: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/availability')
class UpdateAvailability(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(availability_model)
    def put(self):
        """Met à jour la disponibilité du chauffeur"""
        try:
            current_user_id = get_jwt_identity()
            driver = Driver.query.filter_by(user_id=current_user_id).one_or_none()
            if not driver:
                return {"error": "Unauthorized: Driver not found"}, 403

            data = request.get_json()
            availability = data.get('is_available') if data else None
            if availability is None:
                return {"error": "Availability status is required"}, 400

            driver.is_available = bool(availability)
            db.session.commit()
            status_str = "available" if availability else "unavailable"
            return {"message": f"Driver is now {status_str}"}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR update_availability: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/bookings/all')
class DriverAllBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère toutes les réservations assignées au chauffeur"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast(Driver, driver)

        bookings = Booking.query.filter_by(driver_id=driver.id).all()
        if not bookings:
            return {"message": "No bookings assigned"}, 404
        return [b.serialize for b in bookings], 200

@driver_ns.route('/me/bookings/<int:booking_id>/report')
class ReportBookingIssue(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast(Driver, driver)

        booking = Booking.query.filter_by(id=booking_id, driver_id=driver.id).one_or_none()
        if not booking:
            return {"error": "Booking not found"}, 404

        data = request.get_json()
        issue_message = (data or {}).get("issue")
        if not issue_message:
            return {"error": "Issue message is required"}, 400

        # Assure-toi que ce champ existe dans le modèle Booking
        setattr(booking, "issue_report", issue_message)
        try:
            db.session.commit()
            return {"message": "Issue reported successfully"}, 200
        except Exception:
            db.session.rollback()
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/save-push-token')
class SavePushToken(Resource):
    @jwt_required()
    def post(self):
        try:
            # Log & typage strict
            payload_raw = request.get_json(force=True) or {}
            app_logger.info(f"[push-token] payload={payload_raw}")
            payload: dict[str, Any] = tcast(dict[str, Any], payload_raw)

            # token (expo/fcm) requis
            token_any: Any = payload.get('token') or payload.get('expo_token') or payload.get('push_token')
            if not isinstance(token_any, str) or len(token_any) < 10:
                return {"error": "Token FCM/Expo invalide ou manquant."}, 400
            token: str = token_any

            # 1) si driverId fourni -> on essaye de le caster
            driver_id: int | None = None
            raw_id: Any = payload.get('driverId') or payload.get('driver_id')
            if raw_id is not None:
                try:
                    # Convertir en float d'abord pour gérer les nombres décimaux, puis en int
                    driver_id = int(float(raw_id))
                    app_logger.info(f"[push-token] driver_id extrait du payload: {driver_id}")
                except (ValueError, TypeError) as e:
                    app_logger.warning(f"[push-token] Impossible de convertir driver_id={raw_id}: {e}")
                    return {"error": f"Format de driverId invalide: {raw_id}"}, 400

            # 2) sinon on déduit depuis le JWT (user -> driver)
            if driver_id is None:
                app_logger.info("[push-token] driver_id absent du payload, déduction depuis JWT")
                user_pid = get_jwt_identity()
                if not user_pid:
                    return {"error": "Token JWT invalide ou expiré."}, 401                
 
                user = User.query.filter_by(public_id=user_pid).one_or_none()
                if not user:
                    return {"error": "Utilisateur non trouvé pour le JWT."}, 404
                drv = Driver.query.filter_by(user_id=user.id).one_or_none()
                if not drv:
                    return {"error": "Chauffeur introuvable pour cet utilisateur."}, 404
                driver_id = int(drv.id)
                app_logger.info(f"[push-token] driver_id déduit du JWT: {driver_id}")
            # 3) Validation finale et enregistrement
            driver = Driver.query.get(driver_id)
            if not driver:
               app_logger.error(f"[push-token] Driver introuvable pour driver_id={driver_id}")
               return {"error": f"Chauffeur introuvable pour l'ID {driver_id}."}, 404

            # Enregistrement du token
            driver.push_token = token
            db.session.commit()

            app_logger.info(f"[push-token] ✅ Token enregistré avec succès pour driver_id={driver_id}")
            return {"message": "✅ Push token enregistré avec succès.", "driver_id": driver_id}, 200

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"[push-token] ❌ Erreur serveur: {str(e)}", exc_info=True)
            traceback.print_exc()
            return {"error": f"Erreur serveur : {str(e)}"}, 500

@driver_ns.route('/<int:driver_id>/update-profile')
class UpdateDriverProfile(Resource):
    @jwt_required()
    def post(self, driver_id: int):
        driver = Driver.query.get(driver_id)
        if not driver:
            return {"error": "Chauffeur non trouvé."}, 404

        data = request.get_json() or {}
        driver.vehicle_assigned = data.get('vehicle_assigned', driver.vehicle_assigned)
        driver.brand            = data.get('brand',            driver.brand)
        driver.license_plate    = data.get('license_plate',    driver.license_plate)
        driver.driver_photo     = data.get('photo',            driver.driver_photo)
        if driver.user:
            driver.user.phone   = data.get('phone',            driver.user.phone)

        db.session.commit()
        return {"message": "Profil mis à jour avec succès."}, 200

@driver_ns.route('/<int:driver_id>/completed-trips')
class CompletedTrips(Resource):
    @jwt_required()
    def get(self, driver_id: int):
        # Chaque clause est castée en ColumnElement[bool] pour Pylance
        drv_clause: ColumnElement[bool] = tcast(ColumnElement[bool], Booking.driver_id == driver_id)

        st_completed: ColumnElement[bool] = tcast(
            ColumnElement[bool], Booking.status == BookingStatus.COMPLETED
        )
        st_return_completed: ColumnElement[bool] = tcast(
            ColumnElement[bool], Booking.status == BookingStatus.RETURN_COMPLETED
        )
        status_clause: ColumnElement[bool] = tcast(
            ColumnElement[bool], or_(st_completed, st_return_completed)
        )

        trips = (
            Booking.query
            .filter(drv_clause)
            .filter(status_clause)
            .order_by(Booking.scheduled_time.desc())
            .all()
        )
        return [trip.serialize for trip in trips], 200