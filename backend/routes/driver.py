from flask_restx import Namespace, Resource, fields
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import Driver, User, Booking, UserRole, BookingStatus
from ext import role_required, app_logger, db, redis_client, socketio
from datetime import datetime, timezone
from flask_socketio import emit, join_room, leave_room
import logging, re, os, json, traceback, requests

# sentry (si initialisé dans app.py, on garde try/except pour éviter ImportError en tests)
try:
    from app import sentry_sdk
except Exception:
    class _S: 
        def capture_exception(*a, **k): pass
    sentry_sdk = _S()

app_logger = logging.getLogger('app')

driver_ns = Namespace('driver', description='Gestion des chauffeurs')

# Modèles Swagger pour documentation (facultatif)
driver_profile_model = driver_ns.model('DriverProfileUpdate', {
    'first_name': fields.String(description="Prénom"),
    'last_name': fields.String(description="Nom"),
    'phone': fields.String(description="Téléphone")
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
        description="Nouveau statut (en_route, in_progress, completed)"
    )
})


availability_model = driver_ns.model('DriverAvailability', {
    'is_available': fields.Boolean(required=True, description="Disponibilité du chauffeur")
})

now = datetime.now(timezone.utc)


def get_driver_from_token():
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

    # Ajoutez des logs supplémentaires, si nécessaire
    # Par exemple, avant de rechercher le driver, loguez user.id et user.role :
    app_logger.info(f"User details: id={user.id}, role={user.role}")

    if user.role != UserRole.driver:
        app_logger.error(f"User {user.username} n'a pas le rôle 'driver'")
        return None, {"error": "Driver not found"}, 404

    driver = Driver.query.filter_by(user_id=user.id).one_or_none()
    if not driver:
        app_logger.error(f"Driver not found for user ID: {user.id}")
        return None, {"error": "Driver not found"}, 404

    app_logger.info(f"Driver found: {driver.id} for user {user.username}")
    return driver, None, None

def notify_driver_new_booking(driver_id, booking):
    room = f"driver_{driver_id}"
    socketio.emit("new_booking", booking.to_dict(), to=room)

def notify_booking_update(driver_id, booking):
    room = f"driver_{driver_id}"
    socketio.emit("booking_updated", booking.to_dict(), to=room)

def notify_booking_cancelled(driver_id, booking_id):
    room = f"driver_{driver_id}"
    socketio.emit("booking_cancelled", {"id": booking_id}, to=room)

@driver_ns.route('/me/profile')
class DriverProfile(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère le profil du chauffeur"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        return {"profile": driver.serialize}, 200

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(driver_profile_model)
    def put(self):
        """Met à jour le profil du chauffeur"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        data = request.get_json()
        app_logger.info(f"Payload reçu pour mise à jour du profil: {data}")
        if not data:
            return {"error": "Aucune donnée fournie"}, 400
        if not driver.user:
            return {"error": "Aucun utilisateur associé au driver"}, 500

        driver.user.first_name = data.get('first_name', driver.user.first_name)
        driver.user.last_name = data.get('last_name', driver.user.last_name)
        driver.user.phone = data.get('phone', driver.user.phone)
        status_val = data.get('status')
        if status_val:
            if status_val.lower() == "disponible":
                driver.is_active = True
            elif status_val.lower() == "hors service":
                driver.is_active = False
        try:
            db.session.commit()
            app_logger.info(f"Profil du driver {driver.id} mis à jour avec succès")
            return {"profile": driver.serialize, "message": "Profil mis à jour avec succès"}, 200
        except Exception as e:
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
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR update_driver_photo: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/bookings')
class DriverBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère uniquement les prochaines courses actives assignées"""
        from datetime import datetime

        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        now = datetime.now(timezone.utc)

        bookings = (
            Booking.query
            .filter(Booking.driver_id == driver.id)
            .filter(Booking.scheduled_time >= now)
            .filter(Booking.status.in_([
                BookingStatus.ASSIGNED,
                BookingStatus.EN_ROUTE,
                BookingStatus.IN_PROGRESS
            ]))

            .order_by(Booking.scheduled_time.asc())
            .all()
        )

        return [booking.serialize for booking in bookings], 200

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

        try:
            p = request.get_json(force=True)
            print(f"📍 Received location data: {p}")
            print(f"📍 Data type: {type(p)}")
            
            if not p:
                print("❌ No JSON data received")
                return {"error": "No data provided"}, 400
                
            if "latitude" not in p or "longitude" not in p:
                print(f"❌ Missing coordinates in: {list(p.keys())}")
                return {"error": "Latitude and longitude are required"}, 400

            # Validation et conversion
            try:
                lat = float(p["latitude"])
                lon = float(p["longitude"])
                print(f"📍 Parsed coordinates: lat={lat}, lon={lon}")
            except (ValueError, TypeError) as e:
                print(f"❌ Invalid coordinate format: {e}")
                return {"error": "Invalid coordinate format"}, 400

            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                print(f"❌ Coordinates out of range: lat={lat}, lon={lon}")
                return {"error": "Coordinates out of valid range"}, 400

            speed = float(p.get("speed", 0.0) or 0.0)
            heading = float(p.get("heading", 0.0) or 0.0)
            accuracy = float(p.get("accuracy", 0.0) or 0.0)
            ts = p.get("ts") or datetime.now(timezone.utc).isoformat()

            print(f"📍 Driver {driver.id} location: {lat}, {lon} (speed: {speed}, heading: {heading})")

            OSRM = os.getenv("UD_OSRM_BASE_URL", "http://localhost:5001")
            TTL = int(os.getenv("DRIVER_LOC_TTL_SEC", "600"))
            MATCH_WINDOW = int(os.getenv("DRIVER_LOC_MATCH_WINDOW", "5"))

            source = "raw"
            # 1) Snap léger sur chaussée la plus proche
            try:
                r = requests.get(f"{OSRM}/nearest/v1/driving/{lon:.6f},{lat:.6f}",
                                 params={"number": 1}, timeout=2)
                if r.ok:
                    loc = r.json()["waypoints"][0]["location"]
                    lon, lat = float(loc[0]), float(loc[1])
                    source = "osrm_nearest"
            except Exception:
                pass

            point = {"ts": ts, "lat": lat, "lon": lon, "speed": speed, "heading": heading}

            # 2) Ring buffer pour /match
            try:
                ring_key = f"driver:{driver.id}:ring"
                redis_client.lpush(ring_key, json.dumps(point))
                redis_client.ltrim(ring_key, 0, MATCH_WINDOW-1)
                redis_client.expire(ring_key, TTL)
            except Exception:
                pass

            # 3) Lissage map-matching si on a assez de points
            try:
                pts = [json.loads(x) for x in redis_client.lrange(ring_key, 0, MATCH_WINDOW-1)]
                if len(pts) >= 3:
                    coords = ";".join(f'{pp["lon"]:.6f},{pp["lat"]:.6f}' for pp in reversed(pts))
                    r2 = requests.get(f"{OSRM}/match/v1/driving/{coords}",
                                      params={"tidy": "true", "overview": "false"}, timeout=3)
                    if r2.ok and r2.json().get("matchings"):
                        tp = r2.json()["tracepoints"][-1]
                        if tp and tp.get("location"):
                            lon, lat = float(tp["location"][0]), float(tp["location"][1])
                            source = "osrm_match"
            except Exception:
                pass

            # 4) Sauvegarde DB (facultatif) + Redis (obligatoire)
            try:
                # on garde aussi dans la DB si tu veux une dernière position persistée
                driver.latitude = lat
                driver.longitude = lon
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                sentry_sdk.capture_exception(e)

            try:
                key = f"driver:{driver.id}:loc"
                redis_client.hset(key, mapping={
                    "company_id": driver.company_id,
                    "lat": lat, "lon": lon,
                    "speed": speed, "heading": heading,
                    "accuracy": accuracy, "ts": ts, "source": source
                })
                redis_client.expire(key, TTL)
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
            print(f"❌ Unexpected error in location update: {e}")
            print(f"❌ Request data: {request.get_data()}")
            return {"error": f"Internal error: {str(e)}"}, 500

@driver_ns.route('/me/bookings/<int:booking_id>')
class BookingDetails(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id):
        """Récupère les détails d'une réservation assignée au chauffeur"""
        try:
            current_user_id = get_jwt_identity()
            driver = Driver.query.filter_by(user_id=current_user_id).one_or_none()
            if not driver:
                return {"error": "Unauthorized: Driver not found"}, 403
            booking = Booking.query.filter_by(id=booking_id, driver_id=driver.id).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404
            return {
                "id": booking.id,
                "customer_name": booking.customer_name or booking.customer_full_name,
                "client_name": booking.customer_name or booking.customer_full_name,
                "pickup_location": booking.pickup_location,
                "dropoff_location": booking.dropoff_location,
                "scheduled_time": booking.scheduled_time.isoformat(),
                "amount": booking.amount,
                "status": booking.status.value
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_booking_details: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500
        
@driver_ns.route('/company/<int:company_id>/live-locations')
class CompanyLiveLocations(Resource):
    @jwt_required()
    def get(self, company_id):
        """Retourne la dernière position connue de tous les chauffeurs de l'entreprise."""
        try:
            drivers = Driver.query.filter_by(company_id=company_id).all()
            items = []
            for d in drivers:
                key = f"driver:{d.id}:loc"
                h = redis_client.hgetall(key)
                if h:
                    # redis renvoie bytes -> decode
                    def _dec(v): 
                        try: return v.decode()
                        except: return v
                    rec = {k.decode(): _dec(v) for k, v in h.items()}
                    # cast utiles
                    for kf in ("lat","lon","speed","heading","accuracy"):
                        if kf in rec:
                            try: rec[kf] = float(rec[kf])
                            except: pass
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
    def put(self, booking_id):
        data = request.get_json()
        app_logger.info("Body reçu pour status update: %s", data)
        if request.method == 'OPTIONS':
            return {}, 200

        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                app_logger.error("Driver not found for token: %s", get_jwt_identity())
                return error_response, status_code

            booking = Booking.query.filter_by(id=booking_id).first()
            if not booking:
                app_logger.error("Booking with id %s not found", booking_id)
                return {"error": "Booking not found"}, 404

            if booking.driver_id is None and booking.status == BookingStatus.PENDING:
                booking.driver_id = driver.id
            elif booking.driver_id != driver.id:
                return {"error": "Unauthorized access to this booking"}, 403

            data = request.get_json()
            if not data:
                return {"error": "Missing JSON payload"}, 400

            new_status_str = data.get("status")
            # Ici, on inclut tous les statuts valides, dont "return_completed"
            valid_statuses = [
                "en_route", "in_progress", "completed", "return_completed"
            ]
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

            # TERMINER ALLER
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


            # TERMINER RETOUR
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
            notify_booking_update(driver.id, booking)
            return {"message": f"Booking status updated to {new_status_str}"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR update_booking_status: %s - %s", type(e).__name__, str(e), exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/bookings/<int:booking_id>')
class RejectBooking(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def delete(self, booking_id):
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
            availability = data.get('is_available')
            if availability is None:
                return {"error": "Availability status is required"}, 400
            driver.is_available = availability
            db.session.commit()
            status_str = "available" if availability else "unavailable"
            return {"message": f"Driver is now {status_str}"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR update_availability: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/me/bookings')
class DriverBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère les réservations assignées au chauffeur"""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        bookings = Booking.query.filter_by(driver_id=driver.id).all()
        if not bookings:
            return {"message": "No bookings assigned"}, 404
        return [booking.serialize for booking in bookings], 200
    
@driver_ns.route('/me/bookings/<int:booking_id>/report')
class ReportBookingIssue(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=booking_id, driver_id=driver.id).one_or_none()
        if not booking:
            return {"error": "Booking not found"}, 404

        data = request.get_json()
        issue_message = data.get("issue")
        if not issue_message:
            return {"error": "Issue message is required"}, 400

        # Par exemple, enregistrer le problème dans un champ ou envoyer une notification
        booking.issue_report = issue_message  # Assurez-vous que ce champ existe dans le modèle Booking
        try:
            db.session.commit()
            return {"message": "Issue reported successfully"}, 200
        except Exception as e:
            db.session.rollback()
            return {"error": "Une erreur interne est survenue."}, 500

@driver_ns.route('/save-push-token')
class SavePushToken(Resource):
    @jwt_required()
    def post(self):
        try:
            data = request.get_json(force=True)
            driver_id = data.get('driverId')
            token = data.get('token')

            if not driver_id or not isinstance(driver_id, int):
                return {"error": "ID du chauffeur invalide ou manquant."}, 400

            if not token or not isinstance(token, str) or len(token) < 10:
                return {"error": "Token FCM invalide ou manquant."}, 400

            driver = Driver.query.get(driver_id)
            if not driver:
                return {"error": "Chauffeur introuvable."}, 404

            driver.push_token = token
            db.session.commit()

            return {"message": "✅ Push token enregistré avec succès."}, 200

        except Exception as e:
            traceback.print_exc()  # 🔥 important pour voir l'erreur exacte dans la console Flask
            return {"error": f"Erreur serveur : {str(e)}"}, 500

@driver_ns.route('/<int:driver_id>/update-profile')
class UpdateDriverProfile(Resource):
    @jwt_required()
    def post(self, driver_id):
        driver = Driver.query.get(driver_id)
        if not driver:
            return {"error": "Chauffeur non trouvé."}, 404

        data = request.get_json()
        driver.vehicle_assigned = data.get('vehicle_assigned')
        driver.brand = data.get('brand')
        driver.license_plate = data.get('license_plate')
        driver.driver_photo = data.get('photo')
        driver.user.phone = data.get('phone')

        db.session.commit()
        return {"message": "Profil mis à jour avec succès."}, 200

@driver_ns.route('/<int:driver_id>/completed-trips')
class CompletedTrips(Resource):
    @jwt_required()
    def get(self, driver_id):
        trips = Booking.query.filter(
            Booking.driver_id == driver_id,
            Booking.status.in_([BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED])
        ).order_by(Booking.scheduled_time.desc()).all()
        
        return [trip.serialize for trip in trips], 200


