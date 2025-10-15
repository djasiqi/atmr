from __future__ import annotations

import logging
from typing import Any, cast

import sentry_sdk
from flask import request, url_for
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields
from sqlalchemy.orm import joinedload

from ext import db, role_required
from models import Booking, BookingStatus, Client, Driver, User, UserRole
from services.maps import geocode_address, get_distance_duration
from services.unified_dispatch import queue
from shared.time_utils import to_utc

app_logger = logging.getLogger('app')

# Création du Namespace pour les réservations
bookings_ns = Namespace('bookings', description="Opérations relatives aux réservations")

# Modèle Swagger (ajout is_round_trip)
booking_create_model = bookings_ns.model(
    "BookingCreate",
    {
        "customer_name": fields.String(required=True),
        "pickup_location": fields.String(required=True),
        "dropoff_location": fields.String(required=True),
        "scheduled_time": fields.String(required=True, description="ISO 8601"),
        "amount": fields.Float(required=True),
        "medical_facility": fields.String(description="Établissement médical", default=""),
        "doctor_name": fields.String(description="Nom du médecin", default=""),
        "is_round_trip": fields.Boolean(description="Créer également un retour", default=False),
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
        app_logger.warning(f"⚠️ _queue_trigger failed: {e}")

# -----------------------------------------------------
# Helper: construit les liens de pagination RFC 5988
def _build_pagination_links(page: int, per_page: int, total: int, endpoint: str, **kwargs):
    """
    Construit les liens de pagination conformes RFC 5988.

    Returns:
        dict avec 'Link' header + metadata pagination
    """
    total_pages = (total + per_page - 1) // per_page
    links = []

    if page > 1:
        links.append(f'<{url_for(endpoint, page=page-1, per_page=per_page, **kwargs, _external=True)}>; rel="prev"')
    if page < total_pages:
        links.append(f'<{url_for(endpoint, page=page+1, per_page=per_page, **kwargs, _external=True)}>; rel="next"')

    links.append(f'<{url_for(endpoint, page=1, per_page=per_page, **kwargs, _external=True)}>; rel="first"')
    links.append(f'<{url_for(endpoint, page=total_pages, per_page=per_page, **kwargs, _external=True)}>; rel="last"')

    return {
        'Link': ', '.join(links),
        'X-Total-Count': str(total),
        'X-Page': str(page),
        'X-Per-Page': str(per_page),
        'X-Total-Pages': str(total_pages),
    }


# =====================================================
# Création d'une réservation pour un client
# =====================================================
@bookings_ns.route("/clients/<string:public_id>/bookings")
class CreateBooking(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    @bookings_ns.expect(booking_create_model)
    def post(self, public_id):
        """Créer une réservation pour un client (statut PENDING)."""
        try:
            data = request.get_json() or {}
            jwt_public_id = get_jwt_identity()

            user = User.query.filter_by(public_id=jwt_public_id).one_or_none()
            if not user:
                return {"message": "Utilisateur non authentifié"}, 401

            # Client propriétaire (via public_id fourni dans l’URL)
            client = Client.query.join(User).filter(User.public_id == public_id).one_or_none()
            if not client or client.user_id != user.id:
                return {"message": "Client non trouvé ou non associé à cet utilisateur"}, 403

            # Horaire (UTC-aware via helper) — interprète les naïfs en Europe/Zurich puis convertit en UTC
            try:
                # Interprète "YYYY-MM-DD HH:mm" (ou ISO sans Z) en Europe/Zurich et garde NAÏF (pas de tzinfo)
                from shared.time_utils import parse_local_naive
                scheduled_time = parse_local_naive(data["scheduled_time"])
            except Exception as date_error:
                app_logger.error(f"Erreur de conversion scheduled_time: {date_error}")
                return {"error": "Invalid scheduled_time format"}, 400

            # Durée/Distance (grâce à Google DM ou fallback coord si disponible)
            try:
                duration_seconds, distance_meters = get_distance_duration(
                    data["pickup_location"], data["dropoff_location"]
                )
            except Exception as e:
                app_logger.error(f"Distance Matrix error: {e}")
                return {"error": f"Erreur lors du calcul durée/distance: {e}"}, 400

            # Crée l’aller (PENDING)
            new_booking = cast(Any, Booking)(
                customer_name=data["customer_name"],
                pickup_location=data["pickup_location"],
                dropoff_location=data["dropoff_location"],
                scheduled_time=scheduled_time,
                amount=float(data["amount"]),
                status=BookingStatus.PENDING,
                user_id=user.id,
                client_id=client.id,
                company_id=client.company_id,  # lie déjà à l’entreprise si modèle le prévoit
                medical_facility=data.get("medical_facility", ""),
                doctor_name=data.get("doctor_name", ""),
                duration_seconds=duration_seconds,
                distance_meters=distance_meters,
                is_return=False,
            )
            db.session.add(new_booking)
            db.session.flush()  # pour obtenir new_booking.id

            # Géocodage (best effort, pas bloquant)
            try:
                # on passe les valeurs str de la requête (évite Column[str] -> str)
                c = geocode_address(data["pickup_location"])
                if c:
                    cast(Any, new_booking).pickup_lat, cast(Any, new_booking).pickup_lon = c
                c2 = geocode_address(data["dropoff_location"])
                if c2:
                    cast(Any, new_booking).dropoff_lat, cast(Any, new_booking).dropoff_lon = c2
            except Exception as e:
                app_logger.warning(f"Géocodage best-effort échoué: {e}")

            # Retour « placeholder » si demandé (toujours PENDING, éventuellement sans horaire)
            if bool(data.get("is_round_trip", False)):
                return_booking = cast(Any, Booking)(
                    customer_name=new_booking.customer_name,
                    pickup_location=new_booking.dropoff_location,
                    dropoff_location=new_booking.pickup_location,
                    scheduled_time=None,  # sera fixé par /trigger-return
                    amount=0,
                    status=BookingStatus.PENDING,
                    is_return=True,
                    parent_booking_id=new_booking.id,
                    user_id=user.id,
                    client_id=client.id,
                    company_id=client.company_id,
                    duration_seconds=duration_seconds,
                    distance_meters=distance_meters,
                )
                # calque les coords inversées si déjà connues
                try:
                    cast(Any, return_booking).pickup_lat = cast(Any, new_booking).dropoff_lat
                    cast(Any, return_booking).pickup_lon = cast(Any, new_booking).dropoff_lon
                    cast(Any, return_booking).dropoff_lat = cast(Any, new_booking).pickup_lat
                    cast(Any, return_booking).dropoff_lon = cast(Any, new_booking).pickup_lon
                except Exception:
                    pass
                db.session.add(return_booking)

            db.session.commit()

            # ⚠️ Pas de dispatch ici (PENDING seulement). L’entreprise acceptera -> ACCEPTED.
            return {
                "message": "Réservation créée avec succès",
                "booking_id": getattr(cast(Any, new_booking), "id", None)
            }, 201

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR create_booking: {type(e).__name__} - {e}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

#=====================================================
# Récupération, mise à jour et annulation d'une réservation
#=====================================================
@bookings_ns.route("/<int:booking_id>")
class BookingResource(Resource):
    @jwt_required()
    def get(self, booking_id):
        """Récupère une réservation (contrôle d'accès par rôle)."""
        try:
            public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=public_id).one_or_none()
            if not user:
                return {"error": "Utilisateur non authentifié"}, 401

            booking = Booking.query.get(booking_id)
            if not booking:
                return {"error": "Réservation introuvable"}, 404

            if user.role == UserRole.admin:
                return cast(Any, booking).serialize, 200

            # Client propriétaire (par user_id)
            client = Client.query.filter_by(user_id=user.id).one_or_none()
            if client and client.id == booking.client_id:
                return cast(Any, booking).serialize, 200

            # Chauffeur assigné ?
            if booking.driver and booking.driver.user.public_id == public_id:
                return cast(Any, booking).serialize, 200

            return {"error": "Accès non autorisé à cette réservation"}, 403

        except Exception as e:
            app_logger.error(f"❌ ERREUR get_booking: {type(e).__name__} - {e}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    def put(self, booking_id):
        """Met à jour une réservation (si PENDING). Déclenche queue si utile."""
        try:
            public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=public_id).one_or_none()
            if not user:
                return {"error": "Utilisateur non authentifié"}, 401

            booking = Booking.query.get(booking_id)
            if not booking:
                return {"error": "Réservation introuvable"}, 404

            # Admin ou client propriétaire (par user_id)
            client = Client.query.filter_by(user_id=user.id).one_or_none()
            if not (user.role == UserRole.admin or (client and client.id == booking.client_id)):
                return {"error": "Accès non autorisé à la modification"}, 403

            if booking.status != BookingStatus.PENDING:
                return {"error": "Seules les réservations en attente peuvent être modifiées"}, 400

            data = request.get_json() or {}
            booking.pickup_location = data.get("pickup_location", booking.pickup_location)
            booking.dropoff_location = data.get("dropoff_location", booking.dropoff_location)
            if "scheduled_time" in data:
                try:
                    booking.scheduled_time = to_utc(data["scheduled_time"])
                except Exception:
                    return {"error": "Format de date invalide"}, 400

            db.session.commit()

            # Pas de trigger si PENDING (non pris par l’engine). On log juste.
            return {"message": "Réservation mise à jour avec succès"}, 200

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR update_booking: {type(e).__name__} - {e}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    def delete(self, booking_id):
        """Annule une réservation (PENDING ou ASSIGNED). Déclenche queue si nécessaire."""
        try:
            public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=public_id).one_or_none()
            if not user:
                return {"error": "Utilisateur non authentifié"}, 401

            booking = Booking.query.get(booking_id)
            if not booking:
                return {"error": "Réservation introuvable"}, 404

            client = Client.query.filter_by(user_id=user.id).one_or_none()
            if not (user.role == UserRole.admin or (client and client.id == booking.client_id)):
                return {"error": "Accès non autorisé à l'annulation"}, 403

            if booking.status not in {BookingStatus.PENDING, BookingStatus.ASSIGNED}:
                return {"error": "Seules les réservations en attente ou confirmées peuvent être annulées"}, 400

            company_id = booking.company_id
            booking.status = BookingStatus.CANCELED
            db.session.commit()

            # Déclenche la queue seulement si la course impacte le dispatch
            if company_id and booking.status == BookingStatus.CANCELED:
                try:
                    cid = int(company_id)  # sécurise Column[int] -> int
                except Exception:
                    cid = None
                _queue_trigger(cid, "cancel")

            return {"message": "Réservation annulée avec succès"}, 200

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR cancel_booking: {type(e).__name__} - {e}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# =====================================================
# Liste selon le rôle (admin / client)
# =====================================================
@bookings_ns.route("/")
class ListBookings(Resource):
    @jwt_required()
    def get(self):
        """
        Retourne les réservations (paginées).

        Query params:
            - page: numéro de page (défaut: 1)
            - per_page: résultats par page (défaut: 100, max: 500)
            - status: filtre par statut (optionnel)
        """
        try:
            jwt_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=jwt_public_id).one_or_none()
            if not user:
                return {"error": "User not found"}, 401

            # Pagination
            page = int(request.args.get('page', 1))
            per_page = min(int(request.args.get('per_page', 100)), 500)
            status_filter = request.args.get('status')

            if user.role == UserRole.admin:
                query = Booking.query
                if status_filter:
                    query = query.filter_by(status=status_filter)
                pagination = query.paginate(page=page, per_page=per_page, error_out=False)
                total = pagination.total or 0
                bookings = pagination.items

                headers = _build_pagination_links(page, per_page, total, 'bookings.list_bookings')
                result = [cast(Any, b).serialize for b in bookings]
                return {"bookings": result, "total": total}, 200, headers

            elif user.role == UserRole.client:
                client = Client.query.filter_by(user_id=user.id).one_or_none()
                if not client:
                    return {"error": "Unauthorized: No client profile found"}, 403
                # ✅ Eager load client + user pour éviter N+1
                query = Booking.query.options(
                    joinedload(Booking.client).joinedload(Client.user),
                    joinedload(Booking.driver).joinedload(Driver.user),
                    joinedload(Booking.company)
                ).filter_by(client_id=client.id).order_by(Booking.scheduled_time.desc())

                if status_filter:
                    query = query.filter_by(status=status_filter)

                pagination = query.paginate(page=page, per_page=per_page, error_out=False)
                total = pagination.total or 0
                bookings = pagination.items

                headers = _build_pagination_links(page, per_page, total, 'bookings.list_bookings')
                result = [cast(Any, b).serialize for b in bookings]
                return {"bookings": result, "total": total}, 200, headers
            else:
                return {"error": "Unauthorized: You don't have permission"}, 403

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR list_bookings: {type(e).__name__} - {e}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500
