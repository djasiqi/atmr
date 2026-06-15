# routes/company_request_offers.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Routes pour la gestion des offres de transport côté entreprise.

Endpoints:
- GET /api/v1/company/request-offers - Lister les offres reçues
- POST /api/v1/company/request-offers/{id}/accept - Accepter une offre
- POST /api/v1/company/request-offers/{id}/reject - Rejeter une offre
"""

import logging

import sentry_sdk
from flask import request
from flask_jwt_extended import get_jwt, get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields

from application.institutions import AcceptOfferUseCase, RejectOfferUseCase
from application.institutions.accept_offer import AcceptOfferInput
from application.institutions.reject_offer import RejectOfferInput
from models import OfferStatus, RequestOffer
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)

logger = logging.getLogger(__name__)

# Namespace
company_offers_ns = Namespace(
    "company_request_offers",
    description="Gestion des offres de transport reçues (côté entreprise)",
)

# Modèles erreurs
api_error_model = create_api_error_model(company_offers_ns)
not_found_error_model = create_not_found_error_model(company_offers_ns)
permission_error_model = create_permission_error_model(company_offers_ns)
validation_error_model = create_validation_error_model(company_offers_ns)

# Modèles Swagger
transport_request_summary_model = company_offers_ns.model(
    "TransportRequestSummary",
    {
        "id": fields.Integer(description="ID de la demande"),
        "public_id": fields.String(description="ID public"),
        "external_reference": fields.String(description="Référence externe DPI"),
        "institution_id": fields.Integer(description="ID de l'institution"),
        "institution_name": fields.String(description="Nom de l'institution"),
        "mission_type": fields.String(description="Type de mission"),
        "delivery_description": fields.String(description="Description livraison"),
        "scheduled_time": fields.String(description="Heure prévue (ISO8601)"),
        "pickup_location": fields.String(description="Adresse de prise en charge"),
        "pickup_lat": fields.Float(description="Latitude pickup"),
        "pickup_lng": fields.Float(description="Longitude pickup"),
        "dropoff_location": fields.String(description="Adresse de destination"),
        "dropoff_lat": fields.Float(description="Latitude dropoff"),
        "dropoff_lng": fields.Float(description="Longitude dropoff"),
        "is_round_trip": fields.Boolean(description="Aller-retour"),
        "return_time": fields.String(description="Heure retour (ISO8601)"),
        "mobility": fields.Raw(description="Informations mobilité"),
        "contact_on_site": fields.Raw(description="Contact sur site"),
        "notes": fields.String(description="Notes"),
        "billing_intent": fields.String(description="Intention facturation"),
    },
)

offer_model = company_offers_ns.model(
    "RequestOffer",
    {
        "id": fields.Integer(description="ID de l'offre"),
        "status": fields.String(description="Statut de l'offre"),
        "mode": fields.String(description="Mode d'envoi (sequential/broadcast)"),
        "sent_at": fields.String(description="Date d'envoi (ISO8601)"),
        "expires_at": fields.String(description="Date d'expiration (ISO8601)"),
        "can_respond": fields.Boolean(description="Peut encore répondre"),
        "transport_request": fields.Nested(
            transport_request_summary_model,
            description="Détails de la demande",
        ),
    },
)

offers_list_model = company_offers_ns.model(
    "RequestOffersList",
    {
        "offers": fields.List(fields.Nested(offer_model)),
        "total": fields.Integer(description="Nombre total d'offres"),
    },
)

accept_result_model = company_offers_ns.model(
    "AcceptOfferResult",
    {
        "success": fields.Boolean(description="Succès de l'opération"),
        "offer_id": fields.Integer(description="ID de l'offre"),
        "booking_id": fields.Integer(description="ID du booking créé"),
        "transport_request_id": fields.Integer(description="ID de la demande"),
    },
)

reject_result_model = company_offers_ns.model(
    "RejectOfferResult",
    {
        "success": fields.Boolean(description="Succès de l'opération"),
        "offer_id": fields.Integer(description="ID de l'offre"),
        "escalated": fields.Boolean(description="Escalade déclenchée"),
        "next_offer_id": fields.Integer(description="ID de la prochaine offre"),
        "fallback_broadcast": fields.Boolean(
            description="Fallback broadcast déclenché"
        ),
    },
)


def get_company_context() -> tuple[int, int]:
    """Récupère le contexte company depuis le JWT.

    Returns:
        Tuple (company_id, user_id) où user_id est l'ID numérique

    Raises:
        Werkzeug Abort si non authentifié ou pas company
    """
    from flask import abort

    from models import User

    claims = get_jwt()
    company_id = claims.get("company_id")

    if not company_id:
        abort(403, description="Accès réservé aux utilisateurs entreprise")

    # get_jwt_identity() retourne le public_id (UUID), pas l'ID numérique.
    # Résoudre l'ID numérique si nécessaire.
    identity = get_jwt_identity()
    user_id = claims.get("user_id")
    if not user_id:
        # Résoudre depuis le public_id
        user = User.query.filter_by(public_id=identity).first()
        if not user:
            abort(401, description="Utilisateur introuvable")
        user_id = user.id

    return company_id, user_id


@company_offers_ns.route("")
class RequestOffersList(Resource):
    """Liste des offres reçues par l'entreprise."""

    @company_offers_ns.doc(
        description="Liste les offres de transport reçues",
        security="BearerAuth",
        params={
            "status": "Filtrer par statut (PENDING, ACCEPTED, REJECTED, etc.)",
        },
    )
    @company_offers_ns.response(200, "Succès", offers_list_model)
    @company_offers_ns.response(401, "Non authentifié", permission_error_model)
    @company_offers_ns.response(403, "Accès refusé", permission_error_model)
    @jwt_required()
    def get(self):
        """Liste les offres reçues par l'entreprise.

        Auth: JWT company requis
        """
        try:
            company_id, _user_id = get_company_context()

            # Filtrer par statut si spécifié
            status = request.args.get("status")
            if status and status not in OfferStatus.choices():
                return {
                    "error": f"Statut invalide. Valeurs possibles: {OfferStatus.choices()}",
                }, 400

            # Récupérer les offres
            offers = RequestOffer.find_by_company_and_status(
                company_id=company_id,
                status=status,
            )

            return {
                "offers": [o.serialize_for_company() for o in offers],
                "total": len(offers),
            }

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[CompanyOffers] Erreur GET offers: %s - %s",
                type(e).__name__,
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500


@company_offers_ns.route("/<int:offer_id>")
@company_offers_ns.param("offer_id", "ID de l'offre")
class RequestOfferDetail(Resource):
    """Détail d'une offre."""

    @company_offers_ns.doc(
        description="Récupère les détails d'une offre",
        security="BearerAuth",
    )
    @company_offers_ns.response(200, "Succès", offer_model)
    @company_offers_ns.response(401, "Non authentifié", permission_error_model)
    @company_offers_ns.response(403, "Accès refusé", permission_error_model)
    @company_offers_ns.response(404, "Offre non trouvée", not_found_error_model)
    @jwt_required()
    def get(self, offer_id: int):
        """Récupère les détails d'une offre.

        Auth: JWT company requis
        """
        try:
            company_id, _user_id = get_company_context()

            offer = RequestOffer.query.get(offer_id)
            if not offer:
                return {"error": "Offre non trouvée"}, 404

            if offer.company_id != company_id:
                return {"error": "Accès non autorisé à cette offre"}, 403

            return offer.serialize_for_company()

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[CompanyOffers] Erreur GET offer %s: %s",
                offer_id,
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500


@company_offers_ns.route("/<int:offer_id>/travel-estimate")
@company_offers_ns.param("offer_id", "ID de l'offre")
class OfferTravelEstimate(Resource):
    """Durée estimée du trajet aller (Google Directions côté serveur)."""

    @company_offers_ns.doc(
        description="Estime la durée du trajet aller pour une offre (Google Maps)",
        security="BearerAuth",
    )
    @company_offers_ns.response(401, "Non authentifié", permission_error_model)
    @company_offers_ns.response(403, "Accès refusé", permission_error_model)
    @company_offers_ns.response(404, "Offre non trouvée", not_found_error_model)
    @jwt_required()
    def get(self, offer_id: int):
        try:
            from services.institutions.route_travel_estimate_service import (
                estimate_outbound_travel_minutes,
            )

            company_id, _user_id = get_company_context()
            offer = RequestOffer.query.get(offer_id)
            if not offer:
                return {"error": "Offre non trouvée"}, 404
            if offer.company_id != company_id:
                return {"error": "Accès non autorisé à cette offre"}, 403

            payload = estimate_outbound_travel_minutes(offer.transport_request)
            return payload, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[CompanyOffers] Erreur GET travel-estimate offer %s: %s",
                offer_id,
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500


@company_offers_ns.route("/<int:offer_id>/accept")
@company_offers_ns.param("offer_id", "ID de l'offre")
class AcceptOffer(Resource):
    """Accepter une offre."""

    @company_offers_ns.doc(
        description="Accepte une offre de transport (crée un booking)",
        security="BearerAuth",
    )
    @company_offers_ns.response(200, "Succès", accept_result_model)
    @company_offers_ns.response(401, "Non authentifié", permission_error_model)
    @company_offers_ns.response(403, "Accès refusé", permission_error_model)
    @company_offers_ns.response(404, "Offre non trouvée", not_found_error_model)
    @company_offers_ns.response(409, "Offre déjà traitée", api_error_model)
    @company_offers_ns.response(410, "Offre expirée", api_error_model)
    @jwt_required()
    def post(self, offer_id: int):
        """Accepte une offre et crée le booking correspondant.

        Auth: JWT company requis

        Body optionnel:
            proposed_pickup_time (str): Horaire de prise en charge proposé (ISO8601).
                Si fourni, le booking sera créé avec cet horaire au lieu de celui
                demandé par l'institution.

        L'acceptation est atomique: si deux entreprises acceptent en même temps,
        une seule réussira (first accept wins).
        """
        try:
            company_id, user_id = get_company_context()

            # Parser l'horaire proposé si fourni
            data = request.get_json(silent=True) or {}
            proposed_pickup_time = None
            raw_time = data.get("proposed_pickup_time")
            if raw_time:
                from shared.time_utils import validate_proposed_pickup_time
                from services.metrics.institution_metrics import (
                    track_proposed_pickup_time_validation_failed,
                )

                proposed_pickup_time, validation_error = validate_proposed_pickup_time(
                    raw_time
                )
                if validation_error:
                    track_proposed_pickup_time_validation_failed(
                        company_id=company_id,
                        offer_id=offer_id,
                        reason=validation_error,
                    )
                    return {"error": validation_error}, 400

            use_case = AcceptOfferUseCase()
            result = use_case.execute(
                AcceptOfferInput(
                    offer_id=offer_id,
                    company_id=company_id,
                    user_id=user_id,
                    proposed_pickup_time=proposed_pickup_time,
                )
            )

            if not result.success:
                return {
                    "error": result.error,
                    "offer_id": result.offer_id,
                }, result.status_code

            resp = {
                "success": True,
                "offer_id": result.offer_id,
                "booking_id": result.booking_id,
                "transport_request_id": result.transport_request_id,
            }
            if result.return_booking_id is not None:
                resp["return_booking_id"] = result.return_booking_id
            return resp

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[CompanyOffers] Erreur POST accept %s: %s",
                offer_id,
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500


@company_offers_ns.route("/<int:offer_id>/reject")
@company_offers_ns.param("offer_id", "ID de l'offre")
class RejectOffer(Resource):
    """Rejeter une offre."""

    @company_offers_ns.doc(
        description="Rejette une offre de transport",
        security="BearerAuth",
    )
    @company_offers_ns.expect(
        company_offers_ns.model(
            "RejectOfferInput",
            {
                "reason": fields.String(
                    description="Raison du rejet (optionnel)",
                    required=False,
                ),
            },
        )
    )
    @company_offers_ns.response(200, "Succès", reject_result_model)
    @company_offers_ns.response(401, "Non authentifié", permission_error_model)
    @company_offers_ns.response(403, "Accès refusé", permission_error_model)
    @company_offers_ns.response(404, "Offre non trouvée", not_found_error_model)
    @company_offers_ns.response(409, "Offre déjà traitée", api_error_model)
    @jwt_required()
    def post(self, offer_id: int):
        """Rejette une offre.

        Auth: JWT company requis

        Si l'offre est en mode séquentiel, le rejet déclenche automatiquement
        l'escalade vers la préférence suivante ou le fallback broadcast.
        """
        try:
            company_id, user_id = get_company_context()

            data = request.get_json() or {}
            reason = data.get("reason")

            use_case = RejectOfferUseCase()
            result = use_case.execute(
                RejectOfferInput(
                    offer_id=offer_id,
                    company_id=company_id,
                    user_id=user_id,
                    reason=reason,
                )
            )

            if not result.success:
                return {
                    "error": result.error,
                    "offer_id": result.offer_id,
                }, result.status_code

            return {
                "success": True,
                "offer_id": result.offer_id,
                "escalated": result.escalated,
                "next_offer_id": result.next_offer_id,
                "fallback_broadcast": result.fallback_broadcast,
            }

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[CompanyOffers] Erreur POST reject %s: %s",
                offer_id,
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500
