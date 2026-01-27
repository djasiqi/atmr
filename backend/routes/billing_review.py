"""Routes API pour le contrôle facturation (P5).

Endpoints:
- GET /billing/monthly-review : Liste des bookings avec filtres (mois, statut, payeur, clinique)
- POST /billing/bookings/{id}/set-payer : Modifier le payeur d'un booking
- POST /billing/bookings/{id}/lock : Verrouiller un booking
- POST /billing/bookings/{id}/unlock : Déverrouiller un booking
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from flask import request
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)
from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]
from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload

from ext import db, role_required
from models import BillingAuditLog, Booking, Client, ClientStay
from models.enums import BillingReviewStatus, BookingStatus, UserRole
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from schemas.billing_review_schemas import (
    BatchSetPayerRequestSchema,
    BillingReviewListQuerySchema,
    LockBookingRequestSchema,
    SetPayerRequestSchema,
    UnlockBookingRequestSchema,
)
from schemas.validation_utils import handle_validation_error, validate_request
from shared.error_handlers import APIErrorHandler
from shared.response_helpers import success_response


def _round_chf_005(value: float | None) -> float:
    """Arrondir au centime suisse (0.05)."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    d = Decimal(str(v))
    quant = Decimal("0.05")
    return float((d / quant).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * quant)


logger = logging.getLogger(__name__)

# Namespace pour le contrôle facturation
billing_review_ns = Namespace(
    "billing", description="Contrôle facturation et workflow pré-facturation"
)

# Modèles d'erreur standardisés
api_error_model = create_api_error_model(billing_review_ns)
validation_error_model = create_validation_error_model(billing_review_ns)
not_found_error_model = create_not_found_error_model(billing_review_ns)
permission_error_model = create_permission_error_model(billing_review_ns)

# Modèles Swagger pour la réponse
billing_review_item_model = billing_review_ns.model(
    "BillingReviewItem",
    {
        "booking_id": fields.Integer(required=True),
        "date": fields.String(required=True),
        "patient_name": fields.String(required=True),
        "payer_name": fields.String(required=True),
        "payer_type": fields.String(required=True),
        "billing_source": fields.String(allow_none=True),
        "billing_source_ref": fields.String(allow_none=True),
        "status": fields.String(required=True),
        "amount": fields.Float(required=True),
        "has_conflict": fields.Boolean(required=True),
        "has_unvalidated_voucher": fields.Boolean(required=True),
        "missing_recipient": fields.Boolean(required=True),
        "billing_party_id": fields.Integer(allow_none=True),
        "clinic_name": fields.String(allow_none=True),
    },
)


def _get_current_user_id() -> int | None:
    """Récupère l'ID de l'utilisateur actuel depuis le JWT."""
    try:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get("user_id")
        if isinstance(identity, int):
            return identity
        return None
    except Exception:
        return None


def _get_current_company_id() -> int | None:
    """Récupère l'ID de l'entreprise actuelle depuis le JWT."""
    try:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get("company_id")
        return None
    except Exception:
        return None


def _serialize_billing_review_item(booking: Booking) -> dict[str, Any]:
    """Sérialise un booking pour l'affichage dans le contrôle facturation."""
    from models import Company  # Import local pour éviter cycles
    from services.billing.client_stay_resolver import find_active_stay_for_booking

    client = booking.client
    client_user = client.user if client else None
    patient_name = (
        f"{client_user.first_name or ''} {client_user.last_name or ''}".strip()
        if client_user
        else (booking.customer_full_name or "Non spécifié")
    )

    # Déterminer le payeur
    payer_name = "Patient"
    payer_type = "patient"
    billing_party_id = None
    if booking.billing_party:
        payer_name = booking.billing_party.display_name
        payer_type = booking.billing_party.type.value
        billing_party_id = booking.billing_party.id
    elif booking.billed_to_company:
        payer_name = booking.billed_to_company.name
        payer_type = "company"

    # Détecter les alertes
    has_conflict = bool(booking.billing_override_reason)
    has_unvalidated_voucher = False
    if booking.transport_vouchers:
        has_unvalidated_voucher = any(
            v.status.value not in ("validated", "rejected", "expired")
            for v in booking.transport_vouchers
        )
    missing_recipient = (
        booking.billed_to_type != "patient"
        and not booking.billing_party_id
        and not booking.billed_to_company_id
    )

    # Clinique (si séjour actif)
    clinic_name = None
    if client:
        active_stay = find_active_stay_for_booking(booking=booking)
        if active_stay and active_stay.company_id:
            clinic_company = (
                db.session.query(Company).filter(Company.id == active_stay.company_id).first()
            )
            if clinic_company:
                clinic_name = clinic_company.name

    scheduled_dt = booking.scheduled_time
    date_str = scheduled_dt.strftime("%Y-%m-%d") if scheduled_dt else "N/A"

    return {
        "booking_id": booking.id,
        "date": date_str,
        "patient_name": patient_name,
        "payer_name": payer_name,
        "payer_type": payer_type,
        "billing_source": booking.billing_source.value if booking.billing_source else None,
        "billing_source_ref": booking.billing_source_ref,
        "status": booking.billing_review_status.value,
        "amount": float(booking.amount or 0),
        "has_conflict": has_conflict,
        "has_unvalidated_voucher": has_unvalidated_voucher,
        "missing_recipient": missing_recipient,
        "billing_party_id": billing_party_id,
        "clinic_name": clinic_name,
    }


@billing_review_ns.route("/monthly-review")
class BillingMonthlyReview(Resource):
    """Liste des bookings pour le contrôle facturation mensuel."""

    @jwt_required()
    @role_required(UserRole.company)
    @billing_review_ns.doc(
        params={
            "company_id": "ID de l'entreprise",
            "year": "Année (ex: 2026)",
            "month": "Mois (1-12)",
            "status": "Statut (draft, needs_review, ready, locked)",
            "billing_party_id": "ID du tiers payeur (optionnel)",
            "clinic_id": "ID de la clinique (optionnel)",
        }
    )
    @billing_review_ns.marshal_list_with(billing_review_item_model)
    def get(self):
        """Récupère la liste des bookings pour le contrôle facturation."""
        try:
            # Valider les paramètres de requête
            try:
                validated = validate_request(
                    BillingReviewListQuerySchema(), request.args, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            company_id = validated["company_id"]
            year = validated["year"]
            month = validated["month"]
            status_filter = validated.get("status")
            billing_party_id_filter = validated.get("billing_party_id")
            clinic_id_filter = validated.get("clinic_id")

            # Vérifier que l'utilisateur a accès à cette entreprise
            current_company_id = _get_current_company_id()
            if current_company_id and current_company_id != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Accès non autorisé à cette entreprise",
                    logger_instance=logger,
                )

            # Construire la plage de dates pour le mois
            MONTHS_PER_YEAR = 12
            start_date = datetime(year, month, 1)
            if month == MONTHS_PER_YEAR:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)

            # Construire la requête avec eager loading
            query = (
                db.session.query(Booking)
                .options(
                    joinedload(Booking.client).joinedload(Client.user),
                    joinedload(Booking.billing_party),
                    joinedload(Booking.billed_to_company),
                    joinedload(Booking.transport_vouchers),
                )
                .filter(
                    Booking.company_id == company_id,
                    Booking.scheduled_time >= start_date,
                    Booking.scheduled_time < end_date,
                    Booking.status.in_(
                        [
                            BookingStatus.COMPLETED.value,
                            BookingStatus.RETURN_COMPLETED.value,
                        ]
                    ),
                )
            )

            # Appliquer les filtres optionnels
            if status_filter:
                query = query.filter(
                    Booking.billing_review_status == BillingReviewStatus(status_filter)
                )
            if billing_party_id_filter:
                query = query.filter(Booking.billing_party_id == billing_party_id_filter)
            if clinic_id_filter:
                # Filtrer par clinique via les séjours actifs
                # Un booking est lié à une clinique s'il existe un séjour actif
                # pour le même client dont la date du booking tombe dans l'intervalle du séjour
                # et dont le company_id du séjour correspond à la clinique filtrée
                query = query.join(
                    ClientStay,
                    and_(
                        ClientStay.client_id == Booking.client_id,
                        ClientStay.company_id == clinic_id_filter,
                        ClientStay.status == "active",
                        ClientStay.start_date <= Booking.scheduled_time,
                        or_(
                            ClientStay.end_date.is_(None),
                            ClientStay.end_date >= Booking.scheduled_time,
                        ),
                    ),
                ).distinct()

            # Trier par date puis par ID
            bookings = query.order_by(Booking.scheduled_time.desc(), Booking.id.desc()).all()

            # Sérialiser les résultats
            items = [_serialize_billing_review_item(b) for b in bookings]

            return success_response(data=items)

        except Exception as e:
            logger.exception(
                "[BillingReview] Erreur lors de la récupération de la liste: %s",
                e,
            )
            return APIErrorHandler.handle_exception(
                e,
                logger_instance=logger,
                default_message="Erreur lors de la récupération de la liste",
            )


@billing_review_ns.route("/bookings/<int:booking_id>/set-payer")
class SetBookingPayer(Resource):
    """Modifier le payeur d'un booking."""

    @jwt_required()
    @role_required(UserRole.company)
    @billing_review_ns.expect(
        billing_review_ns.model(
            "SetPayerRequest",
            {
                "billed_to_type": fields.String(required=True),
                "billing_party_id": fields.Integer(allow_none=True),
                "billed_to_company_id": fields.Integer(allow_none=True),
                "reason": fields.String(required=True),
            },
        ),
        validate=False,
    )
    def post(self, booking_id: int):
        """Modifie le payeur d'un booking avec audit logging."""
        try:
            body = request.get_json(force=True, silent=True)
            if body is None:
                logger.warning(
                    "[BillingReview] set-payer booking_id=%s: body JSON manquant ou invalide — Content-Type=%s, Content-Length=%s, path=%s",
                    booking_id,
                    request.content_type,
                    request.content_length,
                    request.path,
                )
                return APIErrorHandler.handle_validation_error(
                    "Corps de requête JSON manquant ou invalide (Content-Type: application/json).",
                    field="body",
                    logger_instance=logger,
                )
            try:
                validated = validate_request(SetPayerRequestSchema(), body)
            except ValidationError as e:
                logger.warning(
                    "[BillingReview] set-payer validation failed booking_id=%s: %s",
                    booking_id,
                    getattr(e, "messages", e),
                )
                return handle_validation_error(e)

            from routes.companies import _get_current_company_via_use_case

            company, err, code = _get_current_company_via_use_case()
            if err or not company:
                return (err or {"error": "Entreprise non trouvée"}), (code or 404)
            company_id = int(company.id) if company.id is not None else None
            if not company_id:
                return APIErrorHandler.handle_not_found_error(
                    "Entreprise non trouvée",
                    logger_instance=logger,
                )
            user_id = _get_current_user_id()

            # Récupérer le booking (même company que clinic-monthly-totals / use case)
            booking = (
                db.session.query(Booking)
                .options(
                    joinedload(Booking.billing_party),
                    joinedload(Booking.billed_to_company),
                )
                .filter(Booking.id == booking_id, Booking.company_id == company_id)
                .first()
            )

            if not booking:
                # Debug: existe sans filtre company ? (multi-tenant / filtre sécurité)
                exists_without_filter = (
                    db.session.query(Booking)
                    .filter(Booking.id == booking_id)
                    .first()
                )
                _fmt = (
                    "[BillingReview] set-payer 404: booking_id=%s, current_company_id=%s, "
                    "booking_exists_without_company_filter=%s"
                )
                logger.warning(_fmt, booking_id, company_id, exists_without_filter is not None)
                if exists_without_filter:
                    logger.warning(
                        "[BillingReview] Booking %s appartient à company_id=%s (≠ %s)",
                        booking_id,
                        getattr(exists_without_filter, "company_id", None),
                        company_id,
                    )
                return APIErrorHandler.handle_not_found_error(
                    f"Booking {booking_id} non trouvé",
                    logger_instance=logger,
                )

            # Vérifier que le booking n'est pas verrouillé
            if booking.billing_review_status == BillingReviewStatus.LOCKED:
                return APIErrorHandler.handle_validation_error(
                    "Ce booking est verrouillé et ne peut pas être modifié",
                    field="booking_id",
                    logger_instance=logger,
                )

            # ✅ Optionnel: si clinic sans billed_to_company_id, inférer depuis booking.billed_to_company_id
            if validated["billed_to_type"] == "clinic" and not validated.get("billed_to_company_id"):
                inferred = getattr(booking, "billed_to_company_id", None)
                if inferred is not None:
                    validated["billed_to_company_id"] = inferred
                else:
                    db.session.rollback()
                    return APIErrorHandler.handle_validation_error(
                        "billed_to_company_id manquant pour facturation clinique et impossible à inférer depuis le booking.",
                        field="billed_to_company_id",
                        logger_instance=logger,
                    )

            # Snapshot avant modification (montant arrondi 0.05)
            _raw = float(booking.amount) if booking.amount is not None else 0.0
            old_amount = _round_chf_005(_raw)
            before_snapshot = {
                "billed_to_type": booking.billed_to_type,
                "billing_party_id": booking.billing_party_id,
                "billed_to_company_id": booking.billed_to_company_id,
                "billing_source": booking.billing_source.value if booking.billing_source else None,
                "billing_source_ref": booking.billing_source_ref,
                "amount": old_amount,
            }

            # Appliquer les modifications
            booking.billed_to_type = validated["billed_to_type"]
            booking.billing_party_id = validated.get("billing_party_id")
            booking.billed_to_company_id = validated.get("billed_to_company_id")
            booking.billing_override_reason = validated["reason"]
            # Réinitialiser la source car c'est une modification manuelle
            booking.billing_source = None
            booking.billing_source_ref = None
            # Marquer comme NEEDS_REVIEW si ce n'était pas déjà le cas
            if booking.billing_review_status != BillingReviewStatus.NEEDS_REVIEW:
                booking.billing_review_status = BillingReviewStatus.NEEDS_REVIEW

            # ✅ Recalculer le montant selon le nouveau billed_to_type
            DEBUG_RECALCULATION = os.environ.get("DEBUG_RECALCULATION", "0") == "1"

            new_amount = old_amount  # Par défaut, garder le montant actuel
            clinic_rate = None
            rate_source = None

            if validated["billed_to_type"] == "clinic":
                # Facturation clinique : utiliser le tarif préférentiel de la clinique
                # ✅ Règle robuste : utiliser clinic_company_id du contexte (modal S2), pas booking.billed_to_company_id
                # car booking.billed_to_company_id peut être null, incorrect, ou pointer vers une autre company
                from services.billing.client_stay_resolver import (
                    get_clinic_rate_for_booking,
                )

                clinic_company_id = validated.get("billed_to_company_id")
                if not clinic_company_id:
                    # ❌ Ne PAS utiliser booking.billed_to_company_id comme fallback (peut être null/incorrect)
                    # Le clinic_company_id DOIT venir du contexte du modal S2 (clinique sélectionnée)
                    db.session.rollback()
                    error_msg = (
                        f"Tarif clinique introuvable pour ce transport (booking_id={booking_id}): "
                        "billed_to_company_id manquant dans la requête. "
                        "Veuillez fournir l'ID de la clinique depuis le contexte du modal S2."
                    )
                    logger.error(
                        (
                            "[BillingReview] Booking %s: billed_to_company_id manquant dans la requête pour facturation clinique "
                            "(ne pas utiliser booking.billed_to_company_id comme fallback)"
                        ),
                        booking_id,
                    )
                    return APIErrorHandler.handle_validation_error(
                        error_msg,
                        field="billed_to_company_id",
                        logger_instance=logger,
                    )

                # ✅ Utiliser uniquement le clinic_company_id fourni dans la requête (contexte S2)
                if clinic_company_id:
                    # ✅ Utiliser la nouvelle fonction qui récupère le tarif depuis Company.preferential_rate
                    clinic_rate = get_clinic_rate_for_booking(
                        booking=booking,
                        clinic_company_id=clinic_company_id,
                    )

                    if clinic_rate is not None:
                        new_amount = _round_chf_005(float(clinic_rate))
                        rate_source = f"Company.preferential_rate (clinic_company_id={clinic_company_id})"
                        logger.info(
                            "[BillingReview] Booking %s: tarif clinique appliqué %.2f CHF (au lieu de %.2f CHF) - source: %s",
                            booking_id,
                            new_amount,
                            old_amount,
                            rate_source,
                        )
                        # ✅ LOG DEBUG TEMPORAIRE : Vérifier que l'API renvoie bien 40
                        if DEBUG_RECALCULATION:
                            logger.info(
                                (
                                    "[BillingReview DEBUG] Recalcul tarif clinique - "
                                    "company_id=%s, booking_id=%s, patient_id=%s, "
                                    "clinic_company_id=%s, clinic_rate=%.2f CHF, "
                                    "old_amount=%.2f CHF, new_amount=%.2f CHF, "
                                    "rate_source=%s"
                                ),
                                company_id,
                                booking_id,
                                booking.client_id,
                                clinic_company_id,
                                clinic_rate,
                                old_amount,
                                new_amount,
                                rate_source,
                            )
                    else:
                        # ✅ Comportement strict: erreur 422 si tarif introuvable
                        db.session.rollback()
                        error_msg = (
                            f"Tarif clinique introuvable pour ce transport (booking_id={booking_id}, "
                            + f"clinic_company_id={clinic_company_id}). "
                            + "Veuillez configurer le tarif préférentiel pour cette clinique."
                        )
                        logger.error(
                            "[BillingReview] Booking %s: impossible de trouver tarif clinique pour clinic_company_id=%s",
                            booking_id,
                            clinic_company_id,
                        )
                        return APIErrorHandler.handle_validation_error(
                            error_msg,
                            field="billed_to_company_id",
                            logger_instance=logger,
                        )

                # ✅ Logs DEBUG temporaires
                if DEBUG_RECALCULATION:
                    logger.debug(
                        (
                            "[BillingReview DEBUG] Recalcul montant clinic - booking_id=%s, patient_id=%s, "
                            "clinic_company_id=%s, old_amount=%.2f, clinic_rate_value=%s, "
                            "clinic_rate_source=%s, final_new_amount=%.2f"
                        ),
                        booking_id,
                        booking.client_id,
                        clinic_company_id,
                        old_amount,
                        clinic_rate,
                        rate_source,
                        new_amount,
                    )
            elif validated["billed_to_type"] == "patient":
                # Facturation patient : utiliser le tarif préférentiel du client
                if booking.client_id:
                    client = db.session.query(Client).filter(Client.id == booking.client_id).first()
                    if client and client.preferential_rate is not None:
                        new_amount = _round_chf_005(float(client.preferential_rate))
                        rate_source = f"Client.preferential_rate (client_id={booking.client_id})"
                        logger.info(
                            "[BillingReview] Booking %s: tarif patient appliqué %.2f CHF (au lieu de %.2f CHF) - source: %s",
                            booking_id,
                            new_amount,
                            old_amount,
                            rate_source,
                        )

                        # ✅ Logs DEBUG temporaires
                        if DEBUG_RECALCULATION:
                            logger.debug(
                                (
                                    "[BillingReview DEBUG] Recalcul montant patient - booking_id=%s, patient_id=%s, "
                                    "old_amount=%.2f, patient_rate_value=%.2f, rate_source=%s, final_new_amount=%.2f"
                                ),
                                booking_id,
                                booking.client_id,
                                old_amount,
                                client.preferential_rate,
                                rate_source,
                                new_amount,
                            )

            # Appliquer le nouveau montant
            booking.amount = new_amount

            # Snapshot après modification (inclut le nouveau montant)
            after_snapshot = {
                "billed_to_type": booking.billed_to_type,
                "billing_party_id": booking.billing_party_id,
                "billed_to_company_id": booking.billed_to_company_id,
                "billing_source": None,
                "billing_source_ref": None,
                "amount": new_amount,
            }

            # Créer l'entrée d'audit
            audit_log = BillingAuditLog(
                company_id=company_id,
                booking_id=booking_id,
                actor_user_id=user_id,
                action="set_payer",
                reason=validated["reason"],
                before=before_snapshot,
                after=after_snapshot,
            )
            db.session.add(audit_log)
            db.session.commit()

            logger.info(
                "[BillingReview] Booking %s: payeur modifié par user %s (reason: %s)",
                booking_id,
                user_id,
                validated["reason"],
            )

            # ✅ Retourner le booking mis à jour avec le nouveau montant et les infos de tarif
            response_data = {
                "booking_id": booking_id,
                "amount": new_amount,
                "old_amount": old_amount,
            }

            # Ajouter les infos de tarif si disponibles
            if validated["billed_to_type"] == "clinic" and clinic_rate is not None:
                response_data["clinic_rate"] = float(clinic_rate)
                response_data["rate_source"] = rate_source
            elif validated["billed_to_type"] == "patient" and rate_source:
                response_data["rate_source"] = rate_source

            # ✅ LOG DEBUG TEMPORAIRE : clinic_company_id + preferential_rate (clinic) + old/new amount
            if DEBUG_RECALCULATION:
                clinic_company_id_log = (
                    validated.get("billed_to_company_id")
                    if validated["billed_to_type"] == "clinic"
                    else None
                )
                pref_rate = float(clinic_rate) if clinic_rate is not None else None
                pref_str = f"{pref_rate:.2f}" if pref_rate is not None else "N/A"
                fmt = (
                    "[BillingReview DEBUG] set-payer response: booking_id=%s, clinic_company_id=%s, "
                    + "preferential_rate=%s, old_amount=%.2f, new_amount=%.2f, rate_source=%s"
                )
                logger.info(
                    fmt,
                    booking_id,
                    clinic_company_id_log,
                    pref_str,
                    old_amount,
                    new_amount,
                    rate_source or "",
                )

            return success_response(
                message="Payeur modifié avec succès",
                data=response_data,
            )

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "[BillingReview] Erreur lors de la modification du payeur: %s",
                e,
            )
            return APIErrorHandler.handle_exception(
                e,
                logger_instance=logger,
                default_message="Erreur lors de la modification du payeur",
            )


@billing_review_ns.route("/bookings/<int:booking_id>/lock")
class LockBooking(Resource):
    """Verrouiller un booking."""

    @jwt_required()
    @role_required(UserRole.company)
    @billing_review_ns.expect(
        billing_review_ns.model(
            "LockRequest",
            {
                "reason": fields.String(required=True),
            },
        )
    )
    def post(self, booking_id: int):
        """Verrouille un booking (admin uniquement)."""
        try:
            # Valider la requête
            try:
                validated = validate_request(LockBookingRequestSchema(), request.json)
            except ValidationError as e:
                return handle_validation_error(e)

            from routes.companies import _get_current_company_via_use_case

            company, err, code = _get_current_company_via_use_case()
            if err or not company:
                return (err or {"error": "Entreprise non trouvée"}), (code or 404)
            company_id = int(company.id) if company.id is not None else None
            if not company_id:
                return APIErrorHandler.handle_not_found_error(
                    "Entreprise non trouvée",
                    logger_instance=logger,
                )
            user_id = _get_current_user_id()

            # Récupérer le booking (même company que clinic-monthly-totals / use case)
            booking = (
                db.session.query(Booking)
                .filter(Booking.id == booking_id, Booking.company_id == company_id)
                .first()
            )

            if not booking:
                return APIErrorHandler.handle_not_found_error(
                    f"Booking {booking_id} non trouvé",
                    logger_instance=logger,
                )

            # Vérifier que le booking n'est pas déjà verrouillé
            if booking.billing_review_status == BillingReviewStatus.LOCKED:
                return APIErrorHandler.handle_validation_error(
                    "Ce booking est déjà verrouillé",
                    field="booking_id",
                    logger_instance=logger,
                )

            # Snapshot avant modification
            before_snapshot = {
                "billing_review_status": booking.billing_review_status.value,
                "billing_locked_at": None,
                "billing_locked_by_user_id": None,
            }

            # Verrouiller
            booking.billing_review_status = BillingReviewStatus.LOCKED
            booking.billing_locked_at = datetime.utcnow()
            booking.billing_locked_by_user_id = user_id

            # Snapshot après modification
            after_snapshot = {
                "billing_review_status": booking.billing_review_status.value,
                "billing_locked_at": booking.billing_locked_at.isoformat(),
                "billing_locked_by_user_id": user_id,
            }

            # Créer l'entrée d'audit
            audit_log = BillingAuditLog(
                company_id=company_id,
                booking_id=booking_id,
                actor_user_id=user_id,
                action="lock",
                reason=validated["reason"],
                before=before_snapshot,
                after=after_snapshot,
            )
            db.session.add(audit_log)
            db.session.commit()

            logger.info(
                "[BillingReview] Booking %s: verrouillé par user %s (reason: %s)",
                booking_id,
                user_id,
                validated["reason"],
            )

            return success_response(
                message="Booking verrouillé avec succès",
                data={"booking_id": booking_id},
            )

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "[BillingReview] Erreur lors du verrouillage: %s",
                e,
            )
            return APIErrorHandler.handle_exception(
                e,
                logger_instance=logger,
                default_message="Erreur lors du verrouillage",
            )


@billing_review_ns.route("/bookings/<int:booking_id>/unlock")
class UnlockBooking(Resource):
    """Déverrouiller un booking."""

    @jwt_required()
    @role_required(UserRole.admin)  # Seuls les admins peuvent déverrouiller
    @billing_review_ns.expect(
        billing_review_ns.model(
            "UnlockRequest",
            {
                "reason": fields.String(required=True),
            },
        )
    )
    def post(self, booking_id: int):
        """Déverrouille un booking (admin uniquement)."""
        try:
            # Valider la requête
            try:
                validated = validate_request(
                    UnlockBookingRequestSchema(), request.json
                )
            except ValidationError as e:
                return handle_validation_error(e)

            from routes.companies import _get_current_company_via_use_case

            company, err, code = _get_current_company_via_use_case()
            if err or not company:
                return (err or {"error": "Entreprise non trouvée"}), (code or 404)
            company_id = int(company.id) if company.id is not None else None
            if not company_id:
                return APIErrorHandler.handle_not_found_error(
                    "Entreprise non trouvée",
                    logger_instance=logger,
                )
            user_id = _get_current_user_id()

            # Récupérer le booking (même company que clinic-monthly-totals / use case)
            booking = (
                db.session.query(Booking)
                .filter(Booking.id == booking_id, Booking.company_id == company_id)
                .first()
            )

            if not booking:
                return APIErrorHandler.handle_not_found_error(
                    f"Booking {booking_id} non trouvé",
                    logger_instance=logger,
                )

            # Vérifier que le booking est verrouillé
            if booking.billing_review_status != BillingReviewStatus.LOCKED:
                return APIErrorHandler.handle_validation_error(
                    "Ce booking n'est pas verrouillé",
                    field="booking_id",
                    logger_instance=logger,
                )

            # Snapshot avant modification
            before_snapshot = {
                "billing_review_status": booking.billing_review_status.value,
                "billing_locked_at": (
                    booking.billing_locked_at.isoformat()
                    if booking.billing_locked_at
                    else None
                ),
                "billing_locked_by_user_id": booking.billing_locked_by_user_id,
            }

            # Déverrouiller (retour à READY si pas de conflit, sinon NEEDS_REVIEW)
            if booking.billing_override_reason:
                booking.billing_review_status = BillingReviewStatus.NEEDS_REVIEW
            else:
                booking.billing_review_status = BillingReviewStatus.READY
            booking.billing_locked_at = None
            booking.billing_locked_by_user_id = None

            # Snapshot après modification
            after_snapshot = {
                "billing_review_status": booking.billing_review_status.value,
                "billing_locked_at": None,
                "billing_locked_by_user_id": None,
            }

            # Créer l'entrée d'audit
            audit_log = BillingAuditLog(
                company_id=company_id,
                booking_id=booking_id,
                actor_user_id=user_id,
                action="unlock",
                reason=validated["reason"],
                before=before_snapshot,
                after=after_snapshot,
            )
            db.session.add(audit_log)
            db.session.commit()

            logger.info(
                "[BillingReview] Booking %s: déverrouillé par admin user %s (reason: %s)",
                booking_id,
                user_id,
                validated["reason"],
            )

            return success_response(
                message="Booking déverrouillé avec succès",
                data={"booking_id": booking_id},
            )

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "[BillingReview] Erreur lors du déverrouillage: %s",
                e,
            )
            return APIErrorHandler.handle_internal_error(
                "Erreur lors du déverrouillage",
                logger_instance=logger,
            )


@billing_review_ns.route("/bookings/batch-set-payer")
class BatchSetBookingPayer(Resource):
    """Modifier le payeur de plusieurs bookings en batch."""

    @jwt_required()
    @role_required(UserRole.company)
    @billing_review_ns.expect(
        billing_review_ns.model(
            "BatchSetPayerRequest",
            {
                "booking_ids": fields.List(fields.Integer, required=True),
                "billed_to_type": fields.String(required=True),
                "billing_party_id": fields.Integer(allow_none=True),
                "billed_to_company_id": fields.Integer(allow_none=True),
                "reason": fields.String(required=True),
            },
        )
    )
    def post(self):
        """Modifie le payeur de plusieurs bookings en batch avec audit logging."""
        try:
            # Valider la requête
            try:
                validated = validate_request(
                    BatchSetPayerRequestSchema(), request.json
                )
            except ValidationError as e:
                return handle_validation_error(e)

            from routes.companies import _get_current_company_via_use_case

            company, err, code = _get_current_company_via_use_case()
            if err or not company:
                return (err or {"error": "Entreprise non trouvée"}), (code or 404)
            company_id = int(company.id) if company.id is not None else None
            if not company_id:
                return APIErrorHandler.handle_not_found_error(
                    "Entreprise non trouvée",
                    logger_instance=logger,
                )
            user_id = _get_current_user_id()
            booking_ids = validated["booking_ids"]

            # Limiter le nombre de bookings pour éviter les timeouts
            MAX_BATCH_SIZE = 100
            if len(booking_ids) > MAX_BATCH_SIZE:
                return APIErrorHandler.handle_validation_error(
                    f"Maximum {MAX_BATCH_SIZE} bookings autorisés par opération batch",
                    field="booking_ids",
                    logger_instance=logger,
                )

            # Récupérer les bookings
            bookings = (
                db.session.query(Booking)
                .options(
                    joinedload(Booking.billing_party),
                    joinedload(Booking.billed_to_company),
                )
                .filter(
                    Booking.id.in_(booking_ids),
                    Booking.company_id == company_id,
                )
                .all()
            )

            if not bookings:
                return APIErrorHandler.handle_not_found_error(
                    "Aucun booking trouvé pour les IDs fournis",
                    logger_instance=logger,
                )

            # Vérifier que tous les bookings ne sont pas verrouillés
            locked_bookings = [
                b.id for b in bookings if b.billing_review_status == BillingReviewStatus.LOCKED
            ]
            if locked_bookings:
                return APIErrorHandler.handle_validation_error(
                    f"Certains bookings sont verrouillés et ne peuvent pas être modifiés: {locked_bookings}",
                    field="booking_ids",
                    logger_instance=logger,
                )

            # Appliquer les modifications à tous les bookings
            updated_count = 0
            audit_logs = []

            for booking in bookings:
                # Snapshot avant modification
                before_snapshot = {
                    "billed_to_type": booking.billed_to_type,
                    "billing_party_id": booking.billing_party_id,
                    "billed_to_company_id": booking.billed_to_company_id,
                    "billing_source": (
                        booking.billing_source.value if booking.billing_source else None
                    ),
                    "billing_source_ref": booking.billing_source_ref,
                }

                # Appliquer les modifications
                booking.billed_to_type = validated["billed_to_type"]
                booking.billing_party_id = validated.get("billing_party_id")
                booking.billed_to_company_id = validated.get("billed_to_company_id")
                booking.billing_override_reason = validated["reason"]
                # Réinitialiser la source car c'est une modification manuelle
                booking.billing_source = None
                booking.billing_source_ref = None
                # Marquer comme NEEDS_REVIEW si ce n'était pas déjà le cas
                if booking.billing_review_status != BillingReviewStatus.NEEDS_REVIEW:
                    booking.billing_review_status = BillingReviewStatus.NEEDS_REVIEW

                # Snapshot après modification
                after_snapshot = {
                    "billed_to_type": booking.billed_to_type,
                    "billing_party_id": booking.billing_party_id,
                    "billed_to_company_id": booking.billed_to_company_id,
                    "billing_source": None,
                    "billing_source_ref": None,
                }

                # Créer l'entrée d'audit
                audit_log = BillingAuditLog(
                    company_id=company_id,
                    booking_id=booking.id,
                    actor_user_id=user_id,
                    action="batch_set_payer",
                    reason=validated["reason"],
                    before=before_snapshot,
                    after=after_snapshot,
                )
                audit_logs.append(audit_log)
                updated_count += 1

            # Ajouter tous les logs d'audit
            db.session.add_all(audit_logs)
            db.session.commit()

            logger.info(
                "[BillingReview] Batch: %d bookings modifiés par user %s (reason: %s)",
                updated_count,
                user_id,
                validated["reason"],
            )

            return success_response(
                message=f"Payeur modifié avec succès pour {updated_count} booking(s)",
                data={
                    "updated_count": updated_count,
                    "booking_ids": [b.id for b in bookings],
                },
            )

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "[BillingReview] Erreur lors de la modification batch du payeur: %s",
                e,
            )
            return APIErrorHandler.handle_exception(
                e,
                logger_instance=logger,
                default_message="Erreur lors de la modification batch du payeur",
            )
