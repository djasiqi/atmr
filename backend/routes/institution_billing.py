# routes/institution_billing.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Routes pour la facturation côté institution.

ÉTAPE 5: Permissions de modification des infos de facturation.

Endpoints:
- PUT /api/v1/institutions/requests/{id}/billing - Modifier facturation d'une request
- PUT /api/v1/institutions/bookings/{booking_id}/billing - Modifier facturation d'un booking
"""

import logging
from typing import Any, cast

import sentry_sdk
from flask import request
from flask_jwt_extended import get_jwt, get_jwt_identity, verify_jwt_in_request
from flask_restx import Namespace, Resource, fields
from marshmallow import Schema, validate
from marshmallow import fields as ma_fields

from ext import db
from models import Booking, RequestStatus, TransportRequest
from models.enums import BillingIntent, InstitutionRole
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from security.audit_log import AuditLogger

# Constante pour message d'erreur
ROLE_REQUIRED_MSG = "Rôle requis: %s. Votre rôle: %s"

logger = logging.getLogger(__name__)

# Namespace
institution_billing_ns = Namespace(
    "institution_billing",
    description="Gestion de la facturation côté institution",
)

# Modèles erreurs
api_error_model = create_api_error_model(institution_billing_ns)
not_found_error_model = create_not_found_error_model(institution_billing_ns)
permission_error_model = create_permission_error_model(institution_billing_ns)
validation_error_model = create_validation_error_model(institution_billing_ns)

# Modèles Swagger
billing_update_model = institution_billing_ns.model(
    "BillingUpdate",
    {
        "billing_intent": fields.String(
            description="Intention de facturation",
            enum=BillingIntent.choices(),
        ),
        "billing_details": fields.Raw(
            description="Détails de facturation (JSON)",
        ),
        "override_reason": fields.String(
            required=True,
            description="Motif obligatoire de modification facturation",
        ),
        "billing_change_reason_code": fields.String(
            required=True,
            description="Code motif (liste fermée)",
        ),
        "reason_comment": fields.String(
            description="Commentaire optionnel",
        ),
        "version": fields.Integer(
            description="Version optimiste du booking (si CONVERTED)",
        ),
    },
)

billing_result_model = institution_billing_ns.model(
    "BillingUpdateResult",
    {
        "success": fields.Boolean(description="Succès de l'opération"),
        "billing_intent": fields.String(description="Nouvelle intention"),
        "billing_details": fields.Raw(description="Nouveaux détails"),
        "updated_at": fields.String(description="Date de mise à jour"),
    },
)


# Schéma validation
class BillingUpdateSchema(Schema):
    """Schéma pour mise à jour facturation."""

    billing_intent = ma_fields.String(
        required=False,
        validate=validate.OneOf(BillingIntent.choices()),
    )
    billing_details = ma_fields.Dict(
        required=False,
        allow_none=True,
    )
    override_reason = ma_fields.String(required=True, validate=validate.Length(min=3))
    billing_change_reason_code = ma_fields.String(required=True)
    reason_comment = ma_fields.String(required=False, allow_none=True)
    version = ma_fields.Integer(required=False)


billing_update_schema = BillingUpdateSchema()

# Rôles autorisés pour modifier la facturation
BILLING_ALLOWED_ROLES = {
    InstitutionRole.ADMIN.value,
    InstitutionRole.BILLING.value,
}


def get_billing_context() -> tuple[int, int | None, str | None]:
    """Récupère le contexte institution avec vérification rôle billing.

    Returns:
        Tuple (institution_id, user_id, institution_role)

    Raises:
        Werkzeug Abort si non authentifié, pas institution, ou pas billing/admin
    """
    from flask import abort

    verify_jwt_in_request()

    claims = get_jwt()
    institution_id = claims.get("institution_id")
    institution_role = claims.get("institution_role")

    if not institution_id:
        abort(403, description="Accès réservé aux utilisateurs institution")

    # Vérifier que l'utilisateur a le rôle billing ou admin
    if institution_role not in BILLING_ALLOWED_ROLES:
        msg = ROLE_REQUIRED_MSG % (", ".join(BILLING_ALLOWED_ROLES), institution_role)
        abort(403, description=msg)

    raw_identity = get_jwt_identity()
    user_id: int | None = None
    if raw_identity is not None:
        raw = str(raw_identity).strip()
        if raw:
            if raw.isdigit():
                user_id = int(raw)
            else:
                try:
                    from models import User

                    u = User.query.filter_by(public_id=raw).first()
                    if u:
                        user_id = int(u.id)
                except Exception:
                    user_id = None
    return institution_id, user_id, institution_role


@institution_billing_ns.route("/requests/<int:request_id>")
@institution_billing_ns.param("request_id", "ID de la demande de transport")
class RequestBillingUpdate(Resource):
    """Mise à jour facturation d'une demande de transport."""

    @institution_billing_ns.doc(
        description="Modifie les informations de facturation d'une demande",
        security="BearerAuth",
    )
    @institution_billing_ns.expect(billing_update_model)
    @institution_billing_ns.response(200, "Succès", billing_result_model)
    @institution_billing_ns.response(400, "Données invalides", validation_error_model)
    @institution_billing_ns.response(401, "Non authentifié", permission_error_model)
    @institution_billing_ns.response(403, "Accès refusé", permission_error_model)
    @institution_billing_ns.response(404, "Demande non trouvée", not_found_error_model)
    @institution_billing_ns.response(409, "Demande convertie", api_error_model)
    def put(self, request_id: int):
        """Modifie les informations de facturation d'une demande.

        Auth: JWT institution_role BILLING ou ADMIN requis

        Règles ÉTAPE 5:
        - Modifiable uniquement si status != CONVERTED
        - Si CONVERTED, renvoyer 409 (modifier via booking)
        """
        try:
            institution_id, user_id, role = get_billing_context()

            # Charger la demande
            transport_req = TransportRequest.query.filter_by(
                id=request_id,
                institution_id=institution_id,
            ).first()

            if not transport_req:
                return {"error": "Demande non trouvée"}, 404

            # Vérifier si convertie
            if transport_req.status == RequestStatus.CONVERTED.value:
                return {
                    "error": "Demande convertie en booking. Modifiez la facturation via le booking.",
                    "resulting_booking_id": transport_req.booking_id,
                }, 409

            # Valider les données
            data = request.get_json() or {}
            errors = billing_update_schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = cast(dict[str, Any], billing_update_schema.load(data))

            from services.institutions.booking_change_service import (
                BILLING_CHANGE_REASON_CODES,
            )

            code = (validated.get("billing_change_reason_code") or "").upper()
            if code not in BILLING_CHANGE_REASON_CODES:
                return {
                    "error": "billing_change_reason_code invalide",
                    "allowed": sorted(BILLING_CHANGE_REASON_CODES),
                }, 400

            # Garder anciennes valeurs pour audit
            old_intent = transport_req.billing_intent
            old_details = transport_req.billing_details
            before = {
                "billing_intent": old_intent,
                "billing_details": old_details,
            }

            # Mettre à jour
            if validated.get("billing_intent"):
                transport_req.billing_intent = validated["billing_intent"]

            if "billing_details" in validated:
                transport_req.billing_details = validated["billing_details"]

            after = {
                "billing_intent": transport_req.billing_intent,
                "billing_details": transport_req.billing_details,
            }

            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="request_billing_updated",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution",
                    institution_id=institution_id,
                    result_status="success",
                    action_details={
                        "request_id": transport_req.id,
                        "old_billing_intent": old_intent,
                        "new_billing_intent": transport_req.billing_intent,
                        "old_billing_details": old_details,
                        "new_billing_details": transport_req.billing_details,
                        "role": role,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[InstitutionBilling] Audit log error: %s", audit_err)

            logger.info(
                "[InstitutionBilling] Request %s billing updated by %s (role=%s)",
                request_id,
                user_id,
                role,
            )

            return {
                "success": True,
                "billing_intent": transport_req.billing_intent,
                "billing_details": transport_req.billing_details,
                "updated_at": transport_req.updated_at.isoformat()
                if transport_req.updated_at
                else None,
            }

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[InstitutionBilling] PUT request/%s error: %s", request_id, e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_billing_ns.route("/bookings/<int:booking_id>")
@institution_billing_ns.param("booking_id", "ID du booking")
class BookingBillingUpdate(Resource):
    """Mise à jour facturation d'un booking issu d'une demande institution."""

    @institution_billing_ns.doc(
        description="Modifie les informations de facturation d'un booking",
        security="BearerAuth",
    )
    @institution_billing_ns.expect(billing_update_model)
    @institution_billing_ns.response(200, "Succès", billing_result_model)
    @institution_billing_ns.response(400, "Données invalides", validation_error_model)
    @institution_billing_ns.response(401, "Non authentifié", permission_error_model)
    @institution_billing_ns.response(403, "Accès refusé", permission_error_model)
    @institution_billing_ns.response(404, "Booking non trouvé", not_found_error_model)
    @institution_billing_ns.response(409, "Booking déjà facturé", api_error_model)
    def put(self, booking_id: int):
        """Modifie les informations de facturation d'un booking.

        Auth: JWT institution_role BILLING ou ADMIN requis

        Règles ÉTAPE 5:
        - Le booking doit être issu d'une TransportRequest de l'institution
        - Non modifiable si déjà facturé (invoice_line existante)
        """
        try:
            institution_id, user_id, role = get_billing_context()

            # Trouver le booking via la TransportRequest
            # Le booking doit avoir une source_request liée à cette institution
            transport_req = TransportRequest.query.filter_by(
                booking_id=booking_id,
                institution_id=institution_id,
            ).first()

            if not transport_req:
                return {
                    "error": "Booking non trouvé ou non associé à votre institution",
                }, 404

            booking = Booking.query.get(booking_id)
            if not booking:
                return {"error": "Booking non trouvé"}, 404

            # Vérifier si déjà facturé (via InvoiceLine ou invoice générée)
            # Le champ dans InvoiceLine s'appelle reservation_id (référence booking.id)
            # ⚠️ Exclure les InvoiceLine liées à des factures annulées (CANCELLED)
            from models import Invoice, InvoiceLine
            from models.enums import InvoiceStatus

            existing_line = (
                db.session.query(InvoiceLine)
                .join(Invoice, InvoiceLine.invoice_id == Invoice.id)
                .filter(
                    InvoiceLine.reservation_id == booking_id,
                    Invoice.status != InvoiceStatus.CANCELLED,
                )
                .first()
            )

            if existing_line:
                return {
                    "error": "Booking déjà facturé. Modification de facturation interdite.",
                    "invoice_id": existing_line.invoice_id,
                }, 409

            # Valider les données
            data = request.get_json() or {}
            errors = billing_update_schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = cast(dict[str, Any], billing_update_schema.load(data))

            from services.institutions.booking_change_service import (
                BILLING_CHANGE_REASON_CODES,
                bump_edit_version,
                check_version,
                record_change_event,
                _billing_snapshot,
            )

            code = (validated.get("billing_change_reason_code") or "").upper()
            if code not in BILLING_CHANGE_REASON_CODES:
                return {
                    "error": "billing_change_reason_code invalide",
                    "allowed": sorted(BILLING_CHANGE_REASON_CODES),
                }, 400

            client_version = validated.get("version")
            if client_version is not None:
                conflict = check_version(booking, client_version)
                if conflict:
                    return conflict, 409

            # Garder anciennes valeurs pour audit
            old_billed_to = booking.billed_to_type
            before = _billing_snapshot(booking)

            # Mettre à jour le booking
            # Mapping billing_intent → billed_to_type (institution → clinic)
            intent_to_billed = {
                "institution": "clinic",
                "patient": "patient",
                "insurance": "insurance",
                "curator": "clinic",
                "spc": "clinic",
                "other": "clinic",
            }
            new_intent = validated.get("billing_intent")
            if new_intent:
                new_billed_to = intent_to_billed.get(new_intent, "patient")
                booking.billed_to_type = new_billed_to
                # billed_to_company_id obligatoire si non-patient
                if new_billed_to != "patient":
                    booking.billed_to_company_id = booking.company_id
                else:
                    booking.billed_to_company_id = None

            booking.billing_override_reason = validated.get("override_reason")
            bump_edit_version(booking)
            after = _billing_snapshot(booking)
            financial_role = (
                "billing" if role == InstitutionRole.BILLING.value else "admin"
            )
            record_change_event(
                booking=booking,
                transport_request=transport_req,
                institution_id=institution_id,
                actor_user_id=user_id,
                actor_role=role,
                actor_type="institution_user",
                actor_display_name=None,
                action_type="billing_changed",
                change_scope="billing",
                source="institution_portal",
                before_snapshot=before,
                after_snapshot=after,
                reason=validated.get("override_reason"),
                change_class="major",
                severity="WARNING",
                financial_actor_role=financial_role,
                billing_change_reason_code=code,
            )

            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="booking_billing_updated",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution",
                    institution_id=institution_id,
                    booking_id=booking_id,
                    result_status="success",
                    action_details={
                        "booking_id": booking_id,
                        "transport_request_id": transport_req.id,
                        "old_billed_to_type": old_billed_to,
                        "new_billed_to_type": booking.billed_to_type,
                        "billing_details": validated.get("billing_details"),
                        "role": role,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[InstitutionBilling] Audit log error: %s", audit_err)

            logger.info(
                "[InstitutionBilling] Booking %s billing updated by institution %s",
                booking_id,
                institution_id,
            )

            return {
                "success": True,
                "billing_intent": booking.billed_to_type,
                "billed_to_type": booking.billed_to_type,
                "updated_at": None,
            }

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[InstitutionBilling] PUT booking/%s error: %s", booking_id, e)
            return {"error": f"Erreur serveur: {e!s}"}, 500
