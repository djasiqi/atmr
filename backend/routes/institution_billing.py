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
from flask_jwt_extended import jwt_required
from flask_jwt_extended.exceptions import JWTExtendedException
from flask_restx import Namespace, Resource, fields
from jwt.exceptions import PyJWTError
from marshmallow import Schema, validate
from marshmallow import fields as ma_fields

from ext import db
from models import RequestStatus, TransportRequest
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


def _int_arg(name: str) -> int | None:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _reraise_auth_errors(exc: Exception) -> None:
    """Ne pas transformer les erreurs JWT/auth en 500 ni les remonter à Sentry."""
    if isinstance(exc, (JWTExtendedException, PyJWTError)):
        raise exc
    if hasattr(exc, "code"):
        raise exc
    lowered = str(exc).lower()
    if "signature has expired" in lowered or "token has expired" in lowered:
        raise exc


# Rôles autorisés pour modifier la facturation
BILLING_ALLOWED_ROLES = {
    InstitutionRole.ADMIN.value,
    InstitutionRole.BILLING.value,
}


def get_billing_context() -> tuple[int, int | None, str | None]:
    """Délègue à l'ACL canonique contrôle facturation institution."""
    from security.institution_billing_control_acl import (
        require_institution_billing_control_context,
    )

    return require_institution_billing_control_context()


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
    @jwt_required()
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

            # Mettre à jour
            if validated.get("billing_intent"):
                transport_req.billing_intent = validated["billing_intent"]

            if "billing_details" in validated:
                transport_req.billing_details = validated["billing_details"]

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
            _reraise_auth_errors(e)
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
    @jwt_required()
    def put(self, booking_id: int):
        """Modifie les informations de facturation d'un booking.

        Auth: JWT institution_role BILLING ou ADMIN requis

        Règles ÉTAPE 5:
        - Le booking doit être issu d'une TransportRequest de l'institution
        - Non modifiable si déjà facturé (invoice_line existante)
        """
        try:
            institution_id, user_id, role = get_billing_context()

            from application.institutions.billing_control.resolve import (
                resolve_institution_billing_control_booking,
            )

            ctx = resolve_institution_billing_control_booking(
                booking_id,
                institution_id,
            )

            if ctx is None:
                return {
                    "error": "Booking non trouvé ou non associé à votre institution",
                }, 404

            booking = ctx.booking
            transport_req = ctx.transport_request

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

            from application.institutions.change_booking_payer import (
                change_booking_payer,
            )
            from services.institutions.booking_change_service import (
                BILLING_CHANGE_REASON_CODES,
                check_version,
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

            new_intent = validated.get("billing_intent")
            if not new_intent:
                return {"error": "billing_intent requis pour modifier le payeur."}, 400

            financial_role = (
                "billing" if role == InstitutionRole.BILLING.value else "admin"
            )
            payer_result = change_booking_payer(
                booking,
                target_payer=new_intent,
                transport_request=transport_req,
                institution_id=institution_id,
                actor_user_id=user_id,
                actor_role=role,
                actor_display_name=None,
                override_reason=validated.get("override_reason") or "",
                billing_change_reason_code=code,
                financial_actor_role=financial_role,
            )
            if not payer_result.ok:
                status = int(payer_result.status_code or 422)
                return {
                    "error": payer_result.error or "Modification payeur refusée."
                }, status

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
                        "old_billed_to_type": (payer_result.before or {}).get(
                            "billed_to_type"
                        ),
                        "new_billed_to_type": booking.billed_to_type,
                        "old_billing_party_id": (payer_result.before or {}).get(
                            "billing_party_id"
                        ),
                        "new_billing_party_id": booking.billing_party_id,
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
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            logger.error("[InstitutionBilling] PUT booking/%s error: %s", booking_id, e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


# ── Contrôle facturation institution (INSTITUTION-07) ─────────────────────


control_list_model = institution_billing_ns.model(
    "BillingControlList",
    {
        "items": fields.List(fields.Raw),
        "count": fields.Integer(description="Nombre d'éléments page courante"),
        "pagination": fields.Raw,
        "summary": fields.Raw,
    },
)

control_detail_model = institution_billing_ns.model(
    "BillingControlDetail",
    {
        "booking_id": fields.Integer,
        "control": fields.Raw,
        "payer": fields.Raw,
        "billing": fields.Raw,
        "relationship": fields.Raw,
    },
)

anomaly_model = institution_billing_ns.model(
    "BillingControlAnomaly",
    {
        "anomaly_reason_code": fields.String(required=True),
        "comment": fields.String,
    },
)


@institution_billing_ns.route("/control/bookings")
class BillingControlList(Resource):
    """Liste des bookings en contrôle facturation."""

    @institution_billing_ns.doc(
        description="Liste contrôle facturation (Admin/Billing)",
        security="BearerAuth",
    )
    @institution_billing_ns.response(200, "Succès", control_list_model)
    @institution_billing_ns.response(403, "Accès refusé", permission_error_model)
    @jwt_required()
    def get(self):
        try:
            institution_id, _user_id, _role = get_billing_context()
            from application.institutions.billing_control.query import (
                parse_billing_control_query,
                query_billing_control_bookings,
            )

            parsed = parse_billing_control_query(
                period=request.args.get("period"),
                period_year=_int_arg("period_year"),
                period_month=_int_arg("period_month"),
                control_status=request.args.get("control_status"),
                payer_type=request.args.get("payer_type"),
                transport_company=_int_arg("transport_company"),
                patient=_int_arg("patient"),
                page=_int_arg("page"),
                page_size=_int_arg("page_size"),
            )
            if isinstance(parsed, tuple):
                return {"error": parsed[0]}, parsed[1]

            result = query_billing_control_bookings(institution_id, parsed)
            return {
                "items": result.items,
                "count": len(result.items),
                "pagination": {
                    "page": result.page,
                    "page_size": result.page_size,
                    "total": result.total,
                    "total_pages": result.total_pages,
                },
                "summary": result.summary.to_dict(),
            }
        except Exception as e:
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_billing_ns.route("/control/bookings/<int:booking_id>")
@institution_billing_ns.param("booking_id", "ID du booking")
class BillingControlDetail(Resource):
    """Détail contrôle facturation d'un booking."""

    @jwt_required()
    def get(self, booking_id: int):
        try:
            institution_id, _user_id, _role = get_billing_context()
            from application.institutions.billing_control.list_bookings import (
                booking_control_detail,
            )
            from application.institutions.billing_control.resolve import (
                resolve_institution_billing_control_booking,
            )

            ctx = resolve_institution_billing_control_booking(
                booking_id, institution_id
            )
            if ctx is None:
                return {"error": "Booking non trouvé"}, 404
            return booking_control_detail(ctx.booking, institution_id=institution_id)
        except Exception as e:
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_billing_ns.route("/control/bookings/<int:booking_id>/validate")
class BillingControlValidate(Resource):
    @jwt_required()
    def post(self, booking_id: int):
        try:
            institution_id, user_id, role = get_billing_context()
            from application.institutions.billing_control.mutations import (
                validate_booking_control,
            )
            from application.institutions.billing_control.resolve import (
                resolve_institution_billing_control_booking,
            )

            ctx = resolve_institution_billing_control_booking(
                booking_id, institution_id
            )
            if ctx is None:
                return {"error": "Booking non trouvé"}, 404

            body = request.get_json(silent=True) or {}
            display_name = body.get("actor_display_name")
            result = validate_booking_control(
                ctx.booking,
                transport_request=ctx.transport_request,
                institution_id=institution_id,
                actor_user_id=user_id,
                actor_role=role,
                actor_display_name=display_name,
            )
            if not result.ok:
                return {"error": result.error}, int(result.status_code or 422)
            db.session.commit()
            return {"success": True, "control": result.after}
        except Exception as e:
            db.session.rollback()
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_billing_ns.route("/control/bookings/<int:booking_id>/anomaly")
class BillingControlAnomaly(Resource):
    @jwt_required()
    def post(self, booking_id: int):
        try:
            institution_id, user_id, role = get_billing_context()
            from application.institutions.billing_control.mutations import (
                mark_booking_control_anomaly,
            )
            from application.institutions.billing_control.resolve import (
                resolve_institution_billing_control_booking,
            )

            ctx = resolve_institution_billing_control_booking(
                booking_id, institution_id
            )
            if ctx is None:
                return {"error": "Booking non trouvé"}, 404

            body = request.get_json() or {}
            code = body.get("anomaly_reason_code") or ""
            result = mark_booking_control_anomaly(
                ctx.booking,
                transport_request=ctx.transport_request,
                institution_id=institution_id,
                actor_user_id=user_id,
                actor_role=role,
                actor_display_name=body.get("actor_display_name"),
                anomaly_reason_code=code,
                anomaly_comment=body.get("comment"),
            )
            if not result.ok:
                return {"error": result.error}, int(result.status_code or 422)
            db.session.commit()
            return {"success": True, "control": result.after}
        except Exception as e:
            db.session.rollback()
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_billing_ns.route("/control/bookings/<int:booking_id>/reopen")
class BillingControlReopen(Resource):
    @jwt_required()
    def post(self, booking_id: int):
        try:
            institution_id, user_id, role = get_billing_context()
            from application.institutions.billing_control.mutations import (
                reopen_booking_control,
            )
            from application.institutions.billing_control.resolve import (
                resolve_institution_billing_control_booking,
            )

            ctx = resolve_institution_billing_control_booking(
                booking_id, institution_id
            )
            if ctx is None:
                return {"error": "Booking non trouvé"}, 404

            body = request.get_json(silent=True) or {}
            result = reopen_booking_control(
                ctx.booking,
                transport_request=ctx.transport_request,
                institution_id=institution_id,
                actor_user_id=user_id,
                actor_role=role,
                actor_display_name=body.get("actor_display_name"),
                reason=body.get("reason"),
            )
            if not result.ok:
                return {"error": result.error}, int(result.status_code or 422)
            db.session.commit()
            return {"success": True, "control": result.after}
        except Exception as e:
            db.session.rollback()
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            return {"error": f"Erreur serveur: {e!s}"}, 500
