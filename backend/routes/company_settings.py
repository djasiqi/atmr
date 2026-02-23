"""Routes API pour les paramètres avancés de l'entreprise."""

import logging
from http import HTTPStatus

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import (
    Namespace,
    Resource,
    fields,
)

from ext import db, role_required
from models import (
    BillingParty,
    BillingPartyType,
    Client,
    ClinicBillingPartyMapping,
    Company,
    CompanyBillingSettings,
    CompanyPlanningSettings,
    User,
    UserRole,
)
from routes.companies import get_company_from_token
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

settings_ns = Namespace("company-settings", description="Paramètres avancés entreprise")

# ==================== Models API ====================

billing_settings_model = settings_ns.model(
    "BillingSettings",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "payment_terms_days": fields.Integer(description="Délai de paiement en jours"),
        "overdue_fee": fields.Float(description="Frais de retard"),
        "reminder1_fee": fields.Float(description="Frais 1er rappel"),
        "reminder2_fee": fields.Float(description="Frais 2e rappel"),
        "reminder3_fee": fields.Float(description="Frais 3e rappel"),
        "material_delivery_price_fixed": fields.Float(
            description="Prix fixe livraison matériel (CHF)", allow_null=True
        ),
        "reminder_schedule_days": fields.Raw(description="Planning des rappels"),
        "auto_reminders_enabled": fields.Boolean(
            description="Rappels automatiques activés"
        ),
        "email_sender": fields.String(description="Email expéditeur"),
        "invoice_number_format": fields.String(description="Format de numérotation"),
        "invoice_prefix": fields.String(description="Préfixe des factures"),
        "iban": fields.String(description="IBAN"),
        "qr_iban": fields.String(description="QR-IBAN"),
        "esr_ref_base": fields.String(description="Référence ESR"),
        "invoice_message_template": fields.String(description="Template email facture"),
        "reminder1_template": fields.String(description="Template 1er rappel"),
        "reminder2_template": fields.String(description="Template 2e rappel"),
        "reminder3_template": fields.String(description="Template 3e rappel"),
        "email_signature_mode": fields.String(
            description="Mode signature email: 'form', 'text' ou 'html'", default="form"
        ),
        "email_signature_text": fields.String(
            description="Signature email (mode texte)"
        ),
        "signature_name": fields.String(description="Nom complet (mode form)"),
        "signature_title": fields.String(
            description="Titre (mode form, ex: 'Associé gérant')"
        ),
        "signature_company": fields.String(description="Société (mode form)"),
        "signature_phone_main": fields.String(
            description="Téléphone principal (mode form)"
        ),
        "signature_phone_mobile": fields.String(
            description="Téléphone mobile (mode form)"
        ),
        "signature_email": fields.String(description="Email (mode form)"),
        "signature_website": fields.String(description="Site web (mode form)"),
        "signature_address_line": fields.String(
            description="Ligne adresse (mode form)"
        ),
        "signature_zip": fields.String(description="Code postal (mode form)"),
        "signature_city": fields.String(description="Ville (mode form)"),
        "email_signature_html_template": fields.String(
            description="Template HTML signature (mode HTML, variables: name, phone, email, address, logo_url)"
        ),
        "legal_footer": fields.String(description="Pied de page légal"),
        "pdf_template_variant": fields.String(description="Variante template PDF"),
        # TVA
        "vat_applicable": fields.Boolean(description="TVA applicable", allow_null=True),
        "vat_rate": fields.Float(description="Taux de TVA (%)", allow_null=True),
        "vat_label": fields.String(description="Libellé TVA", allow_null=True),
        "vat_number": fields.String(description="Numéro de TVA", allow_null=True),
        # SMTP
        "smtp_enabled": fields.Boolean(description="SMTP activé"),
        "smtp_server": fields.String(description="Serveur SMTP"),
        "smtp_port": fields.Integer(description="Port SMTP"),
        "smtp_use_tls": fields.Boolean(description="SMTP TLS"),
        "smtp_use_ssl": fields.Boolean(description="SMTP SSL"),
        "smtp_username": fields.String(description="Utilisateur SMTP"),
        "smtp_password_configured": fields.Boolean(
            description="Mot de passe SMTP configuré (lecture seule)"
        ),
        # Cancellation policy
        "cancellation_policy": fields.Raw(
            description="Politique d'annulation (JSONB)", allow_null=True
        ),
    },
)

operational_settings_model = settings_ns.model(
    "OperationalSettings",
    {
        "service_area": fields.String(description="Zone de service", allow_null=True),
        "max_daily_bookings": fields.Integer(description="Limite courses/jour"),
        "dispatch_enabled": fields.Boolean(description="Dispatch automatique activé"),
        "latitude": fields.Float(description="Latitude du siège", allow_null=True),
        "longitude": fields.Float(description="Longitude du siège", allow_null=True),
    },
)

push_privacy_model = settings_ns.model(
    "PushPrivacy",
    {
        "push_privacy_mode": fields.String(
            description="detailed = nom client sur lockscreen ; discreet = pas de nom",
            enum=["detailed", "discreet"],
        ),
    },
)

# ==================== Routes ====================


@settings_ns.route("/operational")
class OperationalSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupérer les paramètres opérationnels."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found(
                    "Company",
                    None,
                    logger,
                )
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg,
                    logger_instance=logger,
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if company:
            return {
                "success": True,
                "data": {
                    "service_area": company.service_area,
                    "max_daily_bookings": company.max_daily_bookings,
                    "dispatch_enabled": company.dispatch_enabled,
                    "latitude": company.latitude,
                    "longitude": company.longitude,
                },
            }, 200
        return APIErrorHandler.handle_not_found(
            "Company",
            None,
            logger,
        )

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(operational_settings_model)
    def put(self):
        """Mettre à jour les paramètres opérationnels."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found(
                    "Company",
                    None,
                    logger,
                )
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg,
                    logger_instance=logger,
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found(
                "Company",
                None,
                logger,
            )

        data = request.get_json()

        try:
            if "service_area" in data:
                company.service_area = data["service_area"]
            if "max_daily_bookings" in data:
                company.max_daily_bookings = int(data["max_daily_bookings"])
            if "dispatch_enabled" in data:
                company.dispatch_enabled = bool(data["dispatch_enabled"])
            if "latitude" in data:
                company.latitude = float(data["latitude"]) if data["latitude"] else None
            if "longitude" in data:
                company.longitude = (
                    float(data["longitude"]) if data["longitude"] else None
                )

            db.session.commit()
            logger.info(
                "[Settings] Operational settings updated for company %s", company.id
            )

            from shared.audit_helpers import audit_log
            audit_log("settings_updated", "settings", resource_type="operational_settings", resource_id=company.id)

            return {
                "success": True,
                "message": "Paramètres opérationnels mis à jour",
                "data": {
                    "service_area": company.service_area,
                    "max_daily_bookings": company.max_daily_bookings,
                    "dispatch_enabled": company.dispatch_enabled,
                    "latitude": company.latitude,
                    "longitude": company.longitude,
                },
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error("[Settings] Error updating operational settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@settings_ns.route("/push-privacy")
class PushPrivacySettings(Resource):
    """Réglage mode discret pour les push (pas de nom client sur lockscreen).

    Contrôle d'accès / multi-tenant :
    - @jwt_required() + @role_required(UserRole.company).
    - get_company_from_token() détermine la company (donc le user = company.user_id).
    - user = User.query.get(company.user_id) : aucun user_id ni company_id dans le body.
    - Seules valeurs acceptées en PATCH : "detailed" | "discreet" (après .strip().lower()), sinon 400.
    """

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupérer le mode push (detailed | discreet)."""
        company, err, code = get_company_from_token()
        if err:
            return err, code
        user = User.query.get(company.user_id) if company else None
        if not user:
            return {"error": "User not found"}, HTTPStatus.NOT_FOUND
        mode = getattr(user, "push_privacy_mode", None) or "detailed"
        return {"push_privacy_mode": mode}, 200

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(push_privacy_model)
    def patch(self):
        """Mettre à jour le mode push (detailed | discreet)."""
        company, err, code = get_company_from_token()
        if err:
            return err, code
        user = User.query.get(company.user_id) if company else None
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


@settings_ns.route("/billing")
class BillingSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.marshal_with(billing_settings_model)
    def get(self):
        """Récupérer les paramètres de facturation."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found(
                    "Company",
                    None,
                    logger,
                )
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg,
                    logger_instance=logger,
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        # Récupérer ou créer les billing settings
        if not company:
            return APIErrorHandler.handle_not_found(
                "Company",
                None,
                logger,
            )

        # ✅ Récupérer directement le modèle SQLAlchemy (pas via repository qui retourne un DTO)
        try:
            import os

            BILLING_DEBUG = os.getenv("BILLING_DEBUG", "0") == "1"

            billing = CompanyBillingSettings.query.filter_by(
                company_id=company.id
            ).first()

            if BILLING_DEBUG:
                logger.info(
                    "[BILLING_DEBUG] GET billing settings: company_id=%s, billing_found=%s, billing_id=%s",
                    company.id,
                    billing is not None,
                    billing.id if billing else None,
                )

            if not billing:
                # Créer avec valeurs par défaut
                if BILLING_DEBUG:
                    logger.info(
                        "[BILLING_DEBUG] GET creating default billing settings for company_id=%s",
                        company.id,
                    )
                billing = CompanyBillingSettings()
                billing.company_id = company.id
                billing.payment_terms_days = 10
                billing.overdue_fee = 15.00
                billing.reminder1_fee = 0.00
                billing.reminder2_fee = 40.00
                billing.reminder3_fee = 0.00
                billing.reminder_schedule_days = {"1": 10, "2": 5, "3": 5}
                billing.auto_reminders_enabled = True
                billing.invoice_number_format = "{PREFIX}-{YYYY}-{MM}-{SEQ4}"
                billing.invoice_prefix = "EM"
                billing.pdf_template_variant = "default"
                db.session.add(billing)
                db.session.commit()
                if BILLING_DEBUG:
                    logger.info(
                        "[BILLING_DEBUG] GET default billing settings created: billing_id=%s",
                        billing.id,
                    )

            # Log des valeurs bancaires avant to_dict
            if BILLING_DEBUG:
                logger.info(
                    (
                        "[BILLING_DEBUG] GET before to_dict: company_id=%s, billing_id=%s, "
                        "_iban_raw=%s, iban decrypted=%s, _qr_iban_raw=%s, qr_iban decrypted=%s, esr_ref_base=%s"
                    ),
                    company.id,
                    billing.id,
                    getattr(billing, "_iban_raw", None),
                    billing.iban,
                    getattr(billing, "_qr_iban_raw", None),
                    billing.qr_iban,
                    billing.esr_ref_base,
                )

            result = billing.to_dict()

            # Log des valeurs dans le résultat
            if BILLING_DEBUG:
                logger.info(
                    (
                        "[BILLING_DEBUG] GET to_dict result: company_id=%s, "
                        "iban in result=%s, iban value=%s, "
                        "qr_iban in result=%s, qr_iban value=%s, "
                        "esr_ref_base in result=%s, esr_ref_base value=%s"
                    ),
                    company.id,
                    "iban" in result,
                    result.get("iban"),
                    "qr_iban" in result,
                    result.get("qr_iban"),
                    "esr_ref_base" in result,
                    result.get("esr_ref_base"),
                )

            return result, 200
        except Exception as e:
            # #region agent log
            import json
            import time
            from pathlib import Path

            log_path = Path("/app/.cursor/debug.log")
            log_data = {
                "location": "company_settings.py:BillingSettings.get",
                "message": "Error in get method",
                "data": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D",
            }
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
            logger.exception("[Settings] Error fetching billing settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(billing_settings_model, validate=False)
    def put(self):
        """Mettre à jour les paramètres de facturation."""
        logger.info("[Settings] PUT /billing handler entered")
        company, err, code = get_company_from_token()
        if err:
            logger.warning("[Settings] PUT /billing company error: %s (code=%s)", err, code)
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found(
                    "Company",
                    None,
                    logger,
                )
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg,
                    logger_instance=logger,
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        data = request.get_json() or {}

        if not company:
            return APIErrorHandler.handle_not_found(
                "Company",
                None,
                logger,
            )

        logger.info(
            "[Settings] Billing settings update request for company %s: %s",
            company.id,
            data,
        )

        try:
            # ✅ Récupérer directement le modèle SQLAlchemy (pas via repository qui retourne un DTO)
            billing = CompanyBillingSettings.query.filter_by(
                company_id=company.id
            ).first()

            if not billing:
                billing = CompanyBillingSettings()
                billing.company_id = company.id
                db.session.add(billing)
                # ✅ BLINDAGE: Gérer les doublons concurrents (contrainte unique sur company_id)
                try:
                    db.session.flush()  # Tester la contrainte unique avant commit
                except Exception as e:
                    db.session.rollback()
                    # Recharger si un autre thread a créé entre temps
                    billing = CompanyBillingSettings.query.filter_by(
                        company_id=company.id
                    ).first()
                    if not billing:
                        logger.error(
                            "[Settings] Failed to create billing settings for company %s: %s",
                            company.id,
                            e,
                        )
                        raise

            # ✅ CORRECTION: Traiter iban/qr_iban/esr_ref_base séparément
            # car ce sont des @hybrid_property et setattr peut ne pas fonctionner correctement
            import os

            BILLING_DEBUG = os.getenv("BILLING_DEBUG", "0") == "1"

            # Gestion spéciale pour les champs bancaires (hybrid_property)
            if "iban" in data:
                value = data["iban"]
                if BILLING_DEBUG:
                    logger.info(
                        "[BILLING_DEBUG] PUT iban: company_id=%s, value=%s, type=%s, is_none=%s, is_empty=%s",
                        company.id,
                        value,
                        type(value).__name__,
                        value is None,
                        value == "",
                    )
                if value is None or value == "":
                    billing.iban = None
                else:
                    billing.iban = value
                if BILLING_DEBUG:
                    logger.info(
                        "[BILLING_DEBUG] PUT iban after set: _iban_raw=%s, iban decrypted=%s",
                        getattr(billing, "_iban_raw", None),
                        billing.iban,
                    )

            if "qr_iban" in data:
                value = data["qr_iban"]
                if BILLING_DEBUG:
                    logger.info(
                        "[BILLING_DEBUG] PUT qr_iban: company_id=%s, value=%s, type=%s, is_none=%s, is_empty=%s",
                        company.id,
                        value,
                        type(value).__name__,
                        value is None,
                        value == "",
                    )
                if value is None or value == "":
                    billing.qr_iban = None
                else:
                    billing.qr_iban = value
                if BILLING_DEBUG:
                    logger.info(
                        "[BILLING_DEBUG] PUT qr_iban after set: _qr_iban_raw=%s, qr_iban decrypted=%s",
                        getattr(billing, "_qr_iban_raw", None),
                        billing.qr_iban,
                    )

            if "esr_ref_base" in data:
                value = data["esr_ref_base"]
                if BILLING_DEBUG:
                    logger.info(
                        "[BILLING_DEBUG] PUT esr_ref_base: company_id=%s, value=%s, type=%s",
                        company.id,
                        value,
                        type(value).__name__,
                    )
                billing.esr_ref_base = value if (value and value != "") else None

            # Mise à jour des autres champs
            updatable_fields = [
                "payment_terms_days",
                "overdue_fee",
                "reminder1_fee",
                "reminder2_fee",
                "reminder3_fee",
                "material_delivery_price_fixed",
                "reminder_schedule_days",
                "auto_reminders_enabled",
                "email_sender",
                "invoice_number_format",
                "invoice_prefix",
                "invoice_message_template",
                "reminder1_template",
                "reminder2_template",
                "reminder3_template",
                "email_signature_mode",
                "email_signature_text",
                "signature_name",
                "signature_title",
                "signature_company",
                "signature_phone_main",
                "signature_phone_mobile",
                "signature_email",
                "signature_website",
                "signature_address_line",
                "signature_zip",
                "signature_city",
                "email_signature_html_template",
                "legal_footer",
                "pdf_template_variant",
                "smtp_server",
                "smtp_port",
                "smtp_use_tls",
                "smtp_use_ssl",
                "smtp_username",
                "smtp_enabled",
            ]

            # smtp_password: hybrid_property chiffree, ne pas ecraser si vide
            if "smtp_password" in data:
                value = data["smtp_password"]
                if value and isinstance(value, str) and value.strip():
                    billing.smtp_password = value.strip()

            # Cancellation policy : validation via Marshmallow schema
            if "cancellation_policy" in data:
                raw_policy = data["cancellation_policy"]
                if raw_policy is None and billing.cancellation_policy is not None:
                    pass  # ne pas ecraser une policy existante avec null
                elif raw_policy is None:
                    billing.cancellation_policy = None
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(billing, "cancellation_policy")
                else:
                    from marshmallow import ValidationError as MarshmallowValidationError  # noqa: I001
                    from application.bookings.cancellation_policy_schema import (
                        CancellationPolicySchema,
                    )

                    try:
                        validated_policy = CancellationPolicySchema().load(raw_policy)
                        billing.cancellation_policy = validated_policy
                    except MarshmallowValidationError as e:
                        logger.warning(
                            "[Settings] Cancellation policy validation failed: %s (input: %s)",
                            e.messages,
                            raw_policy,
                        )
                        return {
                            "success": False,
                            "error": f"Cancellation policy invalid: {e.messages}",
                        }, 400

                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(billing, "cancellation_policy")

            for field in updatable_fields:
                if field in data:
                    value = data[field]
                    # Gérer les valeurs None/empty pour les champs optionnels
                    if value is None or value == "":
                        if field in [
                            "material_delivery_price_fixed",
                            "email_sender",
                            "invoice_message_template",
                            "reminder1_template",
                            "reminder2_template",
                            "reminder3_template",
                            "email_signature_text",
                            "signature_name",
                            "signature_title",
                            "signature_company",
                            "signature_phone_main",
                            "signature_phone_mobile",
                            "signature_email",
                            "signature_website",
                            "signature_address_line",
                            "signature_zip",
                            "signature_city",
                            "email_signature_html_template",
                            "legal_footer",
                            "smtp_server",
                            "smtp_username",
                        ]:
                            setattr(billing, field, None)
                        continue
                    # Conversion spéciale pour material_delivery_price_fixed (Decimal, >= 0)
                    if field == "material_delivery_price_fixed":
                        from decimal import Decimal, InvalidOperation

                        try:
                            parsed = float(value)
                            if parsed < 0:
                                billing.material_delivery_price_fixed = None
                            else:
                                billing.material_delivery_price_fixed = Decimal(
                                    str(parsed)
                                ).quantize(Decimal("0.01"))
                        except (ValueError, TypeError, InvalidOperation):
                            billing.material_delivery_price_fixed = None
                        continue
                    # Conversion spéciale pour reminder_schedule_days
                    # (doit être un dict)
                    if field == "reminder_schedule_days" and isinstance(value, dict):
                        existing = billing.reminder_schedule_days or {}
                        merged = {**existing, **{str(k): int(v) for k, v in value.items() if v is not None}}  # type: ignore[arg-type]
                        setattr(billing, field, merged)
                    else:
                        setattr(billing, field, value)

            # Gestion de la TVA
            if "vat_applicable" in data:
                billing.vat_applicable = bool(data["vat_applicable"])

            if "vat_rate" in data:
                from decimal import Decimal, InvalidOperation

                rate_value = data.get("vat_rate")
                if rate_value is None or rate_value == "":
                    billing.vat_rate = None
                else:
                    try:
                        # Convertir en float d'abord pour gérer les NaN, puis en Decimal
                        float_value = float(rate_value)
                        MAX_VAT_RATE = 100.0  # Taux TVA maximum (100%)
                        if float_value <= 0 or float_value > MAX_VAT_RATE:
                            logger.warning(
                                "Taux TVA hors limites (0-%s): %s",
                                MAX_VAT_RATE,
                                rate_value,
                            )
                            billing.vat_rate = None
                        else:
                            billing.vat_rate = Decimal(str(rate_value)).quantize(
                                Decimal("0.01")
                            )
                            logger.info("Taux TVA mis à jour: %s%%", billing.vat_rate)
                    except (InvalidOperation, ValueError, TypeError) as e:
                        logger.warning(
                            "Taux TVA invalide: %s (erreur: %s)", rate_value, e
                        )
                        billing.vat_rate = None

            if "vat_label" in data:
                billing.vat_label = data.get("vat_label") or None

            if "vat_number" in data:
                billing.vat_number = data.get("vat_number") or None

            # Log structure des champs mis a jour (toujours actif)
            all_tracked = [
                *updatable_fields,
                "iban", "qr_iban", "esr_ref_base", "smtp_password",
                "cancellation_policy", "vat_applicable", "vat_rate",
                "vat_label", "vat_number",
            ]
            fields_received = [f for f in all_tracked if f in data]
            logger.info(
                "[Settings] PUT billing: company_id=%s, fields_in_payload=%s",
                company.id,
                fields_received,
            )

            # Log avant commit avec vérification de détection de changement SQLAlchemy
            if BILLING_DEBUG:
                from sqlalchemy.orm.attributes import flag_modified

                is_modified = db.session.is_modified(billing, include_collections=False)
                dirty = billing in db.session.dirty
                logger.info(
                    (
                        "[BILLING_DEBUG] PUT before commit: company_id=%s, billing_id=%s, "
                        "is_modified=%s, dirty=%s, "
                        "_iban_raw=%s, iban decrypted=%s, _qr_iban_raw=%s, qr_iban decrypted=%s, esr_ref_base=%s"
                    ),
                    company.id,
                    billing.id,
                    is_modified,
                    dirty,
                    getattr(billing, "_iban_raw", None),
                    billing.iban,
                    getattr(billing, "_qr_iban_raw", None),
                    billing.qr_iban,
                    billing.esr_ref_base,
                )
                # Si is_modified=False mais qu'on a modifié, forcer le flag (sécurité)
                if not is_modified and (
                    "iban" in data or "qr_iban" in data or "esr_ref_base" in data
                ):
                    logger.warning(
                        "[BILLING_DEBUG] WARNING: is_modified=False but banking fields were updated, forcing flag_modified"
                    )
                    flag_modified(billing, "_iban_raw")
                    flag_modified(billing, "_qr_iban_raw")

            db.session.commit()
            db.session.refresh(billing)

            if BILLING_DEBUG:
                logger.info(
                    (
                        "[BILLING_DEBUG] PUT after commit (refreshed): company_id=%s, billing_id=%s, "
                        "_iban_raw=%s, iban decrypted=%s, _qr_iban_raw=%s, qr_iban decrypted=%s, esr_ref_base=%s"
                    ),
                    company.id,
                    billing.id,
                    getattr(billing, "_iban_raw", None),
                    billing.iban,
                    getattr(billing, "_qr_iban_raw", None),
                    billing.qr_iban,
                    billing.esr_ref_base,
                )

            logger.info(
                "[Settings] Billing settings updated for company %s", company.id
            )

            result_dict = billing.to_dict()

            # Log du résultat final
            if BILLING_DEBUG:
                logger.info(
                    (
                        "[BILLING_DEBUG] PUT to_dict result: company_id=%s, "
                        "iban in result=%s, iban value=%s, "
                        "qr_iban in result=%s, qr_iban value=%s, "
                        "esr_ref_base in result=%s, esr_ref_base value=%s"
                    ),
                    company.id,
                    "iban" in result_dict,
                    result_dict.get("iban"),
                    "qr_iban" in result_dict,
                    result_dict.get("qr_iban"),
                    "esr_ref_base" in result_dict,
                    result_dict.get("esr_ref_base"),
                )

            from shared.audit_helpers import audit_log
            audit_log("settings_updated", "settings", resource_type="billing_settings", resource_id=company.id)

            return {
                "success": True,
                "message": "Paramètres de facturation mis à jour",
                "data": result_dict,
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.exception("[Settings] Error updating billing settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@settings_ns.route("/planning")
class PlanningSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupérer les paramètres de planning."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found(
                    "Company",
                    None,
                    logger,
                )
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg,
                    logger_instance=logger,
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found(
                "Company",
                None,
                logger,
            )

        # ✅ Récupérer directement le modèle SQLAlchemy (pas via repository qui retourne un DTO)
        planning = CompanyPlanningSettings.query.filter_by(
            company_id=company.id
        ).first()

        if not planning:
            planning = CompanyPlanningSettings()
            planning.company_id = company.id
            planning.settings = {}
            db.session.add(planning)
            db.session.commit()

        return {"success": True, "data": planning.settings}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        """Mettre à jour les paramètres de planning."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found(
                    "Company",
                    None,
                    logger,
                )
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg,
                    logger_instance=logger,
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        data = request.get_json()

        if not company:
            return APIErrorHandler.handle_not_found(
                "Company",
                None,
                logger,
            )

        try:
            # ✅ Récupérer directement le modèle SQLAlchemy (pas via repository qui retourne un DTO)
            planning = CompanyPlanningSettings.query.filter_by(
                company_id=company.id
            ).first()

            if not planning:
                planning = CompanyPlanningSettings()
                planning.company_id = company.id
                planning.settings = data.get("settings", {})
                db.session.add(planning)
            else:
                planning.settings = data.get("settings", {})

            db.session.commit()
            logger.info(
                "[Settings] Planning settings updated for company %s", company.id
            )

            return {
                "success": True,
                "message": "Paramètres de planning mis à jour",
                "data": planning.settings,
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error("[Settings] Error updating planning settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@settings_ns.route("/billing/clinic-mappings")
class ClinicBillingMappings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Lister les mappings clinique → billing_party pour l'entreprise courante."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found("Company", None, logger)
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg, logger_instance=logger
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        mappings = (
            ClinicBillingPartyMapping.query.filter_by(company_id=company.id)
            .order_by(ClinicBillingPartyMapping.id.desc())
            .all()
        )
        payload = []
        for m in mappings:
            clinic = Company.query.filter_by(id=m.clinic_company_id).first()
            bp = BillingParty.query.filter_by(id=m.billing_party_id).first()
            clinic_display_name = clinic.name if clinic else None
            if clinic and company:
                linked_client = Client.query.filter_by(
                    company_id=company.id,
                    is_institution=True,
                    linked_institution_id=clinic.id,
                ).first()
                if linked_client and linked_client.institution_name:
                    # Préfère le nom institution "source of truth" côté transporteur.
                    # Utile quand certains vieux enregistrements Company ont un nom encodé (ex: "Ani??res").
                    clinic_display_name = linked_client.institution_name
            payload.append(
                {
                    "id": m.id,
                    "clinic_company_id": m.clinic_company_id,
                    "clinic_company_name": clinic_display_name,
                    "billing_party_id": m.billing_party_id,
                    "billing_party_name": bp.display_name if bp else None,
                    "is_active": bool(m.is_active),
                }
            )
        return {"success": True, "data": payload}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):  # noqa: PLR0911
        """Créer/mettre à jour un mapping clinique → billing_party (upsert)."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found("Company", None, logger)
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg, logger_instance=logger
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        data = request.get_json() or {}
        clinic_company_id_raw = data.get("clinic_company_id")
        billing_party_id_raw = data.get("billing_party_id")
        is_active = data.get("is_active", True)

        if clinic_company_id_raw is None or billing_party_id_raw is None:
            return APIErrorHandler.handle_validation_error(
                "clinic_company_id et billing_party_id sont requis",
                logger_instance=logger,
            )

        try:
            clinic_company_id = int(clinic_company_id_raw)
            billing_party_id = int(billing_party_id_raw)
        except (TypeError, ValueError):
            return APIErrorHandler.handle_validation_error(
                "clinic_company_id et billing_party_id doivent être des entiers",
                logger_instance=logger,
            )

        clinic = Company.query.filter_by(id=clinic_company_id).first()
        if not clinic:
            return APIErrorHandler.handle_validation_error(
                "Clinique (company) introuvable", logger_instance=logger
            )

        bp = BillingParty.query.filter_by(
            id=billing_party_id, company_id=company.id
        ).first()
        if not bp:
            return APIErrorHandler.handle_validation_error(
                "BillingParty introuvable ou n'appartient pas à l'entreprise",
                logger_instance=logger,
            )

        mapping = ClinicBillingPartyMapping.query.filter_by(
            company_id=company.id, clinic_company_id=clinic_company_id
        ).first()
        if not mapping:
            mapping = ClinicBillingPartyMapping()
            mapping.company_id = company.id
            mapping.clinic_company_id = clinic_company_id
            db.session.add(mapping)

        mapping.billing_party_id = billing_party_id
        mapping.is_active = bool(is_active)
        db.session.commit()

        return {"success": True, "message": "Mapping mis à jour"}, 200


@settings_ns.route("/billing/clinic-mappings/<int:clinic_company_id>")
class ClinicBillingMappingByClinic(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, clinic_company_id: int):
        """Récupérer le mapping pour une clinique spécifique."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found("Company", None, logger)
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg, logger_instance=logger
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        mapping = ClinicBillingPartyMapping.query.filter_by(
            company_id=company.id, clinic_company_id=clinic_company_id
        ).first()

        if not mapping:
            return {"success": True, "data": None}, 200

        clinic = Company.query.filter_by(id=mapping.clinic_company_id).first()
        bp = BillingParty.query.filter_by(id=mapping.billing_party_id).first()
        clinic_display_name = clinic.name if clinic else None
        if clinic and company:
            linked_client = Client.query.filter_by(
                company_id=company.id,
                is_institution=True,
                linked_institution_id=clinic.id,
            ).first()
            if linked_client and linked_client.institution_name:
                clinic_display_name = linked_client.institution_name

        payload = {
            "id": mapping.id,
            "clinic_company_id": mapping.clinic_company_id,
            "clinic_company_name": clinic_display_name,
            "billing_party_id": mapping.billing_party_id,
            "billing_party_name": bp.display_name if bp else None,
            "is_active": bool(mapping.is_active),
        }

        return {"success": True, "data": payload}, 200


@settings_ns.route("/billing/parties")
class BillingParties(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Lister les BillingParty de l'entreprise courante (pour sélection UI)."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found("Company", None, logger)
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg, logger_instance=logger
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        only_active = request.args.get("active", "true").strip().lower() != "false"
        q = BillingParty.query.filter_by(company_id=company.id)
        if only_active:
            q = q.filter(BillingParty.is_active.is_(True))
        parties = q.order_by(BillingParty.display_name.asc()).all()
        return {"success": True, "data": [p.to_dict() for p in parties]}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):  # noqa: PLR0911
        """Créer un BillingParty (V1) depuis le backoffice."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found("Company", None, logger)
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg, logger_instance=logger
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        data = request.get_json() or {}
        display_name = (data.get("display_name") or "").strip()
        type_raw = (data.get("type") or "").strip().lower()
        billing_address = data.get("billing_address")
        contact_email = data.get("contact_email")
        contact_phone = data.get("contact_phone")
        external_ref = data.get("external_ref")
        is_active = data.get("is_active", True)

        if not display_name:
            return APIErrorHandler.handle_validation_error(
                "display_name est requis",
                logger_instance=logger,
            )

        try:
            bp_type = BillingPartyType(type_raw)
        except Exception:
            return APIErrorHandler.handle_validation_error(
                "type invalide (ex: clinic, hospital, ems, opad, other, patient)",
                logger_instance=logger,
            )

        try:
            bp = BillingParty()
            bp.company_id = company.id
            bp.type = bp_type
            bp.display_name = display_name
            bp.billing_address = billing_address
            bp.contact_email = contact_email
            bp.contact_phone = contact_phone
            bp.external_ref = external_ref
            bp.is_active = bool(is_active)
            db.session.add(bp)
            db.session.commit()
        except ValueError as e:
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            db.session.rollback()
            logger.error("[Settings] Error creating BillingParty: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": bp.to_dict()}, 201


@settings_ns.route("/billing/parties/<int:billing_party_id>")
class BillingPartyById(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, billing_party_id: int):
        """Mettre à jour un BillingParty."""
        company, err, code = get_company_from_token()
        if err:
            error_msg = err.get("error", "Company not found")
            error_response, status_code = (
                APIErrorHandler.handle_not_found("Company", None, logger)
                if code == HTTPStatus.NOT_FOUND
                else APIErrorHandler.handle_validation_error(
                    error_msg, logger_instance=logger
                )
            )
            return {
                "success": False,
                "error": error_response.get("error", error_msg),
            }, status_code

        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        bp = BillingParty.query.filter_by(
            id=billing_party_id, company_id=company.id
        ).first()
        if not bp:
            return APIErrorHandler.handle_not_found(
                "BillingParty", billing_party_id, logger
            )

        data = request.get_json() or {}
        display_name = data.get("display_name")
        billing_address = data.get("billing_address")
        contact_email = data.get("contact_email")
        contact_phone = data.get("contact_phone")
        is_active = data.get("is_active")

        try:
            if display_name is not None:
                bp.display_name = display_name.strip() if display_name else None
            if billing_address is not None:
                bp.billing_address = (
                    billing_address.strip() if billing_address else None
                )
            if contact_email is not None:
                bp.contact_email = contact_email.strip() if contact_email else None
            if contact_phone is not None:
                bp.contact_phone = contact_phone.strip() if contact_phone else None
            if is_active is not None:
                bp.is_active = bool(is_active)

            db.session.commit()
        except ValueError as e:
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            db.session.rollback()
            logger.error("[Settings] Error updating BillingParty: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": bp.to_dict()}, 200
