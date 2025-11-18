"""Routes API pour les paramètres avancés de l'entreprise."""

import logging

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields

from ext import db, role_required
from models import CompanyBillingSettings, CompanyPlanningSettings, UserRole
from routes.companies import get_company_from_token

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
        "reminder_schedule_days": fields.Raw(description="Planning des rappels"),
        "auto_reminders_enabled": fields.Boolean(description="Rappels automatiques activés"),
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
        "legal_footer": fields.String(description="Pied de page légal"),
        "pdf_template_variant": fields.String(description="Variante template PDF"),
        # TVA
        "vat_applicable": fields.Boolean(description="TVA applicable", allow_null=True),
        "vat_rate": fields.Float(description="Taux de TVA (%)", allow_null=True),
        "vat_label": fields.String(description="Libellé TVA", allow_null=True),
        "vat_number": fields.String(description="Numéro de TVA", allow_null=True),
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

# ==================== Routes ====================


@settings_ns.route("/operational")
class OperationalSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupérer les paramètres opérationnels."""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code

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
        return {"success": False, "error": "Company not found"}, 404

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(operational_settings_model)
    def put(self):
        """Mettre à jour les paramètres opérationnels."""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code

        if not company:
            return {"success": False, "error": "Company not found"}, 404

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
                company.longitude = float(data["longitude"]) if data["longitude"] else None

            db.session.commit()
            logger.info("[Settings] Operational settings updated for company %s", company.id)

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
            return {"success": False, "error": str(e)}, 500


@settings_ns.route("/billing")
class BillingSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.marshal_with(billing_settings_model)
    def get(self):
        """Récupérer les paramètres de facturation."""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code

        # Récupérer ou créer les billing settings
        if not company:
            return {"success": False, "error": "Company not found"}, 404

        billing = CompanyBillingSettings.query.filter_by(company_id=company.id).first()

        if not billing:
            # Créer avec valeurs par défaut
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

        return billing.to_dict(), 200

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(billing_settings_model, validate=False)
    def put(self):
        """Mettre à jour les paramètres de facturation."""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code

        data = request.get_json() or {}

        if not company:
            return {"success": False, "error": "Company not found"}, 404

        logger.info("[Settings] Billing settings update request for company %s: %s", company.id, data)

        try:
            billing = CompanyBillingSettings.query.filter_by(company_id=company.id).first()

            if not billing:
                billing = CompanyBillingSettings()
                billing.company_id = company.id
                db.session.add(billing)

            # Mise à jour des champs
            updatable_fields = [
                "payment_terms_days",
                "overdue_fee",
                "reminder1_fee",
                "reminder2_fee",
                "reminder3_fee",
                "reminder_schedule_days",
                "auto_reminders_enabled",
                "email_sender",
                "invoice_number_format",
                "invoice_prefix",
                "iban",
                "qr_iban",
                "esr_ref_base",
                "invoice_message_template",
                "reminder1_template",
                "reminder2_template",
                "reminder3_template",
                "legal_footer",
                "pdf_template_variant",
            ]

            for field in updatable_fields:
                if field in data:
                    value = data[field]
                    # Gérer les valeurs None/empty pour les champs optionnels
                    if value is None or value == "":
                        if field in [
                            "email_sender",
                            "iban",
                            "qr_iban",
                            "esr_ref_base",
                            "invoice_message_template",
                            "reminder1_template",
                            "reminder2_template",
                            "reminder3_template",
                            "legal_footer",
                        ]:
                            setattr(billing, field, None)
                        continue
                    # Conversion spéciale pour reminder_schedule_days (doit être un dict)
                    if field == "reminder_schedule_days" and isinstance(value, dict):
                        # S'assurer que les clés sont des strings
                        normalized = {str(k): int(v) for k, v in value.items() if v is not None}
                        setattr(billing, field, normalized)
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
                            logger.warning("Taux TVA hors limites (0-%s): %s", MAX_VAT_RATE, rate_value)
                            billing.vat_rate = None
                        else:
                            billing.vat_rate = Decimal(str(rate_value)).quantize(Decimal("0.01"))
                            logger.info("Taux TVA mis à jour: %s%%", billing.vat_rate)
                    except (InvalidOperation, ValueError, TypeError) as e:
                        logger.warning("Taux TVA invalide: %s (erreur: %s)", rate_value, e)
                        billing.vat_rate = None

            if "vat_label" in data:
                billing.vat_label = data.get("vat_label") or None

            if "vat_number" in data:
                billing.vat_number = data.get("vat_number") or None

            db.session.commit()
            logger.info("[Settings] Billing settings updated for company %s", company.id)

            return {"success": True, "message": "Paramètres de facturation mis à jour", "data": billing.to_dict()}, 200
        except Exception as e:
            db.session.rollback()
            logger.exception("[Settings] Error updating billing settings: %s", e)
            return {"success": False, "error": str(e)}, 500


@settings_ns.route("/planning")
class PlanningSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupérer les paramètres de planning."""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code

        if not company:
            return {"success": False, "error": "Company not found"}, 404

        planning = CompanyPlanningSettings.query.filter_by(company_id=company.id).first()

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
            return {"success": False, "error": err}, code

        data = request.get_json()

        if not company:
            return {"success": False, "error": "Company not found"}, 404

        try:
            planning = CompanyPlanningSettings.query.filter_by(company_id=company.id).first()

            if not planning:
                planning = CompanyPlanningSettings()
                planning.company_id = company.id
                planning.settings = data.get("settings", {})
                db.session.add(planning)
            else:
                planning.settings = data.get("settings", {})

            db.session.commit()
            logger.info("[Settings] Planning settings updated for company %s", company.id)

            return {"success": True, "message": "Paramètres de planning mis à jour", "data": planning.settings}, 200
        except Exception as e:
            db.session.rollback()
            logger.error("[Settings] Error updating planning settings: %s", e)
            return {"success": False, "error": str(e)}, 500
