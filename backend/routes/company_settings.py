"""Routes API pour les paramètres avancés de l'entreprise."""

import json
import logging
import re
from http import HTTPStatus
from typing import Any

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields

from ext import db, role_required
from models import (
    BillingParty,
    BillingPartyType,
    Client,
    ClinicBillingPartyMapping,
    Company,
    CompanyBillingSettings,
    CompanyPlanningSettings,
    GeoUnit,
    PricingModelType,
    PricingProfile,
    PricingProfileVersion,
    ServiceArea,
    User,
    UserRole,
)
from routes.companies import get_company_from_token
from schemas.pricing_schemas import PricingZoneMatrixSettingsSchema
from schemas.service_area_schemas import (
    ServiceAreaCreateSchema,
    ServiceAreaUpdateSchema,
)
from schemas.validation_utils import validate_request
from services.pricing.pricing_engine import (
    build_zone_matrix_summary,
    validate_company_pricing_rules,
    validate_zone_matrix_rules,
)
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

SERVICE_AREA_MAX_LENGTH = 200
SERVICE_AREA_JSON_VERSION = 1
SERVICE_AREA_MODE_SET = {"canton", "district", "commune"}
SERVICE_AREA_TOKEN_PATTERN = re.compile(r"^(commune|district|canton):[A-Za-z0-9_-]+$")


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _build_pricing_summary(company_id: int) -> dict[str, object] | None:
    profile = (
        PricingProfile.query.filter_by(company_id=company_id, is_active=True)
        .order_by(PricingProfile.created_at.desc())
        .first()
    )
    if not profile:
        return None

    version = profile.current_version
    if not version:
        version = (
            profile.versions and sorted(profile.versions, key=lambda v: int(v.version), reverse=True)[0]
        ) or None
    if not version:
        return {
            "enabled": True,
            "profile_name": profile.name,
            "model_type": profile.model_type.value if profile.model_type else None,
            "currency": profile.currency or "CHF",
            "version": None,
            "label": "Profil tarifaire sans version active",
            "details": {},
        }

    rules = version.rules_json or {}
    model = str(rules.get("model") or (profile.model_type.value if profile.model_type else "")).lower()
    currency = profile.currency or "CHF"

    details: dict[str, object] = {}
    label = "Tarification"

    if model == "flat":
        label = "Prix fixe (canton/zone)"
        if isinstance(rules.get("components"), dict):
            details = {
                "base_fee": _to_float_or_none((rules.get("components") or {}).get("base", {}).get("amount")),
                "minimum": _to_float_or_none((rules.get("caps") or {}).get("minimum")),
                "zone_set_id": rules.get("zone_set_id"),
            }
        else:
            details = {
                "base_fee": _to_float_or_none(rules.get("base_fee")),
                "minimum": _to_float_or_none(rules.get("minimum")),
            }
    elif model in {"zone", "zone_matrix"}:
        if model == "zone_matrix" or (
            isinstance(rules.get("zones"), list) and isinstance(rules.get("matrix"), dict)
        ):
            matrix_summary = build_zone_matrix_summary(rules)
            label = "Matrice de zones"
            details = {
                "zones_count": matrix_summary.get("zones_count", 0),
                "transitions_count": matrix_summary.get("transitions_count", 0),
                "matrix_symmetry": bool(rules.get("matrix_symmetry", False)),
                "default_same_zone_price": _to_float_or_none(rules.get("default_same_zone_price")),
                "minimum": _to_float_or_none(rules.get("minimum")),
            }
            model = "zone_matrix"
        else:
            pricing = rules.get("pricing") or {}
            weekday = pricing.get("weekday") if isinstance(pricing, dict) else {}
            weekend = pricing.get("weekend") if isinstance(pricing, dict) else {}
            label = "Prix par zone"
            details = {
                "weekday_one_way": _to_float_or_none((weekday or {}).get("one_way")),
                "weekday_round_trip": _to_float_or_none((weekday or {}).get("round_trip")),
                "weekend_one_way": _to_float_or_none((weekend or {}).get("one_way")),
                "weekend_round_trip": _to_float_or_none((weekend or {}).get("round_trip")),
                "minimum": _to_float_or_none(rules.get("minimum")),
            }
    elif model == "distance":
        label = "Prix au km"
        if isinstance(rules.get("components"), dict):
            details = {
                "base_fee": _to_float_or_none((rules.get("components") or {}).get("base", {}).get("amount")),
                "per_km": _to_float_or_none((rules.get("components") or {}).get("distance", {}).get("per_km")),
                "minimum": _to_float_or_none((rules.get("caps") or {}).get("minimum")),
                "zone_set_id": rules.get("zone_set_id"),
            }
        else:
            details = {
                "base_fee": _to_float_or_none(rules.get("base_fee")),
                "per_km": _to_float_or_none(rules.get("per_km")),
                "minimum": _to_float_or_none(rules.get("minimum")),
            }
    elif model in {"zone_count", "hybrid_stack"}:
        label = "Tarification par zones admin"
        details = {
            "base_fee": _to_float_or_none((rules.get("components") or {}).get("base", {}).get("amount")),
            "zone_unit": _to_float_or_none((rules.get("components") or {}).get("zone_count", {}).get("unit_price")),
            "per_km": _to_float_or_none((rules.get("components") or {}).get("distance", {}).get("per_km")),
            "minimum": _to_float_or_none((rules.get("caps") or {}).get("minimum")),
            "zone_set_id": rules.get("zone_set_id"),
        }
    else:
        label = f"Mode tarifaire: {model or 'inconnu'}"
        details = {"raw_model": model or None}

    return {
        "enabled": True,
        "profile_name": profile.name,
        "model_type": model or None,
        "currency": currency,
        "version": int(version.version) if version.version is not None else None,
        "label": label,
        "details": details,
    }


def _get_company_pricing_rules(company_id: int) -> dict[str, Any] | None:
    profile = (
        PricingProfile.query.filter_by(company_id=company_id, is_active=True)
        .order_by(PricingProfile.created_at.desc())
        .first()
    )
    if not profile:
        return None
    version = _get_current_pricing_version(profile)
    if not version:
        return None
    rules = version.rules_json or {}
    return dict(rules)


def _get_active_pricing_meta(company_id: int) -> dict[str, Any]:
    profile = (
        PricingProfile.query.filter_by(company_id=company_id, is_active=True)
        .order_by(PricingProfile.created_at.desc())
        .first()
    )
    if not profile:
        return {
            "active_pricing_profile_id": None,
            "active_pricing_profile_version_id": None,
            "active_pricing_rules_json_hash": None,
        }
    version = _get_current_pricing_version(profile)
    rules = version.rules_json if version and version.rules_json else None
    rules_hash = None
    if rules is not None:
        rules_hash = str(abs(hash(json.dumps(rules, sort_keys=True))))
    return {
        "active_pricing_profile_id": profile.id,
        "active_pricing_profile_version_id": version.id if version else None,
        "active_pricing_rules_json_hash": rules_hash,
    }


def _validate_service_area_value(value: object) -> str:
    """Valide le format JSON V1 tout en gardant la compatibilité legacy."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("service_area doit être une chaîne.")

    text = value.strip()
    if not text:
        return ""
    if len(text) > SERVICE_AREA_MAX_LENGTH:
        raise ValueError(f"service_area dépasse la taille maximale ({SERVICE_AREA_MAX_LENGTH}).")

    # Legacy: accepter en lecture/écriture, sans normalisation forcée côté backend.
    try:
        parsed = json.loads(text)
    except Exception:
        return text

    if not isinstance(parsed, dict):
        raise ValueError("service_area JSON invalide: objet attendu.")
    version = int(parsed.get("v", 0))
    if version != SERVICE_AREA_JSON_VERSION:
        raise ValueError("service_area JSON invalide: version non supportée.")

    mode = str(parsed.get("mode", "")).strip().lower()
    if mode not in SERVICE_AREA_MODE_SET:
        raise ValueError("service_area JSON invalide: mode inconnu.")

    tokens = parsed.get("tokens")
    if not isinstance(tokens, list) or len(tokens) == 0:
        raise ValueError("service_area JSON invalide: tokens requis.")
    norm_tokens = [str(token).strip() for token in tokens if str(token).strip()]
    if len(norm_tokens) == 0:
        raise ValueError("service_area JSON invalide: tokens vides.")
    if any(not SERVICE_AREA_TOKEN_PATTERN.match(token) for token in norm_tokens):
        raise ValueError("service_area JSON invalide: token non canonique.")
    if mode in {"canton", "district"} and len(norm_tokens) != 1:
        raise ValueError("service_area JSON invalide: canton/district accepte un seul token.")
    if mode == "commune" and any(not token.startswith("commune:") for token in norm_tokens):
        raise ValueError("service_area JSON invalide: mode commune incompatible avec tokens.")
    if mode == "canton" and any(not token.startswith("canton:") for token in norm_tokens):
        raise ValueError("service_area JSON invalide: mode canton incompatible avec tokens.")
    if mode == "district" and any(not token.startswith("district:") for token in norm_tokens):
        raise ValueError("service_area JSON invalide: mode district incompatible avec tokens.")

    canonical = {"v": SERVICE_AREA_JSON_VERSION, "mode": mode, "tokens": norm_tokens}
    return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))

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
        "pricing_summary": fields.Raw(
            description="Résumé du pricing actif (flat/zone/distance)", allow_null=True
        ),
        "rules_json": fields.Raw(
            description="Règles de tarification complètes (pricing profile version)", allow_null=True
        ),
        "active_pricing_profile_id": fields.Integer(
            description="ID du pricing profile actif", allow_null=True
        ),
        "active_pricing_profile_version_id": fields.Integer(
            description="ID de version active du pricing profile", allow_null=True
        ),
        "active_pricing_rules_json_hash": fields.String(
            description="Hash des règles actives (observabilité/cache)", allow_null=True
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

geo_unit_model = settings_ns.model(
    "GeoUnitCompact",
    {
        "id": fields.Integer,
        "type": fields.String,
        "code": fields.String,
        "name": fields.String,
    },
)

service_area_item_model = settings_ns.model(
    "ServiceAreaItem",
    {
        "id": fields.Integer,
        "geo_unit": fields.Nested(geo_unit_model),
        "coverage_mode": fields.String,
        "weight": fields.Integer,
        "is_active": fields.Boolean,
        "created_at": fields.String,
    },
)

service_area_create_model = settings_ns.model(
    "ServiceAreaCreate",
    {
        "geo_unit_id": fields.Integer(required=True),
        "coverage_mode": fields.String(
            required=True, enum=["A_STRICT", "B_PICKUP_ONLY", "C_INTRA_ONLY", "D_NATIONAL"]
        ),
        "weight": fields.Integer(required=False),
        "is_active": fields.Boolean(required=False),
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

pricing_zone_settings_model = settings_ns.model(
    "PricingZoneSettings",
    {
        "model": fields.String(required=False, description="zone_matrix"),
        "zones": fields.Raw(required=False, description="Liste des zones tarifaires"),
        "matrix": fields.Raw(required=False, description="Matrice from->to"),
        "matrix_symmetry": fields.Boolean(required=False),
        "default_same_zone_price": fields.Float(required=False),
        "extras": fields.Raw(required=False),
        "minimum": fields.Float(required=False),
    },
)

# ==================== Routes ====================


def _get_or_create_active_pricing_profile(company_id: int) -> PricingProfile:
    profile = (
        PricingProfile.query.filter_by(company_id=company_id, is_active=True)
        .order_by(PricingProfile.created_at.desc())
        .first()
    )
    if profile:
        return profile

    profile = PricingProfile()
    profile.company_id = company_id
    profile.name = "Zone Matrix V1"
    profile.is_active = True
    profile.model_type = PricingModelType.ZONE
    profile.currency = "CHF"
    db.session.add(profile)
    db.session.flush()
    return profile


def _get_current_pricing_version(profile: PricingProfile) -> PricingProfileVersion | None:
    if profile.current_version:
        return profile.current_version
    if profile.versions:
        return sorted(profile.versions, key=lambda v: int(v.version), reverse=True)[0]
    return None


def _build_default_zone_matrix_rules(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    base = dict(existing or {})
    base.setdefault("model", "zone_matrix")
    base.setdefault("zones", [])
    base.setdefault("matrix", {})
    base.setdefault("matrix_symmetry", False)
    base.setdefault("default_same_zone_price", None)
    base.setdefault("extras", [])
    base.setdefault("minimum", 0)
    return base


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
                company.service_area = _validate_service_area_value(data["service_area"])
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
        except ValueError as exc:
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(str(exc), logger_instance=logger)
        except Exception as e:
            db.session.rollback()
            logger.error("[Settings] Error updating operational settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


def _serialize_service_area(item: ServiceArea) -> dict[str, Any]:
    geo_unit = item.geo_unit
    return {
        "id": item.id,
        "geo_unit": {
            "id": geo_unit.id if geo_unit else None,
            "type": geo_unit.type.value if geo_unit and geo_unit.type else None,
            "code": geo_unit.code if geo_unit else None,
            "name": geo_unit.name if geo_unit else None,
        },
        "coverage_mode": item.coverage_mode.value if item.coverage_mode else None,
        "weight": int(item.weight or 0),
        "is_active": bool(item.is_active),
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }


@settings_ns.route("/service-areas")
class ServiceAreasResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err or not company:
            return (err or {"error": "Company not found"}), (code or HTTPStatus.NOT_FOUND)
        items = (
            ServiceArea.query.filter_by(company_id=company.id)
            .order_by(ServiceArea.id.desc())
            .all()
        )
        return {"items": [_serialize_service_area(item) for item in items]}, HTTPStatus.OK

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(service_area_create_model)
    def post(self):
        company, err, code = get_company_from_token()
        if err or not company:
            return (err or {"error": "Company not found"}), (code or HTTPStatus.NOT_FOUND)
        payload = request.get_json(silent=True) or {}
        try:
            data = validate_request(ServiceAreaCreateSchema(), payload)
        except Exception as exc:
            return APIErrorHandler.handle_validation_error(str(exc), logger_instance=logger)

        geo_unit = GeoUnit.query.filter_by(id=data["geo_unit_id"]).first()
        if not geo_unit:
            return {"error": "GeoUnit introuvable"}, HTTPStatus.NOT_FOUND

        existing = ServiceArea.query.filter_by(
            company_id=company.id,
            geo_unit_id=data["geo_unit_id"],
            coverage_mode=data["coverage_mode"],
        ).first()
        if existing:
            return {
                "error": "Une zone identique existe déjà pour cette entreprise."
            }, HTTPStatus.CONFLICT

        item = ServiceArea()
        item.company_id = company.id
        item.geo_unit_id = data["geo_unit_id"]
        item.coverage_mode = data["coverage_mode"]
        item.weight = int(data.get("weight", 0))
        item.is_active = bool(data.get("is_active", True))
        db.session.add(item)
        db.session.commit()
        return _serialize_service_area(item), HTTPStatus.CREATED


@settings_ns.route("/service-areas/<int:service_area_id>")
class ServiceAreaByIdResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(service_area_create_model)
    def put(self, service_area_id: int):
        company, err, code = get_company_from_token()
        if err or not company:
            return (err or {"error": "Company not found"}), (code or HTTPStatus.NOT_FOUND)
        item = ServiceArea.query.filter_by(id=service_area_id, company_id=company.id).first()
        if not item:
            return {"error": "ServiceArea introuvable"}, HTTPStatus.NOT_FOUND
        payload = request.get_json(silent=True) or {}
        try:
            data = validate_request(ServiceAreaUpdateSchema(), payload)
        except Exception as exc:
            return APIErrorHandler.handle_validation_error(str(exc), logger_instance=logger)

        if "coverage_mode" in data:
            item.coverage_mode = data["coverage_mode"]
        if "weight" in data:
            item.weight = int(data["weight"])
        if "is_active" in data:
            item.is_active = bool(data["is_active"])
        db.session.commit()
        return _serialize_service_area(item), HTTPStatus.OK

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, service_area_id: int):
        company, err, code = get_company_from_token()
        if err or not company:
            return (err or {"error": "Company not found"}), (code or HTTPStatus.NOT_FOUND)
        item = ServiceArea.query.filter_by(id=service_area_id, company_id=company.id).first()
        if not item:
            return {"error": "ServiceArea introuvable"}, HTTPStatus.NOT_FOUND
        db.session.delete(item)
        db.session.commit()
        return {"success": True, "deleted_id": service_area_id}, HTTPStatus.OK


@settings_ns.route("/pricing-zones")
class PricingZonesSettingsResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err or not company:
            return (err or {"error": "Company not found"}), (code or HTTPStatus.NOT_FOUND)

        profile = _get_or_create_active_pricing_profile(company.id)
        version = _get_current_pricing_version(profile)
        rules = _build_default_zone_matrix_rules(version.rules_json if version else None)
        if version and version.rules_json:
            rules = dict(version.rules_json)
        rules = _build_default_zone_matrix_rules(rules)

        return {
            "success": True,
            "profile_id": profile.id,
            "version_id": version.id if version else None,
            "data": rules,
        }, HTTPStatus.OK

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(pricing_zone_settings_model, validate=False)
    def put(self):
        company, err, code = get_company_from_token()
        if err or not company:
            return (err or {"error": "Company not found"}), (code or HTTPStatus.NOT_FOUND)

        payload = request.get_json(silent=True) or {}
        try:
            data = validate_request(PricingZoneMatrixSettingsSchema(), payload)
            normalized_rules = validate_zone_matrix_rules(data)
        except Exception as exc:
            return APIErrorHandler.handle_validation_error(str(exc), logger_instance=logger)

        profile = _get_or_create_active_pricing_profile(company.id)
        current_version = _get_current_pricing_version(profile)
        existing_rules = current_version.rules_json if current_version and current_version.rules_json else {}
        merged_rules = dict(existing_rules or {})
        merged_rules.update(normalized_rules)

        next_version = int(current_version.version) + 1 if current_version else 1
        created_by = getattr(company, "user_id", None)
        version = PricingProfileVersion()
        version.pricing_profile_id = profile.id
        version.version = next_version
        version.rules_json = merged_rules
        version.created_by_user_id = created_by if isinstance(created_by, int) else None
        db.session.add(version)
        db.session.flush()

        profile.current_version_id = version.id
        profile.model_type = PricingModelType.ZONE
        db.session.add(profile)
        db.session.commit()

        return {
            "success": True,
            "profile_id": profile.id,
            "version_id": version.id,
            "data": merged_rules,
        }, HTTPStatus.OK


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
            result["pricing_summary"] = _build_pricing_summary(company.id)
            result["rules_json"] = _get_company_pricing_rules(company.id)
            result.update(_get_active_pricing_meta(company.id))

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
            logger.exception("[Settings] Error fetching billing settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(billing_settings_model, validate=False)
    def put(self):  # noqa: PLR0911
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

            rules_payload = data.get("rules_json")
            if rules_payload is not None:
                if not isinstance(rules_payload, dict):
                    return {"success": False, "error": "rules_json doit être un objet JSON"}, 400
                try:
                    normalized_rules = validate_company_pricing_rules(rules_payload)
                except Exception as exc:
                    return APIErrorHandler.handle_validation_error(str(exc), logger_instance=logger)

                profile = _get_or_create_active_pricing_profile(company.id)
                current_version = _get_current_pricing_version(profile)
                next_version = int(current_version.version) + 1 if current_version else 1
                model_value = str(normalized_rules.get("model") or "").lower()
                if model_value == "flat":
                    profile.model_type = PricingModelType.FLAT
                elif model_value in {"zone_count", "zone_matrix"}:
                    profile.model_type = PricingModelType.ZONE
                elif model_value == "distance":
                    profile.model_type = PricingModelType.DISTANCE
                elif model_value == "hybrid_stack":
                    profile.model_type = PricingModelType.HYBRID

                version = PricingProfileVersion()
                version.pricing_profile_id = profile.id
                version.version = next_version
                version.rules_json = normalized_rules
                version.created_by_user_id = getattr(company, "user_id", None)
                db.session.add(version)
                db.session.flush()
                profile.current_version_id = version.id
                db.session.add(profile)

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

            # ✅ Aligner company.iban + CompanyBillingProfile (QR) sur les paramètres billing
            if "iban" in data or "qr_iban" in data:
                from services.billing.banking_identifiers_sync import sync_banking_identifiers

                sync_banking_identifiers(company, source="billing_settings")

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
            result_dict["pricing_summary"] = _build_pricing_summary(company.id)
            result_dict["rules_json"] = _get_company_pricing_rules(company.id)
            result_dict.update(_get_active_pricing_meta(company.id))

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
