from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from flask import request
from flask_jwt_extended import get_jwt, get_jwt_identity, verify_jwt_in_request
from flask_restx import Namespace, Resource, fields
from marshmallow import ValidationError

from ext import app_logger, db, limiter, redis_client
from models import ContactRequest, DemoRequest, User
from schemas.contact_schemas import ContactRequestBaseOnlySchema, schema_for_category
from schemas.validation_utils import handle_validation_error, validate_request
from services.contact.dedupe import (
    compute_dedupe_hash,
    current_window_bucket,
    find_recent_duplicate,
    normalize_message,
)
from services.contact.dispatcher import get_destination_email, send_contact_notification
from services.contact.scoring import compute_priority
from services.contact.spam_guard import (
    hit_rate_limit,
    in_cooldown,
    is_silent_spam,
    minimal_spam_payload,
)
from services.demo.dispatcher import get_demo_destination_email
from services.demo.scoring import compute_demo_score
from shared.input_sanitizer import (
    sanitize_email,
    sanitize_string,
    strip_control_characters,
)

contact_ns = Namespace("contact", description="Demandes de contact publiques")
logger = logging.getLogger(__name__)

_DEMO_ORG_TYPE_MAP = {
    "transport": "transport_company",
    "transport_company": "transport_company",
    "institution": "institution",
    "ems": "ems",
    "clinic": "clinic",
    "hospital": "hospital",
    "curatorship": "curatorship",
}
_DEMO_USE_CASE_MAP = {
    "transport_company": "planning_dispatch",
    "institution": "reporting",
    "ems": "reporting",
    "clinic": "reporting",
    "hospital": "reporting",
    "curatorship": "multi_company_coordination",
}
_DEMO_TIMING_ALLOWED = {"immediate", "one_three_months", "three_plus_months", "exploration"}
_DEMO_SLOT_ALLOWED = {"this_week", "next_week", "to_define"}
_DEMO_VOLUME_ALLOWED = {"1_5", "5_20", "20_100", "100_plus"}


def _normalize_demo_org_type(raw_value: str | None) -> str:
    value = str(raw_value or "").strip().lower()
    return _DEMO_ORG_TYPE_MAP.get(value, "other")


def _normalize_demo_use_case(org_type: str) -> str:
    return _DEMO_USE_CASE_MAP.get(org_type, "other")


def _normalize_demo_timing(raw_value: str | None) -> str:
    value = str(raw_value or "").strip().lower()
    if value in _DEMO_TIMING_ALLOWED:
        return value
    return "exploration"


def _normalize_demo_slot(raw_value: str | None) -> str:
    value = str(raw_value or "").strip().lower()
    if value in _DEMO_SLOT_ALLOWED:
        return value
    if value == "to_schedule":
        return "to_define"
    return "to_define"


def _normalize_demo_volume(raw_value: str | None) -> str | None:
    value = str(raw_value or "").strip().lower()
    if not value:
        return None
    if value in _DEMO_VOLUME_ALLOWED:
        return value
    return None


def _mirror_demo_contact_to_demo_request(
    contact_row: ContactRequest, sanitized_data: dict[str, Any]
) -> None:
    source = f"contact_request:{contact_row.id}"
    existing = DemoRequest.query.filter(DemoRequest.source == source).first()
    if existing:
        return

    organization_type = _normalize_demo_org_type(
        cast(str | None, sanitized_data.get("organization_type"))
    )
    use_case = _normalize_demo_use_case(organization_type)
    timing = _normalize_demo_timing(cast(str | None, sanitized_data.get("timing")))
    preferred_slot = _normalize_demo_slot(cast(str | None, sanitized_data.get("preferred_slot")))
    volume_range = _normalize_demo_volume(cast(str | None, sanitized_data.get("volume_range")))
    preferred_period = "flexible"
    integration_required = "evaluate"

    score_payload: dict[str, str | None] = {
        "name": contact_row.name or "",
        "email": contact_row.email or "",
        "phone": contact_row.phone,
        "organization": contact_row.organization or "",
        "organization_type": organization_type,
        "use_case": use_case,
        "volume_range": volume_range,
        "integration_required": integration_required,
        "integration_system": None,
        "timing": timing,
        "preferred_slot": preferred_slot,
        "preferred_period": preferred_period,
        "comment": contact_row.message or "",
    }
    score = compute_demo_score(score_payload)

    demo_values: dict[str, Any] = {
        "name": contact_row.name or "",
        "email": contact_row.email or "",
        "phone": contact_row.phone,
        "organization": contact_row.organization or "",
        "organization_type": organization_type,
        "use_case": use_case,
        "volume_range": volume_range,
        "integration_required": integration_required,
        "integration_system": None,
        "timing": timing,
        "preferred_slot": preferred_slot,
        "preferred_period": preferred_period,
        "comment": contact_row.message,
        "score": score,
        "status": "new",
        "trace_id": f"{contact_row.trace_id}_demo",
        "source": source,
        "ip_address": None,
        "user_agent": contact_row.user_agent,
        "assigned_channel": get_demo_destination_email(),
        "email_delivery_status": "linked_contact",
    }
    demo_request = DemoRequest()
    for key, value in demo_values.items():
        setattr(demo_request, key, value)
    db.session.add(demo_request)
    db.session.commit()

contact_request_model = contact_ns.model(
    "ContactRequest",
    {
        "category": fields.String(required=True, example="support"),
        "name": fields.String(required=True, example="Jean Dupont"),
        "email": fields.String(required=True, example="jean.dupont@example.com"),
        "organization": fields.String(required=False, example="Clinique Demo"),
        "phone": fields.String(required=False, example="+41221234567"),
        "subject_detail": fields.String(required=False),
        "message": fields.String(required=True),
        "privacy_consent": fields.Boolean(required=True),
        "website": fields.String(required=False),
        "client_request_id": fields.String(required=False),
    },
)


def _get_ip_address() -> str | None:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr


def _extract_optional_auth_context() -> dict[str, str | int | None]:
    context: dict[str, str | int | None] = {
        "user_id": None,
        "user_public_id": None,
        "user_role": None,
        "company_id": None,
        "institution_id": None,
    }
    try:
        verify_jwt_in_request(optional=True)
        identity = get_jwt_identity()
        claims = get_jwt() or {}
        if identity:
            user = User.query.filter_by(public_id=str(identity)).first()
            context["user_id"] = user.id if user else None
            context["user_public_id"] = str(identity)
        context["user_role"] = claims.get("role")
        context["company_id"] = claims.get("company_id")
        context["institution_id"] = claims.get("institution_id")
    except Exception:
        # Route publique: l'auth reste optionnelle.
        return context
    return context


def _generate_trace_id(prefix: str = "ct") -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    token = "".join(secrets.choice(alphabet) for _ in range(12))
    return f"{prefix}_{token}"


def _hash_ip(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sanitize_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": data.get("category"),
        "name": strip_control_characters(
            sanitize_string(data.get("name"), max_length=120, strip_html=True)
        ),
        "email": sanitize_email(data.get("email")),
        "phone": strip_control_characters(
            sanitize_string(data.get("phone"), max_length=32, strip_html=True)
        ),
        "organization": strip_control_characters(
            sanitize_string(
                data.get("organization") or data.get("company"), max_length=180, strip_html=True
            )
        ),
        "message": strip_control_characters(
            sanitize_string(data.get("message"), max_length=4000, strip_html=True)
        ),
        "subject_detail": strip_control_characters(
            sanitize_string(data.get("subject_detail"), max_length=64, strip_html=True)
        ),
        "reference": strip_control_characters(
            sanitize_string(data.get("reference"), max_length=120, strip_html=True)
        ),
        "urgency": data.get("urgency"),
        "organization_type": data.get("organization_type"),
        "sites_count": strip_control_characters(
            sanitize_string(data.get("sites_count"), max_length=32, strip_html=True)
        ),
        "integration_required": data.get("integration_required"),
        "integration_system": strip_control_characters(
            sanitize_string(data.get("integration_system"), max_length=120, strip_html=True)
        ),
        "fleet_size_range": strip_control_characters(
            sanitize_string(data.get("fleet_size_range"), max_length=64, strip_html=True)
        ),
        "service_area": strip_control_characters(
            sanitize_string(data.get("service_area"), max_length=160, strip_html=True)
        ),
        "timing": data.get("timing"),
        "preferred_slot": data.get("preferred_slot"),
        "volume_range": data.get("volume_range"),
        "situation": strip_control_characters(
            sanitize_string(data.get("situation"), max_length=220, strip_html=True)
        ),
        "website": strip_control_characters(
            sanitize_string(data.get("website"), max_length=256, strip_html=True)
        ),
        "client_request_id": strip_control_characters(
            sanitize_string(data.get("client_request_id"), max_length=64, strip_html=True)
        ),
    }


def _email_send_lock_key(dedupe_hash: str | None) -> str | None:
    if not dedupe_hash:
        return None
    return f"contact:notify:{dedupe_hash}"


def _acquire_email_send_lock(dedupe_hash: str | None, trace_id: str) -> bool:
    lock_key = _email_send_lock_key(dedupe_hash)
    if not lock_key or not redis_client:
        return True
    return bool(redis_client.set(lock_key, trace_id, ex=120, nx=True))


def _has_sent_for_hash(dedupe_hash: str, current_id: int, within_minutes: int = 5) -> bool:
    window_start = datetime.now(UTC) - timedelta(minutes=within_minutes)
    existing = (
        ContactRequest.query.filter(ContactRequest.dedupe_hash == dedupe_hash)
        .filter(ContactRequest.id != current_id)
        .filter(ContactRequest.email_delivery_status.in_(["sending", "sent"]))
        .filter(ContactRequest.created_at >= window_start)
        .first()
    )
    return existing is not None


def _mark_email_sending(contact_request_id: int) -> int:
    return (
        ContactRequest.query.filter(ContactRequest.id == contact_request_id)
        .filter(ContactRequest.email_delivery_status == "pending")
        .update({"email_delivery_status": "sending"}, synchronize_session=False)
    )


def _build_contact_request(values: dict[str, Any]) -> ContactRequest:
    """Instantiate ContactRequest without kwargs for stricter type-checkers."""
    row = ContactRequest()
    for key, value in values.items():
        setattr(row, key, value)
    return row


@contact_ns.route("/requests")
class ContactRequests(Resource):
    @contact_ns.expect(contact_request_model)
    @contact_ns.response(200, "Demande acceptee")
    @contact_ns.response(400, "Requete invalide")
    @contact_ns.response(429, "Trop de requetes")
    @limiter.limit("30 per day")
    def post(self):  # noqa: PLR0911
        payload = request.get_json(silent=True) or {}

        try:
            base_data = validate_request(ContactRequestBaseOnlySchema(), payload)
            schema = schema_for_category(base_data["category"])
            data = cast(dict[str, Any], schema.load(payload))
        except ValidationError as error:
            return handle_validation_error(error)

        ip_address = _get_ip_address()
        ip_hash = _hash_ip(ip_address)
        category = str(data.get("category"))
        if ip_hash and (hit_rate_limit(ip_hash, category) or in_cooldown(ip_hash, category)):
            return {"error": "rate_limited", "message": "Trop de requetes."}, 429

        sanitized_data = _sanitize_payload(data)
        if not sanitized_data["email"]:
            return {"error": "validation_error", "message": "Adresse email invalide."}, 400

        trace_id = _generate_trace_id()
        auth_context = _extract_optional_auth_context()

        message_normalized = normalize_message(sanitized_data["message"] or "")
        dedupe_hash = compute_dedupe_hash(
            sanitized_data["email"] or "",
            category,
            message_normalized,
        )
        dedupe_window_bucket = current_window_bucket()

        if is_silent_spam(sanitized_data):
            spam_trace = _generate_trace_id()
            request_row = _build_contact_request(
                {
                    "category": category,
                    "name": sanitized_data["name"] or "",
                    "email": sanitized_data["email"] or "",
                    "phone": None,
                    "organization": None,
                    "message": None,
                    "message_normalized": None,
                    "dedupe_hash": dedupe_hash,
                    "dedupe_window_bucket": dedupe_window_bucket,
                    "client_request_id": sanitized_data["client_request_id"],
                    "payload_json": minimal_spam_payload(sanitized_data),
                    "ip_hash": ip_hash,
                    "user_agent": (request.user_agent.string or "")[:512],
                    "user_id": auth_context["user_id"],
                    "user_public_id": auth_context["user_public_id"],
                    "user_role": auth_context["user_role"],
                    "company_id": auth_context["company_id"],
                    "institution_id": auth_context["institution_id"],
                    "status": "spam",
                    "priority": "standard",
                    "assigned_channel": get_destination_email(category),
                    "email_delivery_status": "suppressed_spam",
                    "trace_id": spam_trace,
                }
            )
            db.session.add(request_row)
            db.session.commit()
            return {"ok": True, "trace_id": spam_trace}, 200

        duplicate = find_recent_duplicate(dedupe_hash, window_minutes=5)
        if duplicate:
            return {"ok": True, "trace_id": duplicate.trace_id}, 200

        priority = compute_priority(sanitized_data)
        destination_email = get_destination_email(category)
        payload_json = {
            key: value
            for key, value in sanitized_data.items()
            if key
            not in {
                "category",
                "name",
                "email",
                "phone",
                "organization",
                "message",
                "website",
            }
            and value not in (None, "")
        }

        contact_request = _build_contact_request(
            {
                "name": sanitized_data["name"] or "",
                "email": sanitized_data["email"] or "",
                "organization": sanitized_data["organization"],
                "phone": sanitized_data["phone"],
                "category": category,
                "message": sanitized_data["message"] or "",
                "message_normalized": message_normalized,
                "dedupe_hash": dedupe_hash,
                "dedupe_window_bucket": dedupe_window_bucket,
                "client_request_id": sanitized_data["client_request_id"],
                "payload_json": payload_json,
                "ip_hash": ip_hash,
                "user_agent": (request.user_agent.string or "")[:512],
                "user_id": auth_context["user_id"],
                "user_public_id": auth_context["user_public_id"],
                "user_role": auth_context["user_role"],
                "company_id": auth_context["company_id"],
                "institution_id": auth_context["institution_id"],
                "status": "new",
                "priority": priority,
                "assigned_channel": destination_email,
                "email_delivery_status": "pending",
                "trace_id": trace_id,
            }
        )

        try:
            db.session.add(contact_request)
            db.session.commit()

            if _has_sent_for_hash(dedupe_hash, contact_request.id) or not _acquire_email_send_lock(
                dedupe_hash, trace_id
            ):
                contact_request.email_delivery_status = "suppressed_duplicate"
                contact_request.status = "triaged"
                db.session.commit()
                return {"ok": True, "trace_id": trace_id}, 200

            updated = _mark_email_sending(contact_request.id)
            if updated == 0:
                return {"ok": True, "trace_id": trace_id}, 200

            email_sent = False
            try:
                email_result = send_contact_notification(
                    {
                        **sanitized_data,
                        **auth_context,
                        "priority": priority,
                        "category": category,
                        "payload_json": payload_json,
                        "trace_id": trace_id,
                    }
                )
                email_sent = bool(email_result.get("ok"))
            except Exception:
                logger.exception("contact_email_send_failed trace_id=%s", trace_id)

            contact_request.email_delivery_status = "sent" if email_sent else "failed"
            contact_request.status = "triaged" if email_sent else "new"
            db.session.commit()
        except Exception:
            db.session.rollback()
            logger.exception("contact_request_failed trace_id=%s", trace_id)
            return {
                "error": "internal_error",
                "message": "Une erreur est survenue. Merci de reessayer plus tard.",
                "trace_id": trace_id,
            }, 500

        app_logger.info(
            "contact_request_processed",
            extra={
                "trace_id": trace_id,
                "category": category,
                "priority": priority,
                "assigned_channel": destination_email,
                "email_status": contact_request.email_delivery_status,
                "ip_hash": ip_hash,
                "message_len": len(sanitized_data.get("message") or ""),
            },
        )

        if category == "demo":
            try:
                _mirror_demo_contact_to_demo_request(contact_request, sanitized_data)
            except Exception:
                db.session.rollback()
                logger.exception("contact_demo_mirror_failed trace_id=%s", trace_id)

        return {"ok": True, "trace_id": trace_id}, 200
