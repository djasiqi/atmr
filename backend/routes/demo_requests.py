from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from flask import current_app, make_response, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Namespace, Resource, fields
from marshmallow import ValidationError

from ext import db, limiter, role_required
from middleware.trace_id import get_trace_id
from models import DemoAccess, DemoRequest, User, UserRole
from models.enums import InstitutionRole
from schemas.demo_request_schemas import DemoRequestSchema
from schemas.validation_utils import handle_validation_error, validate_request
from services.demo.access_service import (
    DemoAccessError,
    consume_magic_link,
    provision_demo_access,
    resend_demo_access,
    revoke_demo_access,
)
from services.demo.dispatcher import (
    get_demo_destination_email,
    send_demo_acknowledgement,
    send_demo_notification,
    send_demo_rejection_email,
)
from services.demo.scoring import compute_demo_score, derive_demo_priority
from shared.input_sanitizer import (
    sanitize_email,
    sanitize_string,
    strip_control_characters,
)

demo_requests_ns = Namespace("demo_requests", description="Demandes de demonstration")
admin_demo_requests_ns = Namespace(
    "admin_demo_requests", description="Actions admin sur demandes de demo"
)
admin_demo_accesses_ns = Namespace(
    "admin_demo_accesses", description="Actions admin sur acces de demo"
)
demo_access_ns = Namespace("demo_access", description="Acces demo public")
logger = logging.getLogger(__name__)
MIN_FORM_SUBMIT_MS = 1200
DEMO_PASSWORD_MIN_LENGTH = 8

demo_request_model = demo_requests_ns.model(
    "DemoRequest",
    {
        "name": fields.String(required=True),
        "email": fields.String(required=True),
        "phone": fields.String(required=False),
        "organization": fields.String(required=True),
        "organization_type": fields.String(required=True),
        "use_case": fields.String(required=True),
        "volume_range": fields.String(required=False),
        "integration_required": fields.String(required=True),
        "integration_system": fields.String(required=False),
        "timing": fields.String(required=True),
        "preferred_slot": fields.String(required=True),
        "preferred_period": fields.String(required=True),
        "comment": fields.String(required=False),
        "privacy_consent": fields.Boolean(required=True),
        "honeypot": fields.String(required=False),
        "form_started_at_ms": fields.Integer(required=False),
        "acknowledgement_already_sent": fields.Boolean(required=False),
        "source": fields.String(required=False),
    },
)

magic_link_consume_model = demo_access_ns.model(
    "DemoAccessConsume",
    {
        "token": fields.String(required=True),
    },
)

demo_analytics_model = demo_access_ns.model(
    "DemoAnalyticsEvent",
    {
        "event": fields.String(required=True),
        "payload": fields.Raw(required=False),
    },
)

demo_set_password_model = demo_access_ns.model(
    "DemoAccessSetPassword",
    {
        "new_password": fields.String(required=True, min_length=8),
    },
)

demo_status_update_model = admin_demo_requests_ns.model(
    "DemoRequestStatusUpdate",
    {
        "status": fields.String(
            required=True,
            description="Nouveau statut (new|contacted|qualified|rejected)",
        ),
    },
)

provision_profile_model = admin_demo_requests_ns.model(
    "DemoProvisionProfile",
    {
        "organization_name": fields.String(required=False),
        "organization_type": fields.String(required=False),
        "organization_address": fields.String(required=False),
        "organization_contact_phone": fields.String(required=False),
        "organization_contact_email": fields.String(required=False),
        "workspace_display_name": fields.String(required=False),
        "demo_login_email": fields.String(required=False),
        "user_first_name": fields.String(required=False),
        "user_last_name": fields.String(required=False),
        "user_phone": fields.String(required=False),
        "user_role": fields.String(required=False),
        "provision_template": fields.String(required=False),
        "demo_persona": fields.String(required=False),
        "guide_variant": fields.String(required=False),
        "seed_context": fields.Raw(required=False),
        "internal_admin_notes": fields.String(required=False),
        "visible_demo_notes": fields.String(required=False),
        "workspace_seed_notes": fields.String(required=False),
    },
)

admin_provision_request_model = admin_demo_requests_ns.model(
    "AdminProvisionDemoAccessRequest",
    {
        "provision_profile": fields.Nested(provision_profile_model, required=False),
    },
)


def _get_ip_address() -> str | None:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr


def _build_demo_request(values: dict[str, Any]) -> DemoRequest:
    row = DemoRequest()
    for key, value in values.items():
        setattr(row, key, value)
    return row


def _response_from_demo_access_error(error: DemoAccessError):
    return {
        "ok": False,
        "code": error.code,
        "message": error.message,
        "trace_id": get_trace_id(),
    }, error.status_code


def _resolve_demo_journey(org_type_raw: str | None) -> str:
    org_type = str(org_type_raw or "").strip().lower()
    if org_type in {"transport_company", "transport"}:
        return "transport"
    if org_type in {"institution", "ems", "clinic", "hospital", "curatorship"}:
        return "institution"
    return "generic"


@demo_requests_ns.route("")
class DemoRequests(Resource):
    @demo_requests_ns.expect(demo_request_model, validate=False)
    @demo_requests_ns.response(201, "Demande enregistree")
    @demo_requests_ns.response(400, "Requete invalide")
    @demo_requests_ns.response(429, "Trop de requetes")
    @limiter.limit("30 per day")
    def post(self):
        payload = request.get_json(silent=True) or {}
        try:
            data = validate_request(DemoRequestSchema(), payload)
        except ValidationError as error:
            return handle_validation_error(error)

        if not data.get("privacy_consent"):
            return {
                "error": "validation_error",
                "message": "Le consentement est requis pour envoyer la demande.",
            }, 400

        honeypot = (data.get("honeypot") or "").strip()
        if honeypot:
            return {
                "ok": True,
                "message": "Merci. Votre demande de demonstration a bien ete transmise.",
            }, 201

        form_started_at_ms = data.get("form_started_at_ms")
        if (
            isinstance(form_started_at_ms, int)
            and int(time.time() * 1000) - form_started_at_ms < MIN_FORM_SUBMIT_MS
        ):
            return {
                "error": "validation_error",
                "message": "Le formulaire a ete soumis trop rapidement.",
            }, 400

        email = sanitize_email(data["email"])
        if not email:
            return {
                "error": "validation_error",
                "message": "Adresse email invalide.",
            }, 400

        clean = {
            "name": strip_control_characters(
                sanitize_string(data["name"], max_length=120)
            )
            or "",
            "email": email,
            "phone": strip_control_characters(
                sanitize_string(data.get("phone"), max_length=32)
            ),
            "organization": strip_control_characters(
                sanitize_string(data["organization"], max_length=180)
            )
            or "",
            "organization_type": data["organization_type"],
            "use_case": data["use_case"],
            "volume_range": data.get("volume_range"),
            "integration_required": data["integration_required"],
            "integration_system": strip_control_characters(
                sanitize_string(data.get("integration_system"), max_length=180)
            ),
            "timing": data["timing"],
            "preferred_slot": data["preferred_slot"],
            "preferred_period": data["preferred_period"],
            "comment": strip_control_characters(
                sanitize_string(data.get("comment"), max_length=3000)
            ),
        }

        if clean["integration_required"] != "yes":
            clean["integration_system"] = None

        score = compute_demo_score(clean)
        priority = derive_demo_priority(score)
        trace_id = uuid.uuid4().hex
        destination = get_demo_destination_email()

        source = (
            str(data.get("source") or "web_demo_request").strip() or "web_demo_request"
        )
        source = source[:64]
        demo_values = {
            **clean,
            "score": score,
            "status": "new",
            "trace_id": trace_id,
            "source": source,
            "ip_address": _get_ip_address(),
            "user_agent": (request.user_agent.string or "")[:512],
            "assigned_channel": destination,
            "email_delivery_status": "pending",
        }
        demo_request = _build_demo_request(demo_values)

        try:
            db.session.add(demo_request)
            db.session.flush()
            email_result = send_demo_notification(
                {**clean, "score": score, "priority": priority, "trace_id": trace_id}
            )
            if not email_result.get("ok"):
                logger.warning(
                    "demo_admin_notification_failed trace_id=%s destination=%s error=%s",
                    trace_id,
                    destination,
                    email_result.get("error", "unknown"),
                )
            if not bool(data.get("acknowledgement_already_sent")):
                send_demo_acknowledgement({**clean, "trace_id": trace_id})
            demo_request.email_delivery_status = (
                "sent" if email_result.get("ok") else "failed"
            )
            db.session.commit()
        except Exception:
            db.session.rollback()
            return {
                "error": "internal_error",
                "message": "Une erreur est survenue. Merci de reessayer plus tard.",
                "trace_id": trace_id,
            }, 500

        return {
            "ok": True,
            "message": "Merci. Un membre de l'equipe LIRIE vous contacte sous 24h ouvrees.",
            "request_id": demo_request.id,
            "trace_id": trace_id,
            "score": score,
            "priority": priority,
        }, 201


@admin_demo_requests_ns.route("/<int:demo_request_id>/provision-access")
class AdminProvisionDemoAccess(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @admin_demo_requests_ns.expect(admin_provision_request_model, validate=False)
    @limiter.limit("60 per hour")  # Verrouillage: éviter abus provision admin
    def post(self, demo_request_id: int):
        actor_id = get_jwt_identity()
        payload = request.get_json(silent=True) or {}
        provision_profile = payload.get("provision_profile") or None
        try:
            result = provision_demo_access(
                demo_request_id=demo_request_id,
                actor_id=int(actor_id)
                if actor_id and str(actor_id).isdigit()
                else None,
                provision_source="manual",
                provisioning_mode="shared_workspace",
                provision_profile=provision_profile,
            )
        except DemoAccessError as error:
            db.session.rollback()
            return _response_from_demo_access_error(error)
        except Exception as exc:
            db.session.rollback()
            logger.exception(
                "provision_demo_access_failed demo_request_id=%s error=%s trace_id=%s",
                demo_request_id,
                exc,
                get_trace_id(),
            )
            return {
                "ok": False,
                "code": "internal_error",
                "message": "Provisionnement demo impossible.",
                "trace_id": get_trace_id(),
            }, 500

        response = {
            "ok": True,
            "demo_request_id": result.demo_request.id,
            "demo_access_id": result.demo_access.id,
            "access_status": result.demo_access.status,
            "demo_expires_at": (
                result.demo_access.demo_expires_at.isoformat()
                if result.demo_access.demo_expires_at
                else None
            ),
            "email_sent": result.email_sent,
            "reused_existing_access": result.reused_existing_access,
            "provision_summary": result.provision_summary,
            "trace_id": get_trace_id(),
        }
        if not result.email_sent:
            response["code"] = "access_provisioned_email_failed"
            response["message"] = "Acces provisionne, mais envoi email en echec."
            response["email_error"] = result.email_error
        return response, 200


@admin_demo_requests_ns.route("/<int:demo_request_id>/status")
class AdminUpdateDemoRequestStatus(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @admin_demo_requests_ns.expect(demo_status_update_model)
    def post(self, demo_request_id: int):
        payload = request.get_json(silent=True) or {}
        status = str(payload.get("status") or "").strip().lower()
        allowed_statuses = {"new", "contacted", "qualified", "rejected"}
        if status not in allowed_statuses:
            return {
                "ok": False,
                "code": "validation_error",
                "message": "Statut invalide.",
            }, 400

        row = db.session.get(DemoRequest, demo_request_id)
        if not row:
            return {
                "ok": False,
                "code": "request_not_found",
                "message": "La demande de demo est introuvable.",
            }, 404

        row.status = status
        email_sent = None
        email_error = None
        if status == "rejected":
            email_result = send_demo_rejection_email(demo_request=row)
            email_sent = bool(email_result.get("ok"))
            if not email_sent:
                email_error = str(email_result.get("error") or "email_error")
        db.session.commit()
        response = {"ok": True, "id": row.id, "status": row.status}
        if email_sent is not None:
            response["email_sent"] = email_sent
            if email_error:
                response["email_error"] = email_error
        return response, 200


@admin_demo_requests_ns.route("")
class AdminDemoRequestsList(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        rows = (
            DemoRequest.query.order_by(DemoRequest.created_at.desc()).limit(100).all()
        )
        output = []
        for row in rows:
            latest_access = (
                DemoAccess.query.filter_by(demo_request_id=row.id)
                .order_by(DemoAccess.created_at.desc())
                .first()
            )
            output.append(
                {
                    **row.serialize,
                    "latest_access": latest_access.serialize if latest_access else None,
                }
            )
        return {"ok": True, "items": output}, 200


@admin_demo_accesses_ns.route("/<int:access_id>/resend")
class AdminResendDemoAccess(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def post(self, access_id: int):
        actor_id = get_jwt_identity()
        try:
            result = resend_demo_access(
                access_id=access_id,
                actor_id=int(actor_id)
                if actor_id and str(actor_id).isdigit()
                else None,
            )
        except DemoAccessError as error:
            db.session.rollback()
            return _response_from_demo_access_error(error)
        except Exception:
            db.session.rollback()
            return {
                "ok": False,
                "code": "internal_error",
                "message": "Renvoi de l'acces demo impossible.",
            }, 500

        response = {
            "ok": True,
            "demo_request_id": result.demo_request.id,
            "demo_access_id": result.demo_access.id,
            "access_status": result.demo_access.status,
            "demo_expires_at": (
                result.demo_access.demo_expires_at.isoformat()
                if result.demo_access.demo_expires_at
                else None
            ),
            "email_sent": result.email_sent,
        }
        if not result.email_sent:
            response["code"] = "access_provisioned_email_failed"
            response["message"] = "Renvoi effectue, mais email en echec."
            response["email_error"] = result.email_error
        return response, 200


@admin_demo_accesses_ns.route("/<int:access_id>/revoke")
class AdminRevokeDemoAccess(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def post(self, access_id: int):
        actor_id = get_jwt_identity()
        try:
            access = revoke_demo_access(
                access_id=access_id,
                actor_id=int(actor_id)
                if actor_id and str(actor_id).isdigit()
                else None,
            )
        except DemoAccessError as error:
            db.session.rollback()
            return _response_from_demo_access_error(error)
        except Exception:
            db.session.rollback()
            return {
                "ok": False,
                "code": "internal_error",
                "message": "Revocation de l'acces demo impossible.",
            }, 500
        return {
            "ok": True,
            "demo_access_id": access.id,
            "status": access.status,
        }, 200


@demo_access_ns.route("/consume-magic-link")
class ConsumeDemoMagicLink(Resource):
    @demo_access_ns.expect(magic_link_consume_model)
    @limiter.limit("30 per minute")  # Protection brute-force sur token invalide
    def post(self):
        payload = request.get_json(silent=True) or {}
        token = payload.get("token")
        try:
            result = consume_magic_link(token or "")
        except DemoAccessError as error:
            db.session.rollback()
            return _response_from_demo_access_error(error)
        except Exception:
            db.session.rollback()
            return {
                "ok": False,
                "code": "internal_error",
                "message": "Consommation du lien magique impossible.",
                "trace_id": get_trace_id(),
            }, 500
        access = db.session.get(DemoAccess, result.get("demo_access_id"))
        demo_user = access.demo_user if access else None
        if not demo_user:
            return {
                "ok": False,
                "code": "internal_error",
                "message": "Compte demo introuvable.",
                "trace_id": get_trace_id(),
            }, 500

        role = (
            demo_user.role.value
            if hasattr(demo_user.role, "value")
            else str(demo_user.role or "client")
        )
        demo_company = demo_user.company
        demo_institution_id = demo_user.institution_id
        institution_role = demo_user.institution_role or (
            InstitutionRole.ADMIN.value
            if str(role).upper() == UserRole.INSTITUTION.value
            else None
        )
        profile_email = demo_user.email
        claims = {
            "user_id": demo_user.id,
            "role": role,
            "company_id": demo_company.id if demo_company else None,
            "driver_id": None,
            "institution_id": demo_institution_id,
            "institution_role": institution_role,
            "force_password_change": bool(demo_user.force_password_change),
            "aud": "atmr-api",
        }
        access_token = create_access_token(
            identity=str(demo_user.public_id),
            additional_claims=claims,
            expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
            fresh=True,
        )
        refresh_token = create_refresh_token(
            identity=str(demo_user.public_id),
            additional_claims={"aud": "atmr-api"},
            expires_delta=current_app.config["JWT_REFRESH_TOKEN_EXPIRES"],
        )

        response_data = {
            **result,
            "target_env": "demo",
            "redirect_to": "/demo/home",
            "recommended_journey": _resolve_demo_journey(
                access.demo_request.organization_type if access else None
            ),
            "token": access_token,
            "refresh_token": refresh_token,
            "trace_id": get_trace_id(),
            "user": {
                "id": demo_user.id,
                "public_id": demo_user.public_id,
                "username": demo_user.username,
                "email": profile_email,
                "first_name": demo_user.first_name,
                "last_name": demo_user.last_name,
                "role": role,
                "institution_role": institution_role,
                "force_password_change": bool(demo_user.force_password_change),
            },
        }
        response = make_response(response_data, 200)
        response.set_cookie(
            current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
            access_token,
            httponly=current_app.config["COOKIE_HTTP_ONLY"],
            secure=current_app.config["COOKIE_SECURE"],
            samesite=current_app.config["COOKIE_SAME_SITE"],
            max_age=int(current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()),
            path=current_app.config["COOKIE_PATH"],
            domain=current_app.config["COOKIE_DOMAIN"],
        )
        response.set_cookie(
            current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
            refresh_token,
            httponly=current_app.config["COOKIE_HTTP_ONLY"],
            secure=current_app.config["COOKIE_SECURE"],
            samesite=current_app.config["COOKIE_SAME_SITE"],
            max_age=int(
                current_app.config["JWT_REFRESH_TOKEN_EXPIRES"].total_seconds()
            ),
            path=current_app.config["COOKIE_PATH"],
            domain=current_app.config["COOKIE_DOMAIN"],
        )
        return response


@demo_access_ns.route("/set-password")
class SetDemoPassword(Resource):
    @jwt_required()
    @demo_access_ns.expect(demo_set_password_model)
    @limiter.limit("10 per minute")  # Protection changement mot de passe répété
    def post(self):
        payload = request.get_json(silent=True) or {}
        new_password = str(payload.get("new_password") or "").strip()
        if len(new_password) < DEMO_PASSWORD_MIN_LENGTH:
            return {
                "ok": False,
                "code": "validation_error",
                "message": (
                    f"Le mot de passe doit contenir au moins {DEMO_PASSWORD_MIN_LENGTH} caracteres."
                ),
            }, 400

        identity = get_jwt_identity()
        user = User.query.filter(User.public_id == str(identity)).first()
        if not user:
            return {
                "ok": False,
                "code": "user_not_found",
                "message": "Compte introuvable.",
            }, 404

        email = str(user.email or "").strip().lower()
        if not email.startswith("demo-"):
            return {
                "ok": False,
                "code": "forbidden",
                "message": "Action reservee aux comptes demo.",
            }, 403

        user.set_password(new_password)
        user.force_password_change = False
        db.session.commit()
        role = (
            user.role.value
            if hasattr(user.role, "value")
            else str(user.role or "client")
        )
        institution_role = user.institution_role or (
            InstitutionRole.ADMIN.value
            if str(role).upper() == UserRole.INSTITUTION.value
            else None
        )
        demo_company = user.company
        claims = {
            "user_id": user.id,
            "role": role,
            "company_id": demo_company.id if demo_company else None,
            "driver_id": None,
            "institution_id": user.institution_id,
            "institution_role": institution_role,
            "force_password_change": False,
            "aud": "atmr-api",
        }
        access_token = create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
            expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
            fresh=True,
        )
        refresh_token = create_refresh_token(
            identity=str(user.public_id),
            additional_claims={"aud": "atmr-api"},
            expires_delta=current_app.config["JWT_REFRESH_TOKEN_EXPIRES"],
        )

        response_data = {
            "ok": True,
            "target_env": "demo",
            "redirect_to": "/demo/home",
            "token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": user.id,
                "public_id": user.public_id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": role,
                "institution_role": institution_role,
                "force_password_change": False,
            },
        }
        response = make_response(response_data, 200)
        response.set_cookie(
            current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
            access_token,
            httponly=current_app.config["COOKIE_HTTP_ONLY"],
            secure=current_app.config["COOKIE_SECURE"],
            samesite=current_app.config["COOKIE_SAME_SITE"],
            max_age=int(current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()),
            path=current_app.config["COOKIE_PATH"],
            domain=current_app.config["COOKIE_DOMAIN"],
        )
        response.set_cookie(
            current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
            refresh_token,
            httponly=current_app.config["COOKIE_HTTP_ONLY"],
            secure=current_app.config["COOKIE_SECURE"],
            samesite=current_app.config["COOKIE_SAME_SITE"],
            max_age=int(
                current_app.config["JWT_REFRESH_TOKEN_EXPIRES"].total_seconds()
            ),
            path=current_app.config["COOKIE_PATH"],
            domain=current_app.config["COOKIE_DOMAIN"],
        )
        return response


@demo_access_ns.route("/analytics")
class DemoAnalyticsEvent(Resource):
    @demo_access_ns.expect(demo_analytics_model)
    @limiter.limit("60 per minute")
    def post(self):
        payload = request.get_json(silent=True) or {}
        event = (payload.get("event") or "").strip()
        data = payload.get("payload") or {}
        allowed = {"demo_session_start", "demo_step_reached", "demo_completed"}
        if event not in allowed:
            return {"ok": False, "code": "invalid_event"}, 400

        current_app.logger.info(
            "[demo-analytics] event=%s ip=%s payload=%s",
            event,
            _get_ip_address(),
            data,
        )
        return {"ok": True}, 200
