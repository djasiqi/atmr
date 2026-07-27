import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from urllib.parse import quote

import sentry_sdk
from flask import (
    current_app,
    make_response,
    render_template,
    request,
)
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (
    Namespace,
    Resource,
    fields,
)
from itsdangerous import URLSafeTimedSerializer
from itsdangerous.exc import BadSignature, SignatureExpired
from marshmallow import (
    Schema,
    ValidationError,
)
from marshmallow import fields as ma_fields
from sqlalchemy.orm import joinedload

from application.users import (
    AuthenticateUserInput,
    AuthenticateUserUseCase,
    RegisterUserInput,
    RegisterUserUseCase,
)
from ext import db, limiter, redis_client, role_required
from middleware.trace_id import get_trace_id
from models import (
    ActivationSession,
    Booking,
    Client,
    Driver,
    User,
)  # Client utilisé pour création directe, User pour type annotations
from models.enums import ClientType, UserRole
from repositories.user_repository import UserRepository
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from routes.api_error_utils import auth_error
from schemas.auth_schemas import (
    FinalizeActivationSchema,
    LoginSchema,
    PasswordlessOtpRequestSchema,
    PasswordlessOtpVerifySchema,
    RegisterSchema,
    ResendActivationSchema,
    UpdateActivationPhoneSchema,
    VerifyEmailActivationSchema,
    VerifySmsActivationSchema,
)
from schemas.validation_utils import handle_validation_error, validate_request
from security.audit_log import AuditLogger
from security.refresh_token_service import (
    RefreshStoreUnavailableError,
    _hash_refresh_token,
    get_user_active_sessions,
    is_token_revoked,
    mark_token_rotated,
    refresh_fail_closed_enabled,
    revoke_active_tokens_for_device,
    revoke_all_user_tokens,
    revoke_refresh_token,
    store_refresh_token,
    update_token_last_used,
)
from security.security_metrics import (
    security_login_attempts_total,
    security_login_failures_total,
    security_logout_total,
    security_token_refreshes_total,
)
from services.demo.access_service import enforce_demo_user_access_validity
from services.notifications.email import send_email_notification
from services.public_guest_booking_pricing import compute_public_guest_booking_price
from services.security.authentication import RefreshTokenService
from services.security.csrf import generate_csrf_token
from services.security.login_origin import validate_login_origin_for_web
from shared.client_surface_contracts import (
    CANONICAL_ADDRESS_CONTRACT_VERSION,
    PREVIEW_CONTRACT_VERSION,
    PRICING_CONTRACT_VERSION,
    STATUS_DICTIONARY_VERSION,
)
from shared.constants import AuthErrorCodes
from shared.driver_surface_contracts import (
    DRIVER_SOCKET_CONTRACT_VERSION,
    DRIVER_TRACKING_CONTRACT_VERSION,
    MISSION_SNAPSHOT_VERSION,
    MISSION_STATUS_VERSION,
)
from shared.error_handlers import APIErrorHandler
from shared.logging_utils import mask_email

logger = logging.getLogger(__name__)

# Initialisation des repositories
user_repo = UserRepository()

auth_ns = Namespace("auth", description="Opérations liées à l'authentification")

# ✅ S1: Modèle Swagger pour la réponse CSRF token
csrf_token_response_model = auth_ns.model(
    "CSRFTokenResponse",
    {
        "csrf_token": fields.String(
            required=True, description="Token CSRF à inclure dans les requêtes mutantes"
        ),
        "ttl": fields.Integer(
            required=True, description="Durée de vie du token en secondes"
        ),
    },
)

# ✅ S1: Modèle Swagger pour la réponse CSRF token
csrf_token_response_model = auth_ns.model(
    "CSRFTokenResponse",
    {
        "csrf_token": fields.String(
            required=True, description="Token CSRF à inclure dans les requêtes mutantes"
        ),
        "ttl": fields.Integer(
            required=True, description="Durée de vie du token en secondes"
        ),
    },
)

# Constante pour la longueur du hash de version du mot de passe
PASSWORD_HASH_VERSION_LENGTH = 16
_MOBILE_UA_MARKERS = (
    "okhttp",
    "cfnetwork",
    "darwin",
    "iphone",
    "ipad",
    "android",
    "mobile",
    "lirioprations",
    "lirioperations",
)
_PUBLIC_PRE_REQUEST_CACHE: dict[str, str] = {}


def _is_mobile_request() -> bool:
    """Détecte une requête mobile (app native iOS/Android)."""
    if request.headers.get("X-Requested-With") == "Expo":
        return True
    # Header envoyé par l'app unifiée (Expo) — fiable même sans UA « mobile ».
    client_platform = (request.headers.get("X-Client-Platform") or "").strip().lower()
    if client_platform in {"ios", "android"}:
        return True
    user_agent = (request.headers.get("User-Agent") or "").lower()
    return any(marker in user_agent for marker in _MOBILE_UA_MARKERS)


def _clear_web_auth_cookies(response) -> None:
    """Supprime les cookies auth web (Domain configuré + host-only legacy).

    Après passage à ``COOKIE_DOMAIN=.lirie.ch``, d'anciens cookies host-only
    (www ou api, sans Domain) peuvent rester : le navigateur les traite comme
    des cookies distincts. Un logout qui n'efface que ``Domain=.lirie.ch``
    laisse donc une session fantôme — d'où l'obligation de vider manuellement.
    """
    access_name = current_app.config["COOKIE_ACCESS_TOKEN_NAME"]
    refresh_name = current_app.config["COOKIE_REFRESH_TOKEN_NAME"]
    path = current_app.config.get("COOKIE_PATH") or "/"
    secure = bool(current_app.config.get("COOKIE_SECURE"))
    httponly = bool(current_app.config.get("COOKIE_HTTP_ONLY"))
    samesite = current_app.config.get("COOKIE_SAME_SITE") or "Lax"

    domains: list[str | None] = [None]
    configured = (current_app.config.get("COOKIE_DOMAIN") or "").strip() or None
    if configured:
        domains.append(configured)
        stripped = configured.lstrip(".")
        if stripped and stripped not in domains:
            domains.append(stripped)

    for domain in domains:
        for name in (access_name, refresh_name):
            try:
                response.delete_cookie(
                    name,
                    path=path,
                    domain=domain,
                    secure=secure,
                    httponly=httponly,
                    samesite=samesite,
                )
            except TypeError:
                response.delete_cookie(name, path=path, domain=domain)
            # Ceinture : certains clients n'honorent que Set-Cookie expires=0
            response.set_cookie(
                name,
                "",
                expires=0,
                max_age=0,
                path=path,
                domain=domain,
                secure=secure,
                httponly=httponly,
                samesite=samesite,
            )


def _resolve_access_token_expires(is_mobile_request: bool) -> timedelta:
    """Résout la durée d'expiration de l'access token selon le client."""
    if is_mobile_request:
        return current_app.config.get(
            "JWT_MOBILE_ACCESS_TOKEN_EXPIRES",
            current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
        )
    return current_app.config["JWT_ACCESS_TOKEN_EXPIRES"]


def _resolve_refresh_token_expires(
    *, is_mobile_request: bool, remember_me: bool
) -> timedelta:
    """Résout la durée du refresh token selon le client et l'option remember_me.

    - Clients mobile/API: durée par défaut (``JWT_REFRESH_TOKEN_EXPIRES``) pour
      ne pas casser les flux existants. Le flag ``remember_me`` est ignoré.
    - Web sans remember_me: durée courte (``JWT_REFRESH_TOKEN_SHORT_EXPIRES_SECONDS``,
      défaut 1h). Combiné à un cookie de session (Max-Age=None) côté navigateur.
    - Web avec remember_me: durée longue (``JWT_REFRESH_TOKEN_LONG_EXPIRES_SECONDS``,
      défaut 30j). Cookie persistant.
    """
    default_delta: timedelta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
    if is_mobile_request:
        return default_delta

    if remember_me:
        long_seconds = int(
            os.getenv("JWT_REFRESH_TOKEN_LONG_EXPIRES_SECONDS", str(30 * 24 * 3600))
        )
        return timedelta(seconds=long_seconds)

    short_seconds = int(
        os.getenv("JWT_REFRESH_TOKEN_SHORT_EXPIRES_SECONDS", str(60 * 60))
    )
    return timedelta(seconds=short_seconds)


def _resolve_remember_me_from_refresh_token(
    refresh_token: str,
    *,
    is_mobile_request: bool,
) -> bool:
    """Déduit remember_me du JWT refresh (claim explicite ou TTL hérité).

    Pour les tokens émis avant l'introduction de la claim, on infère depuis le TTL.
    """
    if is_mobile_request:
        return False

    try:
        decoded = decode_token(refresh_token, allow_expired=False)
    except Exception:
        return False

    if "remember_me" in decoded:
        return bool(decoded["remember_me"])

    ttl = int(decoded.get("exp", 0)) - int(decoded.get("iat", 0))
    short_seconds = int(
        os.getenv("JWT_REFRESH_TOKEN_SHORT_EXPIRES_SECONDS", str(60 * 60))
    )
    long_seconds = int(
        os.getenv("JWT_REFRESH_TOKEN_LONG_EXPIRES_SECONDS", str(30 * 24 * 3600))
    )
    if ttl <= short_seconds * 2:
        return False
    return ttl >= long_seconds // 2


def _refresh_cookie_max_age(
    *, remember_me: bool, refresh_expires_delta: timedelta
) -> int | None:
    """Max-Age cookie refresh : persistant si remember_me, session sinon."""
    if remember_me:
        return int(refresh_expires_delta.total_seconds())
    return None


ACTIVATION_EMAIL_TTL_MINUTES = int(os.getenv("ACTIVATION_EMAIL_TTL_MINUTES", "30"))
ACTIVATION_SMS_TTL_MINUTES = int(os.getenv("ACTIVATION_SMS_TTL_MINUTES", "5"))
ACTIVATION_SMS_MAX_ATTEMPTS = int(os.getenv("ACTIVATION_SMS_MAX_ATTEMPTS", "5"))
ACTIVATION_RESEND_COOLDOWN_SECONDS = int(
    os.getenv("ACTIVATION_RESEND_COOLDOWN_SECONDS", "60")
)
ACTIVATION_RESEND_DAILY_LIMIT = int(os.getenv("ACTIVATION_RESEND_DAILY_LIMIT", "10"))
ACTIVATION_SMS_LOCK_MINUTES = int(os.getenv("ACTIVATION_SMS_LOCK_MINUTES", "15"))
RESET_PASSWORD_TOKEN_TTL_SECONDS = int(
    os.getenv("RESET_PASSWORD_TOKEN_TTL_SECONDS", str(60 * 60))
)

# Masquage téléphone (suffixe affiché)
_PHONE_MASK_MIN_DIGITS_FOR_TAIL = 2


def _activation_serializer() -> URLSafeTimedSerializer:
    secret_key = current_app.config.get("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY non configurée")
    return URLSafeTimedSerializer(secret_key)


def _hash_plain_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mask_phone(phone: str | None) -> str:
    if not phone:
        return "inconnu"
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) <= _PHONE_MASK_MIN_DIGITS_FOR_TAIL:
        return "*" * len(digits)
    return f"+** *** *** {digits[-2:]}"


def _build_activation_status(session: ActivationSession) -> dict[str, object]:
    return {
        "email_verified": bool(session.email_verified_at),
        "phone_verified": bool(session.phone_verified_at),
        "is_complete": bool(session.email_verified_at and session.phone_verified_at),
        "is_finalized": bool(session.consumed_at),
        # Statut livraison (pas email_last_error — interne uniquement)
        "email_delivery_status": session.email_delivery_status,
    }


def _build_activation_email_link(token: str) -> str:
    environment = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
    default_frontend_url = (
        "http://localhost:3000"
        if environment in {"development", "testing"}
        else "https://www.lirie.ch"
    )
    frontend_url = (
        os.getenv("FRONTEND_URL")
        or os.getenv("PUBLIC_FRONTEND_URL")
        or os.getenv("PUBLIC_APP_URL")
        or default_frontend_url
    ).rstrip("/")
    return f"{frontend_url}/activate-account?token={token}"


def _resolve_public_web_base_url() -> str:
    environment = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
    default_frontend_url = (
        "http://localhost:3000"
        if environment in {"development", "testing"}
        else "https://www.lirie.ch"
    )
    return (
        os.getenv("FRONTEND_URL")
        or os.getenv("PUBLIC_FRONTEND_URL")
        or os.getenv("PUBLIC_APP_URL")
        or default_frontend_url
    ).rstrip("/")


def _build_reset_password_web_link(token: str) -> str:
    # URL canonique web : /reset-password?token=...
    return f"{_resolve_public_web_base_url()}/reset-password?token={quote(token)}"


def _build_reset_password_deep_link(token: str) -> str:
    scheme = (os.getenv("PUBLIC_MOBILE_SCHEME") or "lirie").strip() or "lirie"
    return f"{scheme}://reset-password?token={quote(token)}"


def _public_link_serializer() -> URLSafeTimedSerializer:
    secret_key = current_app.config.get("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY non configurée")
    return URLSafeTimedSerializer(secret_key)


def _resolve_booking_status_token_ttl_seconds() -> int:
    raw = os.getenv("BOOKING_STATUS_TOKEN_TTL_SECONDS", "3600")
    try:
        value = int(raw)
        return max(300, value)
    except Exception:
        return 3600


def _booking_status_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _build_public_pre_request_key(draft_id: str) -> str:
    return f"public:pre_request:{draft_id}"


def _build_public_guest_booking_key(guest_booking_id: str) -> str:
    return f"public:guest_booking:{guest_booking_id}"


def _decode_guest_booking_status_token(token: str) -> tuple[str | None, str | None]:
    """Retourne ``(guest_booking_id, code_erreur)`` ; code_erreur si token invalide."""
    serializer = _public_link_serializer()
    try:
        decoded = serializer.loads(
            token,
            salt="guest-booking-status-link",
            max_age=_resolve_guest_booking_ttl_seconds(),
        )
    except SignatureExpired:
        return None, "token_expired"
    except BadSignature:
        return None, "token_invalid"
    guest_booking_id = str((decoded or {}).get("guest_booking_id") or "").strip()
    if not guest_booking_id:
        return None, "token_invalid"
    return guest_booking_id, None


def _build_passwordless_otp_key(session_id: str) -> str:
    return f"auth:passwordless_otp:{session_id}"


def _resolve_guest_booking_ttl_seconds() -> int:
    raw = os.getenv("PUBLIC_GUEST_BOOKING_TTL_SECONDS", "604800")
    with suppress(Exception):
        value = int(raw)
        return max(1800, value)
    return 604800


def _resolve_passwordless_otp_ttl_seconds() -> int:
    raw = os.getenv("PASSWORDLESS_OTP_TTL_SECONDS", "600")
    with suppress(Exception):
        value = int(raw)
        return max(120, value)
    return 600


def _create_passwordless_otp_code() -> str:
    return f"{secrets.randbelow(1000000):06d}"


def _user_token_version(user: User) -> int:
    """Version JWT pour révocation globale après changement de mot de passe."""
    return int(getattr(user, "token_version", 0) or 0)


def _passwordless_allowed_in_environment() -> bool:
    """Passwordless OTP uniquement en développement (Lot 0 SEC-03)."""
    environment = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
    return environment == "development"


def _passwordless_debug_code_enabled() -> bool:
    """debug_code uniquement si ENVIRONMENT=development ET PASSWORDLESS_DEBUG_CODE=true."""
    if not _passwordless_allowed_in_environment():
        return False
    flag = str(os.getenv("PASSWORDLESS_DEBUG_CODE", "false")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _public_cache_get(key: str) -> str | None:
    if redis_client:
        raw = redis_client.get(key)
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        if isinstance(raw, str):
            return raw
    return _PUBLIC_PRE_REQUEST_CACHE.get(key)


def _public_cache_setex(key: str, ttl_seconds: int, value: str) -> None:
    if redis_client:
        redis_client.setex(key, ttl_seconds, value)
        return
    _PUBLIC_PRE_REQUEST_CACHE[key] = value


def _public_cache_delete(key: str) -> None:
    if redis_client:
        redis_client.delete(key)
    _PUBLIC_PRE_REQUEST_CACHE.pop(key, None)


def _map_public_booking_status(raw_status: str | None) -> tuple[str, str]:
    value = (raw_status or "").strip().lower()
    if value in {"accepted", "confirmed", "assigned"}:
        return "confirmed", "Transport confirme"
    if value in {"in_progress", "in-progress", "started", "ongoing"}:
        return "in_progress", "Transport en route"
    if value in {"completed", "done", "finished"}:
        return "completed", "Transport termine"
    if value in {"cancelled", "canceled", "rejected"}:
        return "cancelled", "Transport annule"
    if value in {"pending", "awaiting_assignment", "new"}:
        return "pending", "Transport en attente"
    return "unknown", "Statut indisponible"


def _public_guest_booking_id_from_dossier_str(raw: str) -> int | None:
    """Réservation rapide : le client peut saisir le n° de dossier (id booking) au lieu du long jeton.

    Réservé aux ``Booking`` dont ``created_via == public_guest`` pour ne pas exposer
    toutes les courses.
    """
    t = (raw or "").strip()
    if not re.fullmatch(r"\d{3,12}", t):
        return None
    from models.enums import BookingCreatedVia

    try:
        bid = int(t)
    except ValueError:
        return None
    b = db.session.get(Booking, bid)
    if b is None:
        return None
    cv = getattr(b, "created_via", None)
    if cv == BookingCreatedVia.PUBLIC_GUEST:
        return bid
    if getattr(cv, "value", None) == BookingCreatedVia.PUBLIC_GUEST.value:
        return bid
    return None


def _load_booking_status_from_token(token: str) -> tuple[int | None, str | None]:
    t = (token or "").strip()
    revoked_key = f"public:booking_status:revoked:{_booking_status_token_hash(t)}"
    if _public_cache_get(revoked_key):
        return None, "revoked"
    serializer = _public_link_serializer()
    max_age = _resolve_booking_status_token_ttl_seconds()
    try:
        data = serializer.loads(t, salt="booking-status-public-link", max_age=max_age)
    except SignatureExpired:
        return None, "expired"
    except BadSignature:
        guest_bid = _public_guest_booking_id_from_dossier_str(t)
        if guest_bid is not None:
            return int(guest_bid), None
        return None, "invalid"
    if not isinstance(data, dict):
        return None, "invalid"
    booking_id_raw = data.get("booking_id")
    if booking_id_raw is None:
        return None, "invalid"
    try:
        booking_id = int(booking_id_raw)
    except Exception:
        return None, "invalid"
    return booking_id, None


def _reset_user_password_with_policy(user: User, new_password: str):
    from security.password_policy import (
        PasswordPolicyError,
        PasswordPolicyService,
    )

    try:
        PasswordPolicyService.validate_password(
            new_password, user_id=user.id, check_history=True
        )
    except PasswordPolicyError as e:
        return APIErrorHandler.handle_validation_error(
            e.message,
            field="new_password",
            logger_instance=logger,
        )

    was_forced = bool(getattr(user, "force_password_change", False))
    user.set_password(new_password)  # nosem
    user.force_password_change = False
    # Lot 0 SEC-02: invalider tous les access tokens déjà émis
    user.token_version = int(getattr(user, "token_version", 0) or 0) + 1
    if hasattr(user, "password_expires_at"):
        user.password_expires_at = None
    if hasattr(user, "temporary_password_created_at"):
        user.temporary_password_created_at = None
    should_mark_first_login = (
        hasattr(user, "first_login_completed_at")
        and not user.first_login_completed_at
        and user.institution_id
        and getattr(user, "authentication_method", "email") == "username"
    )
    if should_mark_first_login:
        user.first_login_completed_at = datetime.now(UTC)
        if was_forced:
            try:
                from models.institution_user_audit_event import (
                    InstitutionUserAuditEvent,
                )

                InstitutionUserAuditEvent.record(
                    institution_id=user.institution_id,
                    target_user_id=user.id,
                    performed_by_user_id=user.id,
                    event_type="first_password_change_completed",
                    ip_address=request.remote_addr if request else None,
                    user_agent=request.headers.get("User-Agent") if request else None,
                )
            except Exception as audit_err:
                logger.warning(
                    "Audit first_password_change_completed failed: %s", audit_err
                )

    try:
        from security.security_metrics import (
            security_token_invalidations_total,
        )
        from security.token_blacklist import revoke_token

        revoke_all_user_tokens(user.id, reason="Changement de mot de passe")
        security_token_invalidations_total.labels(reason="password_change").inc()
        with suppress(Exception):
            revoke_token()
    except Exception as revoke_error:
        logger.warning(
            "Échec révocation tokens lors changement mot de passe (ignoré): %s",
            str(revoke_error),
        )

    db.session.commit()
    return {
        "message": "Mot de passe réinitialisé avec succès.",
        "reason": "password_reset_succeeded",
        "outcome_class": "success",
        "retryable": False,
        "require_relogin": True,
    }, 200


def _password_reset_terminal_error(
    *, reason: str, message: str, status_code: int = 400
):
    return {
        "error": message,
        "reason": reason,
        "outcome_class": "terminal_error",
        "retryable": False,
    }, status_code


def _generate_sms_otp() -> str:
    return f"{secrets.randbelow(1000000):06d}"


def _is_same_utc_day(a: datetime, b: datetime) -> bool:
    return a.astimezone(UTC).date() == b.astimezone(UTC).date()


def _enforce_resend_policy(
    *, last_sent_at: datetime | None, resend_count: int
) -> tuple[bool, str | None, int]:
    now = datetime.now(UTC)
    if last_sent_at:
        elapsed = int((now - last_sent_at).total_seconds())
        if elapsed < ACTIVATION_RESEND_COOLDOWN_SECONDS:
            return False, "cooldown", ACTIVATION_RESEND_COOLDOWN_SECONDS - elapsed

    daily_count = resend_count
    if last_sent_at and not _is_same_utc_day(last_sent_at, now):
        daily_count = 0
    if daily_count >= ACTIVATION_RESEND_DAILY_LIMIT:
        return False, "daily_limit", 0

    return True, None, 0


def _send_activation_email(user: User, token: str) -> None:
    """Envoi synchrone (tests / fallback). Lève sur échec — pas de return silencieux."""
    user_email_raw = getattr(user, "email", None)
    if user_email_raw is None or not str(user_email_raw).strip():
        raise ValueError("Email utilisateur manquant pour envoi activation")
    user_email = str(user_email_raw).strip()
    verification_link = _build_activation_email_link(token)
    text_body = (
        "Bienvenue sur LIRIE.\n\n"
        "Cliquez sur ce lien pour confirmer votre email:\n"
        f"{verification_link}\n\n"
        "Ce lien expire rapidement. Si vous n'etes pas a l'origine de cette action, ignorez cet email."
    )

    html_body = ""
    with suppress(Exception):
        html_body = render_template(
            "emails/activation_email.html",
            activation_link=verification_link,
            product_name="LIRIE",
            company_name="LIRIE",
            current_year=datetime.now(UTC).year,
        )

    send_result = send_email_notification(
        email=user_email,
        subject="Activation de votre compte",
        body=html_body or text_body,
        html=bool(html_body),
        notification_type="activation_signup",
        from_name=os.getenv("ACTIVATION_EMAIL_FROM_NAME", "LIRIE"),
        from_email=os.getenv("ACTIVATION_EMAIL_FROM", "noreply@lirie.ch"),
        reply_to=os.getenv("ACTIVATION_EMAIL_REPLY_TO", "support@lirie.ch"),
        raise_on_error=True,
    )
    if not bool(send_result.get("ok")):
        error_msg = str(send_result.get("error", "Email provider error"))
        raise RuntimeError(error_msg)


def _activation_email_send_failed_body(
    *,
    activation_session_id: str,
    user: User,
    message: str = "Impossible d'envoyer l'email pour le moment. Vérifiez l'adresse puis réessayez.",
) -> tuple[dict[str, object], int]:
    return (
        {
            "error": "email_send_failed",
            "message": message,
            "activation_session_id": activation_session_id,
            "masked_email": mask_email(user.email or ""),
            "masked_phone": _mask_phone(user.phone),
        },
        502,
    )


def _send_activation_sms(user: User, code: str) -> bool:
    if not user.phone:
        return False
    from services.notifications.sms import send_sms_notification

    sms_result = send_sms_notification(
        phone=user.phone,
        message=f"ATMR: votre code d'activation est {code}. Il expire dans quelques minutes.",
        notification_type="activation_signup",
    )
    return bool(sms_result.get("ok"))


# Modèle Swagger pour la connexion (login)
login_model = auth_ns.model(
    "Login",
    {
        "email": fields.String(
            required=True,
            description="L'adresse email de l'utilisateur (format email valide)",
        ),
        "password": fields.String(
            required=True, description="Le mot de passe de l'utilisateur", min_length=6
        ),
        "remember_me": fields.Boolean(
            required=False,
            default=False,
            description=(
                "Si true (web) : refresh token long-lived (ex: 30j) avec cookie "
                "persistant. Si false/absent : refresh token court (ex: 1h) avec "
                "cookie de session. Ignoré pour les clients mobiles."
            ),
        ),
    },
)

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(auth_ns)
validation_error_model = create_validation_error_model(auth_ns)
not_found_error_model = create_not_found_error_model(auth_ns)
permission_error_model = create_permission_error_model(auth_ns)

# ✅ P0: Modèle de réponse succès pour login
login_success_model = auth_ns.model(
    "LoginSuccess",
    {
        "message": fields.String(
            required=True,
            description="Message de confirmation",
            example="Connexion réussie",
        ),
        "token": fields.String(
            required=True, description="Token d'accès JWT (pour mobile)"
        ),
        "refresh_token": fields.String(
            required=True, description="Token de rafraîchissement (pour mobile)"
        ),
        "user": fields.Raw(
            required=True,
            description="Informations utilisateur",
            example={
                "id": 1,
                "public_id": "abc123",
                "username": "john.doe",
                "email": "john.doe@example.com",
                "role": "client",
                "force_password_change": False,
            },
        ),
        "trace_id": fields.String(
            required=False,
            description="ID de traçage pour le support",
            example="a1b2c3d4",
        ),
    },
)

# Modèle Swagger pour obtenir un token fresh
fresh_token_request_model = auth_ns.model(
    "FreshTokenRequest",
    {
        "password": fields.String(
            required=True,
            description="Le mot de passe de l'utilisateur pour vérification",
            min_length=6,
        ),
    },
)

# Modèle Swagger pour la réponse de fresh token
fresh_token_response_model = auth_ns.model(
    "FreshTokenResponse",
    {
        "access_token": fields.String(
            required=True, description="Nouveau token d'accès 'fresh'"
        ),
        "message": fields.String(description="Message de confirmation"),
    },
)

# Modèle Swagger pour l'inscription (register)
register_model = auth_ns.model(
    "Register",
    {
        "username": fields.String(
            required=True,
            description="Le nom d'utilisateur",
            min_length=3,
            max_length=50,
        ),
        "email": fields.String(
            required=True,
            description="L'adresse email de l'utilisateur (format email valide)",
        ),
        "password": fields.String(
            required=True, description="Le mot de passe de l'utilisateur", min_length=6
        ),
        "first_name": fields.String(description="Prénom", default=None, max_length=100),
        "last_name": fields.String(description="Nom", default=None, max_length=100),
        "phone": fields.String(
            description="Numéro de téléphone", default=None, max_length=20
        ),
        "address": fields.String(description="Adresse", default=None, max_length=500),
        "birth_date": fields.String(
            description="Date de naissance (YYYY-MM-DD)",
            default=None,
            pattern="^\\d{4}-\\d{2}-\\d{2}$",
        ),
        "gender": fields.String(
            description="Genre (male|female|other)",
            default=None,
            enum=["male", "female", "other"],
        ),
        "profile_image": fields.String(
            description="URL ou données base64 de l'image de profil", default=None
        ),
    },
)

# Modèle Swagger pour la réponse de logout (succès)
logout_response_model = auth_ns.model(
    "LogoutResponse",
    {
        "message": fields.String(description="Message de confirmation de déconnexion"),
    },
)

# Modèle Swagger pour la réponse d'erreur de logout
logout_error_model = auth_ns.model(
    "LogoutError",
    {
        "error": fields.String(description="Message d'erreur"),
    },
)

# Modèle Swagger pour la réponse de révocation de sessions
revoke_sessions_response_model = auth_ns.model(
    "RevokeSessionsResponse",
    {
        "message": fields.String(
            required=True,
            description="Message de confirmation avec nombre de sessions révoquées",
        ),
        "sessions_revoked": fields.Integer(
            required=True, description="Nombre de sessions révoquées"
        ),
    },
)

# Modèle Swagger pour une session
session_model = auth_ns.model(
    "Session",
    {
        "id": fields.Integer(required=True, description="ID de la session"),
        "device_id": fields.String(required=False, description="ID de l'appareil"),
        "device_name": fields.String(required=False, description="Nom de l'appareil"),
        "created_at": fields.String(
            required=True, description="Date de création (ISO8601)"
        ),
        "expires_at": fields.String(
            required=True, description="Date d'expiration (ISO8601)"
        ),
        "last_used_at": fields.String(
            required=False, description="Date de dernière utilisation (ISO8601)"
        ),
        "is_revoked": fields.Boolean(
            required=True, description="Si la session est révoquée"
        ),
        "revoked_at": fields.String(
            required=False, description="Date de révocation (ISO8601)"
        ),
    },
)

# Modèle Swagger pour la réponse de liste de sessions
list_sessions_response_model = auth_ns.model(
    "ListSessionsResponse",
    {
        "sessions": fields.List(
            fields.Nested(session_model),
            required=True,
            description="Liste des sessions actives",
        ),
        "count": fields.Integer(
            required=True, description="Nombre total de sessions actives"
        ),
    },
)

# Schéma Marshmallow pour valider les données d'inscription


class UserSchema(Schema):
    username = ma_fields.String(required=True)
    email = ma_fields.Email(required=True)
    password = ma_fields.String(required=True)

    # --- CORRECTION APPLIQUÉE ICI ---
    # Remplacement de 'missing' par 'load_default'
    first_name = ma_fields.String(load_default=None)
    last_name = ma_fields.String(load_default=None)
    phone = ma_fields.String(load_default=None)
    address = ma_fields.String(load_default=None)
    birth_date = ma_fields.Date(load_default=None)
    gender = ma_fields.String(load_default=None)
    profile_image = ma_fields.String(load_default=None)


# ========================
# 1. Connexion / Login
# ========================
def _login_post_body():
    """Corps du POST /login (hors gestionnaire d'erreur global)."""
    try:
        data = request.get_json() or {}
    except Exception as json_error:
        # Gérer spécifiquement les erreurs de parsing JSON (BadRequest 400)
        # pour éviter qu'elles soient transformées en 500 par le gestionnaire global
        from werkzeug.exceptions import (
            BadRequest,
        )

        if isinstance(json_error, BadRequest):
            logger.warning("Erreur parsing JSON dans login: %s", json_error)
            return APIErrorHandler.handle_validation_error(
                "Format JSON invalide dans la requête",
                logger_instance=logger,
            )
        # Si ce n'est pas une BadRequest, laisser le gestionnaire global la gérer
        raise

    # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
    try:
        validated_data = validate_request(LoginSchema(), data)
    except ValidationError as e:
        return handle_validation_error(e)

    email = validated_data["email"]
    password = validated_data["password"]
    remember_me = bool(validated_data.get("remember_me", False))

    # ✅ DDD: Utiliser le use case pour authentifier l'utilisateur
    uc = AuthenticateUserUseCase()
    input_data = AuthenticateUserInput(email=email, password=password)
    auth_result = uc.execute(input_data)

    user = None if not auth_result.success else auth_result.user

    if not user:
        err_code = (auth_result.error or {}).get("error", "invalid_credentials")
        # ✅ Priorité 7: Audit logging pour login échoué
        try:
            AuditLogger.log_action(
                action_type="login_failed",
                action_category="security",
                user_type="unknown",
                result_status="failure",
                result_message="Email ou mot de passe invalide",
                action_details={
                    "email": mask_email(email),
                    "reason": err_code,
                },
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )
            # ✅ Priorité 7: Métriques Prometheus pour login échoué
            security_login_attempts_total.labels(type="failed").inc()
            security_login_failures_total.inc()
        except Exception as audit_error:
            # Ne pas bloquer la réponse si l'audit logging échoue
            logger.warning("Échec audit logging login_failed: %s", audit_error)

        # ✅ S3: Enregistrer tentative échouée pour détection d'alertes
        try:
            from security.security_alerts import SecurityAlertService

            SecurityAlertService.record_login_failure(
                ip_address=request.remote_addr or "unknown", email=email
            )
        except Exception as alert_error:
            logger.debug(
                "[SecurityAlerts] Failed to record login failure: %s",
                alert_error,
            )

        # ✅ P0: Ajouter trace_id dans l'erreur
        trace_id = get_trace_id()
        logger.warning(
            "Login failed - email: %s, reason: %s, trace_id: %s",
            mask_email(email),
            err_code,
            trace_id,
        )
        # Toujours renvoyer un code générique pour éviter d'exposer
        # si l'email existe ou si le mot de passe est incorrect.
        auth_code = AuthErrorCodes.INVALID_CREDENTIALS
        return auth_error(
            auth_code,
            "Les données de connexion sont incorrectes.",
            401,
            details={"trace_id": trace_id},
        )

    is_active, error_message = _check_user_profile_active(user)
    if not is_active:
        trace_id = get_trace_id()
        pending_activation_id = None
        pending_masked_email = None
        pending_masked_phone = None
        reason = (
            "account_pending_activation"
            if getattr(user, "account_status", None) == "pending_activation"
            else "account_disabled"
        )
        if reason == "account_pending_activation":
            latest_activation = (
                ActivationSession.query.filter_by(user_id=user.id)
                .order_by(ActivationSession.created_at.desc())
                .first()
            )
            if latest_activation:
                pending_activation_id = latest_activation.activation_session_id
                pending_masked_email = mask_email(user.email or "")
                pending_masked_phone = _mask_phone(user.phone)
        logger.warning(
            "Login rejected (inactive profile) - email: %s, reason: %s, trace_id: %s",
            mask_email(email),
            error_message,
            trace_id,
        )
        return {
            "error": error_message or "Compte désactivé",
            "reason": reason,
            "trace_id": trace_id,
            "activation_session_id": pending_activation_id,
            "masked_email": pending_masked_email,
            "masked_phone": pending_masked_phone,
            "outcome_class": "terminal_error",
            "retryable": False,
        }, 403

    is_mobile_request = _is_mobile_request()

    # Création du token avec le rôle dans additional_claims
    # ✅ SECURITY: Ajout claim 'aud' (audience) pour prévenir token replay
    claims = {
        "role": user.role.value,
        "company_id": _resolve_company_id(user),
        "driver_id": getattr(user, "driver_id", None),
        "institution_id": getattr(user, "institution_id", None),
        "institution_role": getattr(user, "institution_role", None),
        "aud": "atmr-api",  # Audience claim pour sécurité
        "token_version": _user_token_version(user),
    }
    access_token = create_access_token(
        identity=str(user.public_id),
        # ⚠️ ID numérique attendu par dispatch_routes
        additional_claims=claims,
        expires_delta=_resolve_access_token_expires(is_mobile_request),
        fresh=True,  # ✅ Token fresh lors de la connexion initiale
    )

    # Création du refresh token
    # (durée configurée dans JWT_REFRESH_TOKEN_EXPIRES)
    # ✅ SECURITY: Ajouter la claim 'aud' et 'pwd_hash' pour invalidation après changement de mot de passe
    pwd_hash_version = _get_password_hash_version(user)
    refresh_expires_delta = _resolve_refresh_token_expires(
        is_mobile_request=is_mobile_request,
        remember_me=remember_me,
    )
    refresh_claims: dict[str, object] = {
        "aud": "atmr-api",
        "pwd_hash": pwd_hash_version,
        "token_version": _user_token_version(user),
    }
    if not is_mobile_request:
        refresh_claims["remember_me"] = remember_me
    refresh_token = create_refresh_token(
        identity=str(user.public_id),
        additional_claims=refresh_claims,
        expires_delta=refresh_expires_delta,
    )

    # ✅ PHASE 2: Stocker le refresh token dans Redis et DB (fail-closed Lot 1)
    try:
        token_service = RefreshTokenService()
        token_service.store_token(
            user.id,
            refresh_token,
            ttl_seconds=int(refresh_expires_delta.total_seconds()),
        )

        refresh_expires_at = datetime.now(UTC) + refresh_expires_delta
        device_id = request.headers.get("X-Device-ID")
        device_name = request.headers.get("X-Device-Name")
        if device_id:
            revoke_active_tokens_for_device(
                user.id,
                device_id,
                reason="Remplacé par nouvelle session (même appareil)",
            )
        store_refresh_token(
            token=refresh_token,
            user_id=user.id,
            expires_at=refresh_expires_at,
            device_id=device_id,
            device_name=device_name,
        )

        max_active_tokens = _resolve_max_active_refresh_tokens(user)
        token_service.limit_active_tokens(user.id, max_active_tokens)
    except Exception as store_error:
        logger.error(
            "Échec stockage refresh token: %s - %s",
            type(store_error).__name__,
            str(store_error),
        )
        if refresh_fail_closed_enabled() and not current_app.config.get("TESTING"):
            return {
                "error": "service_unavailable",
                "message": "Stockage session indisponible. Réessayez.",
            }, 503

    # ✅ Priorité 7: Audit logging pour login réussi
    try:
        AuditLogger.log_action(
            action_type="login_success",
            action_category="security",
            user_id=user.id,
            user_type=user.role.value if user.role else "unknown",
            result_status="success",
            action_details={
                "email": mask_email(email),
                "username": user.username,
                "role": user.role.value if user.role else None,
            },
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
        )
        # ✅ Priorité 7: Métrique Prometheus pour login réussi
        security_login_attempts_total.labels(type="success").inc()
    except Exception as audit_error:
        # Ne pas bloquer le login si l'audit logging échoue
        logger.warning("Échec audit logging login_success: %s", audit_error)

    # ✅ Migration localStorage → cookies httpOnly
    # ✅ P0: Ajouter trace_id dans la réponse
    trace_id = get_trace_id()
    logger.info(
        "Login success",
        extra={
            "trace_id": trace_id,
            "user_id": user.id,
            "email": mask_email(email),
        },
    )

    # Créer la réponse JSON
    response_data = {
        "message": "Connexion réussie",
        "user": {
            "id": user.id,
            "public_id": user.public_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "force_password_change": user.force_password_change,
            "must_complete_onboarding": _must_complete_onboarding(user),
            "onboarding_reasons": _onboarding_reasons(user),
            "password_expires_at": user.password_expires_at.isoformat()
            if getattr(user, "password_expires_at", None)
            else None,
        },
        "trace_id": trace_id,
    }

    # Lot 1-E : web = cookies only (pas de tokens JSON) ; mobile = Bearer/JSON
    if is_mobile_request:
        response_data["token"] = access_token
        response_data["access_token"] = access_token
        response_data["refresh_token"] = refresh_token

    # Créer la réponse avec make_response pour pouvoir définir les cookies
    response = make_response(response_data, 200)

    # ✅ Définir cookies httpOnly pour web (pas pour mobile)
    if not is_mobile_request:
        # Effacer d'éventuels cookies host-only hérités avant de poser Domain=.lirie.ch
        _clear_web_auth_cookies(response)
        # Cookie access_token
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

        # Cookie refresh_token
        # remember_me=True  -> cookie persistant (Max-Age aligné sur le TTL serveur)
        # remember_me=False -> cookie de session (max_age=None) supprimé à la
        # fermeture du navigateur, TTL serveur court (cf. _resolve_refresh_token_expires)
        refresh_cookie_max_age = (
            int(refresh_expires_delta.total_seconds()) if remember_me else None
        )
        response.set_cookie(
            current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
            refresh_token,
            httponly=current_app.config["COOKIE_HTTP_ONLY"],
            secure=current_app.config["COOKIE_SECURE"],
            samesite=current_app.config["COOKIE_SAME_SITE"],
            max_age=refresh_cookie_max_age,
            path=current_app.config["COOKIE_PATH"],
            domain=current_app.config["COOKIE_DOMAIN"],
        )

    return response


@auth_ns.route("/login")
class Login(Resource):
    @auth_ns.expect(login_model)
    @auth_ns.response(200, "Connexion réussie", login_success_model)
    @auth_ns.response(400, "Erreur de validation", validation_error_model)
    @auth_ns.response(401, "Credentials invalides", permission_error_model)
    @auth_ns.response(403, "Compte désactivé ou non autorisé", permission_error_model)
    @auth_ns.response(429, "Rate limit dépassé", api_error_model)
    @auth_ns.response(500, "Erreur serveur", api_error_model)
    # Limite d'appels pour éviter le brute force
    @limiter.limit("5 per minute")
    def post(self):
        """Authentifie un utilisateur et renvoie un token d'accès."""
        try:
            # Lot 1-D : login web (cookies) → Origin/Referer obligatoire.
            # Lot 1-E : login mobile (Bearer/JSON) → pas de contrôle Origin
            # (les clients natifs n'envoient souvent ni Origin ni Referer).
            if not _is_mobile_request():
                origin_ok, origin_err = validate_login_origin_for_web()
                if not origin_ok:
                    return {
                        "error": origin_err or "origin_not_allowed",
                        "message": "Origine non autorisée.",
                    }, 403
            return _login_post_body()
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR login: %s - %s", type(e).__name__, str(e))
            # ✅ Priorité 7: Audit logging pour erreur interne login
            try:
                data = request.get_json() or {}
                AuditLogger.log_action(
                    action_type="login_error",
                    action_category="security",
                    user_type="unknown",
                    result_status="failure",
                    result_message=f"Erreur interne: {type(e).__name__}",
                    action_details={
                        "email": mask_email(data.get("email", "")) if data else ""
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception:
                # Ignorer les erreurs d'audit logging
                # dans le gestionnaire d'erreurs
                pass
            return APIErrorHandler.handle_exception(
                Exception("Erreur lors de la connexion"),
                logger,
            )


# ========================
# Helper: résolution company_id pour JWT claims
# ========================
def _resolve_company_id(user: User) -> int | None:
    """Résout le company_id depuis la relation User -> Company.

    Le modèle User n'a pas de colonne company_id directe ;
    la relation est Company.user_id -> User.id.
    """
    company = getattr(user, "company", None)
    return company.id if company else None


# ========================
# Helpers: Bootstrap & Context switching (Unified Mobile Platform)
# ========================
BOOTSTRAP_RESPONSE_VERSION = "1.0.0"
CONTEXT_SWITCH_REDIS_PREFIX = "auth:active_context:"
CONTEXT_SWITCH_REDIS_TTL_SECONDS = 7 * 24 * 3600


def _bool_config(name: str, default: bool = False) -> bool:
    raw = current_app.config.get(name)
    if raw is None:
        raw = os.getenv(name)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _feature_flags_config() -> dict[str, object]:
    raw = current_app.config.get("MOBILE_FEATURE_FLAGS")
    if raw is None:
        raw = os.getenv("MOBILE_FEATURE_FLAGS", "{}")
    if isinstance(raw, dict):
        flags = dict(cast("dict[str, object]", raw))
    else:
        try:
            parsed = json.loads(str(raw))
        except Exception:
            parsed = {}
        flags = parsed if isinstance(parsed, dict) else {}

    if "saferpay_enabled" not in flags:
        try:
            from services.saferpay.config import saferpay_configured

            flags["saferpay_enabled"] = bool(saferpay_configured())
        except Exception:
            flags["saferpay_enabled"] = False

    try:
        from services.infrastructure.runtime_flags import (
            get_mobile_startup_runtime_flags,
        )

        flags.update(get_mobile_startup_runtime_flags())
    except Exception:
        flags.setdefault("ios_startup_fatal_recovery_disabled", False)

    return cast("dict[str, object]", flags)


def _account_status_value(user: User) -> str:
    status = getattr(user, "account_status", None)
    if hasattr(status, "value"):
        status = status.value
    return str(status or "").strip().lower()


def _normalized_account_status(user: User) -> str:
    status = _account_status_value(user)
    if status in {"disabled", "suspended"}:
        return "suspended"
    if status in {"pending_activation", "invited", "inactive"}:
        return "inactive"
    return "active"


def _onboarding_required(user: User) -> bool:
    status = _account_status_value(user)
    return status in {"pending_activation", "invited"}


def _onboarding_reasons(user: User) -> list[str]:
    reasons: list[str] = []
    if bool(getattr(user, "force_password_change", False)):
        reasons.append("force_password_change")
    status = _account_status_value(user)
    if status == "invited":
        reasons.append("invited")
    elif status == "pending_activation":
        reasons.append("pending_activation")
    # futur : if getattr(user, "must_accept_cgu", False): reasons.append("cgu")
    # futur : if getattr(user, "must_setup_mfa", False): reasons.append("mfa")
    return reasons


def _must_complete_onboarding(user: User) -> bool:
    return bool(_onboarding_reasons(user))


def _permissions_for_context(context_type: str) -> list[str]:
    if context_type == "client":
        return [
            "booking:create",
            "booking:read:self",
            "booking:history:read",
            "profile:read:self",
            "profile:update:self",
        ]
    if context_type == "driver":
        return [
            "mission:read",
            "mission:update_status",
            "mission:location:update",
            "notification:read",
            "chat:read",
        ]
    if context_type == "company":
        return [
            "company:dashboard:read",
            "company:rides:read",
            "company:notifications:read",
        ]
    if context_type == "institution":
        return [
            "institution:dashboard:read",
            "institution:requests:read",
            "institution:notifications:read",
        ]
    return []


def _company_allows_driver_workspace_switch(company: object | None) -> bool:
    """Entreprise transport : bascule mobile entreprise↔chauffeur (double casquette opérateur)."""
    if company is None:
        return False
    company_id = getattr(company, "id", None)
    return company_id is not None


def _allow_mobile_company_driver_context_switch(
    user: User, *, context_type: str, company: object | None
) -> bool:
    """Indique si ce contexte peut participer a la bascule securisée entreprise (mobile).

    Règles:
    - Un compte **chauffeur seul** (role=driver) ne doit jamais avoir le flag: pas d'accès
      a la gestion d'entreprise via /auth/switch-context. Les donnée entreprise sont
      confidentials.
    - Seul le **compte entreprise** (role=company), qui a aussi l'espace chauffeur rattaché,
      peut basculer entre `company` et `driver` (double casquette opérateur).
    - Les chauffeurs d'**urgence** (DriverType.EMERGENCY) qui doivent ouvrir l'app entreprise
      utilisent le flux dédié `POST /driver/me/switch-to-enterprise` (émission d'un jeton
      entreprise), pas cette bascule de contexte.
    - Toute entreprise liée au compte (indépendant de dispatch_mode / dispatch_enabled).
    """
    if user.role is not UserRole.COMPANY:
        return False
    if context_type not in ("company", "driver"):
        return False
    return _company_allows_driver_workspace_switch(company)


def _build_available_contexts(user: User) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []

    def add_context(
        *,
        context_id: str,
        context_type: str,
        label: str,
        organization_id: int | None,
        organization_name: str | None,
        is_default: bool = False,
        allow_mobile_context_switch: bool = False,
    ) -> None:
        if any(c.get("context_id") == context_id for c in contexts):
            return
        contexts.append(
            {
                "context_id": context_id,
                "context_type": context_type,
                "label": label,
                "organization_id": organization_id,
                "organization_name": organization_name,
                "permissions": _permissions_for_context(context_type),
                "is_default": is_default,
                "allow_mobile_context_switch": allow_mobile_context_switch,
            }
        )

    role = user.role
    if role == UserRole.CLIENT:
        add_context(
            context_id="client:self",
            context_type="client",
            label="Compte client",
            organization_id=None,
            organization_name=None,
            is_default=True,
        )
    elif role == UserRole.DRIVER:
        driver = getattr(user, "driver", None)
        drv_company = getattr(driver, "company", None) if driver else None
        add_context(
            context_id=f"driver:{driver.id if driver else user.public_id}",
            context_type="driver",
            label="Espace chauffeur",
            organization_id=getattr(driver, "company_id", None),
            organization_name=getattr(drv_company, "name", None),
            is_default=True,
            allow_mobile_context_switch=_allow_mobile_company_driver_context_switch(
                user, context_type="driver", company=drv_company
            ),
        )
    elif role == UserRole.COMPANY:
        company = getattr(user, "company", None)
        add_context(
            context_id=f"company:{company.id if company else user.public_id}",
            context_type="company",
            label="Espace entreprise",
            organization_id=getattr(company, "id", None),
            organization_name=getattr(company, "name", None),
            is_default=True,
            allow_mobile_context_switch=_allow_mobile_company_driver_context_switch(
                user, context_type="company", company=company
            ),
        )
    elif role == UserRole.INSTITUTION:
        institution_id = cast("int | None", getattr(user, "institution_id", None))
        add_context(
            context_id=f"institution:{institution_id if institution_id else user.public_id}",
            context_type="institution",
            label="Espace institution",
            organization_id=institution_id,
            organization_name=None,
            is_default=True,
        )

    # Contextes additionnels (multi-contexte)
    if getattr(user, "clients", None):
        add_context(
            context_id="client:self",
            context_type="client",
            label="Compte client",
            organization_id=None,
            organization_name=None,
        )
    if getattr(user, "driver", None):
        drv = user.driver
        d_company = getattr(drv, "company", None)
        add_context(
            context_id=f"driver:{drv.id}",
            context_type="driver",
            label="Espace chauffeur",
            organization_id=getattr(drv, "company_id", None),
            organization_name=getattr(d_company, "name", None),
            allow_mobile_context_switch=_allow_mobile_company_driver_context_switch(
                user, context_type="driver", company=d_company
            ),
        )
    if getattr(user, "company", None):
        cmp = user.company
        add_context(
            context_id=f"company:{cmp.id}",
            context_type="company",
            label="Espace entreprise",
            organization_id=cmp.id,
            organization_name=cmp.name,
            allow_mobile_context_switch=_allow_mobile_company_driver_context_switch(
                user, context_type="company", company=cmp
            ),
        )
    institution_id = cast("int | None", getattr(user, "institution_id", None))
    if institution_id:
        add_context(
            context_id=f"institution:{institution_id}",
            context_type="institution",
            label="Espace institution",
            organization_id=institution_id,
            organization_name=None,
        )
    return contexts


def _context_by_id(
    contexts: list[dict[str, object]], context_id: str | None
) -> dict[str, object] | None:
    if not context_id:
        return None
    for c in contexts:
        if c.get("context_id") == context_id:
            return c
    return None


def _is_company_driver_cross_context_switch(
    a: dict[str, object] | None, b: dict[str, object] | None
) -> bool:
    if not a or not b:
        return False
    ta, tb = a.get("context_type"), b.get("context_type")
    return (ta == "company" and tb == "driver") or (ta == "driver" and tb == "company")


def _load_user_for_bootstrap(public_id: str) -> User | None:
    return (
        User.query.options(
            joinedload(cast("Any", User.driver)).joinedload(Driver.company),
            joinedload(cast("Any", User.clients)),
            joinedload(cast("Any", User.company)),
        )
        .filter_by(public_id=public_id)
        .first()
    )


def _prepare_user_for_bootstrap(user: User) -> User:
    """Provisionne la fiche chauffeur entreprise si besoin, puis recharge le user ORM."""
    from application.companies.drivers.ensure_company_operator_driver import (
        EnsureCompanyOperatorDriverUseCase,
    )

    result = EnsureCompanyOperatorDriverUseCase().execute(user)
    if result.created:
        db.session.commit()
        reloaded = _load_user_for_bootstrap(str(user.public_id))
        if reloaded is not None:
            return reloaded
    return user


def _redis_active_context_key(user_public_id: str) -> str:
    return f"{CONTEXT_SWITCH_REDIS_PREFIX}{user_public_id}"


def _get_saved_active_context(user_public_id: str) -> str | None:
    if not redis_client:
        return None
    value = redis_client.get(_redis_active_context_key(user_public_id))
    if not value:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _save_active_context(user_public_id: str, context_id: str) -> None:
    if not redis_client:
        return
    redis_client.setex(
        _redis_active_context_key(user_public_id),
        CONTEXT_SWITCH_REDIS_TTL_SECONDS,
        context_id,
    )


def _default_context_id(contexts: list[dict[str, object]]) -> str | None:
    for context in contexts:
        if context.get("is_default"):
            return cast("str", context.get("context_id"))
    if contexts:
        return cast("str", contexts[0].get("context_id"))
    return None


def _resolve_active_context_id(
    *,
    contexts: list[dict[str, object]],
    preferred_context_id: str | None,
) -> str | None:
    context_ids = {
        cast("str", context.get("context_id"))
        for context in contexts
        if context.get("context_id")
    }
    if preferred_context_id and preferred_context_id in context_ids:
        return preferred_context_id
    return _default_context_id(contexts)


def _bootstrap_base_response(
    *,
    request_id: str,
    is_authenticated: bool,
) -> dict[str, object]:
    return {
        "bootstrap_version": BOOTSTRAP_RESPONSE_VERSION,
        "is_authenticated": is_authenticated,
        "user": None,
        "account_status": "inactive",
        "onboarding_status": {"required": False},
        "available_contexts": [],
        "active_context_id": None,
        "feature_flags": _feature_flags_config(),
        "min_supported_app_version": str(
            current_app.config.get(
                "MOBILE_MIN_SUPPORTED_APP_VERSION",
                os.getenv("MOBILE_MIN_SUPPORTED_APP_VERSION", "0.1.0"),
            )
        ),
        "maintenance_mode": _bool_config("MAINTENANCE_MODE", False),
        "degraded_mode": _bool_config("DEGRADED_MODE", False),
        "server_time": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "status_dictionary_version": STATUS_DICTIONARY_VERSION,
        "pricing_contract_version": PRICING_CONTRACT_VERSION,
        "canonical_address_contract_version": CANONICAL_ADDRESS_CONTRACT_VERSION,
        "preview_contract_version": PREVIEW_CONTRACT_VERSION,
        "mission_status_version": MISSION_STATUS_VERSION,
        "mission_snapshot_version": MISSION_SNAPSHOT_VERSION,
        "driver_socket_contract_version": DRIVER_SOCKET_CONTRACT_VERSION,
        "driver_tracking_contract_version": DRIVER_TRACKING_CONTRACT_VERSION,
    }


# ========================
# Helper Functions pour Refresh Token
# ========================
def _get_password_hash_version(user: User) -> str:
    """Génère un hash de version basé sur le hash du mot de passe.

    Utilisé pour invalider automatiquement les refresh tokens
    après un changement de mot de passe sans stockage server-side.

    Args:
        user: L'utilisateur dont on veut obtenir la version du mot de passe

    Returns:
        Hash SHA256 des 16 premiers caractères du hash du mot de passe
    """
    password_hash = getattr(user, "password", "")
    if not password_hash:
        return ""
    # Prendre les premiers caractères du hash pour créer une version stable
    # Utiliser SHA256 pour créer un hash plus court et sécurisé
    hash_snippet = (
        str(password_hash)[:PASSWORD_HASH_VERSION_LENGTH]
        if len(str(password_hash)) > PASSWORD_HASH_VERSION_LENGTH
        else str(password_hash)
    )
    return hashlib.sha256(hash_snippet.encode()).hexdigest()[
        :PASSWORD_HASH_VERSION_LENGTH
    ]


def _check_user_profile_active(user: User) -> tuple[bool, str | None]:
    """Vérifie si le profil associé à l'utilisateur est actif.

    Args:
        user: L'utilisateur à vérifier

    Returns:
        Tuple (is_active, error_message):
        - Si actif: (True, None)
        - Si inactif: (False, "Compte désactivé")
        - Si pas de profil: (True, None) - on considère comme actif par défaut
    """
    if getattr(user, "account_status", None) == "pending_activation":
        return False, "Compte en attente de validation email/SMS."

    if user.role == UserRole.driver and user.driver and not user.driver.is_active:
        return False, "Compte désactivé"

    if user.role == UserRole.client and user.clients:
        # Un utilisateur peut avoir plusieurs clients (1-N)
        # On vérifie qu'au moins un client est actif
        active_clients = [c for c in user.clients if c.is_active]
        if not active_clients:
            return False, "Compte désactivé"

    # Institution: vérifier account_status
    if user.role == UserRole.INSTITUTION or user.institution_id:
        if getattr(user, "archived_at", None):
            return False, "Compte archivé"
        if getattr(user, "account_status", None) == "disabled":
            return False, "Compte désactivé"
        if getattr(user, "account_status", None) == "invited":
            return False, "Compte non encore activé. Vérifiez votre email d'invitation."
        password_expires_at = getattr(user, "password_expires_at", None)
        if password_expires_at and password_expires_at.tzinfo is None:
            password_expires_at = password_expires_at.replace(tzinfo=UTC)
        if (
            getattr(user, "force_password_change", False)
            and password_expires_at
            and password_expires_at < datetime.now(UTC)
        ):
            return (
                False,
                "Mot de passe temporaire expiré. Contactez votre administrateur.",
            )

    # Comptes demo: validité stricte alignée sur la fenêtre d'accès démo.
    demo_valid, demo_error = enforce_demo_user_access_validity(user)
    if not demo_valid:
        return False, demo_error or "Accès démo expiré."

    # Pour les autres rôles (admin, company) ou si pas de profil, on considère comme actif
    return True, None


def _resolve_max_active_refresh_tokens(user: User) -> int:
    """Limite de sessions refresh actives selon le rôle utilisateur."""
    role = user.role
    if role == UserRole.driver:
        return int(os.getenv("MAX_ACTIVE_REFRESH_TOKENS_DRIVER", "15"))
    if role in (UserRole.company, UserRole.institution):
        return int(os.getenv("MAX_ACTIVE_REFRESH_TOKENS_COMPANY", "15"))
    return int(os.getenv("MAX_ACTIVE_REFRESH_TOKENS", "5"))


def _validate_refresh_token(
    refresh_token: str,
) -> tuple[str | None, dict[str, str] | None]:
    """Valide un refresh token et retourne l'user_public_id ou une erreur.

    Vérifie aussi que le mot de passe n'a pas changé depuis l'émission du token
    et que le token n'est pas révoqué dans la DB.

    Args:
        refresh_token: Le token JWT à valider

    Returns:
        Tuple (user_public_id, error_response):
        - Si valide: (user_public_id, None)
        - Si invalide: (None, {error: "...", status_code: 401})
    """
    try:
        decoded = decode_token(refresh_token, allow_expired=False)
        user_public_id = decoded.get("sub")

        # Vérifier que c'est bien un refresh token (pas un access token)
        # Les refresh tokens n'ont pas les claims "role" ou "company_id"
        token_type = decoded.get("type", "")
        is_access_token = "role" in decoded or "company_id" in decoded

        if token_type != "refresh" and is_access_token:
            error_response, _ = APIErrorHandler.handle_validation_error(
                "Le token fourni n'est pas un refresh token",
                logger_instance=logger,
            )
            return None, error_response

        if not user_public_id:
            error_response, _ = APIErrorHandler.handle_validation_error(
                "Refresh token invalide (identity manquante)",
                logger_instance=logger,
            )
            return None, error_response

        # ✅ SECURITY: Vérifier que le mot de passe n'a pas changé
        # Récupérer l'utilisateur pour vérifier le hash du mot de passe
        user_dto = user_repo.find_by_public_id(user_public_id)
        if user_dto:
            # Récupérer le modèle User pour accéder aux méthodes de hash
            user = user_repo.find_model_by_public_id(user_public_id)
            if user:
                # Vérifier si le token a un claim pwd_hash (nouveaux tokens)
                token_pwd_hash = decoded.get("pwd_hash")
                if token_pwd_hash:
                    current_pwd_hash = _get_password_hash_version(user)
                    if token_pwd_hash != current_pwd_hash:
                        logger.warning(
                            "Refresh token rejeté : mot de passe modifié pour user %s",
                            user_public_id,
                        )
                        return None, {
                            "error": "Refresh token invalide (mot de passe modifié)"
                        }

        # Lot 1-C : validation Redis (fail-closed si indisponible)
        if refresh_fail_closed_enabled() and user_dto:
            try:
                user_model = user_repo.find_model_by_public_id(user_public_id)
                if user_model:
                    redis_svc = RefreshTokenService()
                    if not redis_svc.is_token_valid(
                        refresh_token, user_id=user_model.id
                    ):
                        return None, {
                            "error": "Refresh token révoqué ou absent du store"
                        }
            except RefreshStoreUnavailableError:
                return None, {
                    "error": "service_unavailable",
                    "message": "Store refresh indisponible",
                    "_http_status": 503,
                }

        # ✅ SECURITY: Vérifier si le token est révoqué dans la DB (Phase 2)
        # grace_window=True : rotation soft (5 min) + réutilisation legacy acceptée
        try:
            request_device_id = request.headers.get("X-Device-ID") if request else None
            if is_token_revoked(
                refresh_token,
                grace_window=True,
                request_device_id=request_device_id,
            ):
                logger.warning(
                    "Refresh token rejeté : token révoqué pour user %s",
                    user_public_id,
                )
                error_response, _ = APIErrorHandler.handle_permission_error(
                    "Refresh token révoqué",
                    logger_instance=logger,
                )
                return None, error_response
        except RefreshStoreUnavailableError:
            return None, {
                "error": "service_unavailable",
                "message": "Store refresh indisponible",
                "_http_status": 503,
            }
        except Exception as revoke_check_error:
            if refresh_fail_closed_enabled():
                logger.error(
                    "Erreur vérification révocation token (fail-closed): %s",
                    str(revoke_check_error),
                )
                return None, {
                    "error": "service_unavailable",
                    "message": "Store refresh indisponible",
                    "_http_status": 503,
                }
            logger.debug(
                "Erreur vérification révocation token (ignorée): %s",
                str(revoke_check_error),
            )

        # ✅ SECURITY: Mettre à jour la date de dernière utilisation
        try:
            update_token_last_used(refresh_token)
        except Exception as update_error:
            # Ne pas bloquer le refresh si la mise à jour échoue
            logger.debug(
                "Erreur mise à jour last_used_at (ignorée): %s", str(update_error)
            )

        return user_public_id, None

    except Exception as decode_error:
        logger.warning("Erreur décodage refresh token: %s", str(decode_error))
        error_response, _ = APIErrorHandler.handle_validation_error(
            "Refresh token invalide ou expiré",
            logger_instance=logger,
        )
        return None, error_response


# ========================
# 2. Refresh Token
# ========================
# Modèle Swagger pour la requête de refresh token
refresh_token_request_model = auth_ns.model(
    "RefreshTokenRequest",
    {
        "refresh_token": fields.String(
            required=False,
            description="Refresh token JWT (optionnel si fourni dans Authorization header)",
        ),
    },
)

# Modèle Swagger pour la réponse de refresh token
refresh_token_response_model = auth_ns.model(
    "RefreshTokenResponse",
    {
        "access_token": fields.String(
            required=True, description="Nouveau token d'accès JWT"
        ),
        "refresh_token": fields.String(
            required=True, description="Nouveau refresh token (rotation automatique)"
        ),
        "user": fields.Nested(
            auth_ns.model(
                "RefreshTokenUserInfo",
                {
                    "public_id": fields.String(
                        required=True, description="ID public de l'utilisateur"
                    ),
                    "role": fields.String(
                        required=True, description="Rôle de l'utilisateur"
                    ),
                    "company_id": fields.Integer(
                        required=False, description="ID de la compagnie (si applicable)"
                    ),
                    "driver_id": fields.Integer(
                        required=False, description="ID du chauffeur (si applicable)"
                    ),
                },
            ),
            required=True,
            description="Informations minimales de l'utilisateur",
        ),
        "trace_id": fields.String(
            required=False,
            description="ID de traçage pour le support",
            example="a1b2c3d4",
        ),
    },
)


# ========================
# Login Test (sans CSRF pour tests de charge)
# ========================
@auth_ns.route("/login-test")
class LoginTest(Resource):
    """Endpoint de login pour tests de charge (dev/test uniquement, sans CSRF)."""

    @auth_ns.expect(login_model)
    @auth_ns.response(200, "Connexion réussie", login_success_model)
    @auth_ns.response(401, "Credentials invalides")
    @auth_ns.response(403, "Endpoint disponible uniquement en dev/test")
    def post(self):
        """Authentifie un utilisateur sans vérification CSRF (tests uniquement)."""
        import os

        from flask import abort, current_app, request
        from flask_jwt_extended import (
            create_access_token,
            create_refresh_token,
        )

        from models.user import User

        # ⚠️ Sécurité : Disponible uniquement en environnement dev/test
        flask_env = os.getenv("FLASK_ENV", "production")
        if flask_env not in ["development", "testing"]:
            abort(
                403,
                description="Endpoint disponible uniquement en environnement dev/test",
            )

        # Récupérer les données
        data = request.get_json()
        if not data:
            return auth_error(
                "invalid_request",
                "Pas de données JSON fournies",
                400,
            )

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return auth_error(
                "invalid_request",
                "Email et mot de passe requis",
                400,
            )

        # Chercher l'utilisateur
        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            return auth_error(
                AuthErrorCodes.INVALID_CREDENTIALS,
                "Identifiants invalides",
                401,
            )

        # Note: Pas de vérification is_active car c'est un endpoint de test simplifié

        # Générer les tokens JWT avec les claims nécessaires (notamment "aud" pour validation audience)
        claims = {
            "user_id": user.id,  # ⚠️ ID numérique attendu par dispatch_routes
            "role": user.role.value,
            "company_id": _resolve_company_id(user),
            "driver_id": getattr(user, "driver_id", None),
            "institution_id": getattr(user, "institution_id", None),
            "institution_role": getattr(user, "institution_role", None),
            "aud": "atmr-api",  # ✅ Audience claim pour passer validation JWT
            "token_version": _user_token_version(user),
        }
        access_token = create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
            expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
            fresh=True,  # Token fresh pour tests
        )
        refresh_token = create_refresh_token(identity=str(user.public_id))

        # Retourner la réponse (format simplifié pour tests)
        return {
            "message": "Connexion réussie",
            "token": access_token,
            "access_token": access_token,  # Compatibilité
            "refresh_token": refresh_token,
            "user": {
                "id": user.id,
                "public_id": user.public_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            },
        }, 200


@auth_ns.route("/refresh-token")
class RefreshToken(Resource):
    @auth_ns.expect(refresh_token_request_model)
    @auth_ns.response(200, "Token rafraîchi avec succès", refresh_token_response_model)
    @auth_ns.response(400, "Requête invalide", validation_error_model)
    @auth_ns.response(401, "Refresh token invalide ou expiré", permission_error_model)
    @auth_ns.response(403, "Compte désactivé", permission_error_model)
    @auth_ns.response(404, "Utilisateur non trouvé", not_found_error_model)
    @auth_ns.response(429, "Rate limit dépassé", api_error_model)
    @auth_ns.response(500, "Erreur interne", api_error_model)
    # ✅ S2: Rate limiting plus strict pour refresh token (protection contre abus)
    @limiter.limit("20 per minute")
    def post(self):
        """Rafraîchit l'access token à partir d'un refresh token.

        Accepte le refresh_token en body (JSON) ou dans le header Authorization Bearer.

        ✅ Rotation automatique : Génère toujours un nouveau refresh_token et révoque l'ancien.
        Retourne un nouveau access_token, un nouveau refresh_token et les informations
        minimales de l'utilisateur.
        """
        try:
            # Lot 1-E : web = cookie only ; mobile = JSON body only (cookies ignorés)
            refresh_token = None
            is_mobile_request = _is_mobile_request()
            refresh_token_from_cookie = False

            if is_mobile_request:
                data = request.get_json(silent=True) or {}
                refresh_token = (
                    data.get("refresh_token")
                    or data.get("refreshToken")
                    or data.get("token")
                )
            else:
                refresh_token = request.cookies.get(
                    current_app.config["COOKIE_REFRESH_TOKEN_NAME"]
                )
                refresh_token_from_cookie = bool(refresh_token)

            # 3. Validation : refresh_token requis
            if not refresh_token:
                trace_id = get_trace_id()
                logger.warning(
                    "Refresh token missing - trace_id: %s",
                    trace_id,
                )
                logger.info(
                    "auth_refresh_failure",
                    extra={
                        "event": "auth_refresh_failure",
                        "cause": "missing_refresh_token",
                        "device_id": request.headers.get("X-Device-ID"),
                        "session_diag": request.headers.get("X-Session-Diag"),
                        "trace_id": trace_id,
                    },
                )
                # Mobile sans body / web sans cookie → 401 (fail-closed), pas 400
                return {
                    "error": "refresh_token_required",
                    "trace_id": trace_id,
                }, 401

            # 4. Valider le refresh token (inclut vérification révocation, pwd_hash, etc.)
            user_public_id, error_response = _validate_refresh_token(refresh_token)
            if error_response or not user_public_id:
                trace_id = get_trace_id()
                http_status = 401
                if error_response and error_response.pop("_http_status", None) == 503:
                    http_status = 503
                logger.warning(
                    "Refresh token invalid - trace_id: %s status=%s",
                    trace_id,
                    http_status,
                )
                logger.info(
                    "auth_refresh_failure",
                    extra={
                        "event": "auth_refresh_failure",
                        "cause": (
                            "store_unavailable"
                            if http_status == 503
                            else "invalid_or_expired"
                        ),
                        "device_id": request.headers.get("X-Device-ID"),
                        "session_diag": request.headers.get("X-Session-Diag"),
                        "trace_id": trace_id,
                    },
                )
                if error_response:
                    error_response["trace_id"] = trace_id
                    return error_response, http_status
                return {
                    "error": "Refresh token invalide",
                    "trace_id": trace_id,
                }, 401

            # 5. Vérifier que l'utilisateur existe
            user_dto = user_repo.find_by_public_id(user_public_id)
            if not user_dto:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    user_public_id,
                    logger,
                )

            # ✅ SECURITY: Vérifier que le profil (Driver/Client) est actif
            # Récupérer le modèle User pour accéder aux méthodes de profil
            user = user_repo.find_model_by_public_id(user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    user_public_id,
                    logger,
                )
            is_active, error_message = _check_user_profile_active(user)
            if not is_active:
                trace_id = get_trace_id()
                logger.warning(
                    "Refresh token rejeté : compte désactivé pour user %s (role: %s)",
                    user_public_id,
                    user.role.value if user.role else "unknown",
                )
                logger.info(
                    "auth_refresh_failure",
                    extra={
                        "event": "auth_refresh_failure",
                        "cause": "account_disabled",
                        "user_public_id": user_public_id,
                        "device_id": request.headers.get("X-Device-ID"),
                        "session_diag": request.headers.get("X-Session-Diag"),
                        "trace_id": trace_id,
                    },
                )
                return {
                    "error": error_message or "Compte désactivé",
                    "reason": "account_disabled",
                    "trace_id": trace_id,
                }, 403

            # ✅ SECURITY: Ajout claim 'aud' (audience) pour prévenir token replay
            claims = {
                "role": user.role.value,
                "company_id": _resolve_company_id(user),
                "driver_id": getattr(user, "driver_id", None),
                "institution_id": getattr(user, "institution_id", None),
                "institution_role": getattr(user, "institution_role", None),
                "aud": "atmr-api",  # Audience claim pour sécurité
                "token_version": _user_token_version(user),
            }

            # 6. Générer nouveau access_token
            access_expires_delta = _resolve_access_token_expires(is_mobile_request)
            new_access_token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
                expires_delta=access_expires_delta,
            )

            # 7. ✅ ROTATION AUTOMATIQUE : Générer toujours un nouveau refresh_token
            # Conserver la politique remember_me d'origine (false→false, true→true).
            pwd_hash_version = _get_password_hash_version(user)
            remember_me = _resolve_remember_me_from_refresh_token(
                refresh_token,
                is_mobile_request=is_mobile_request,
            )
            refresh_expires_delta = _resolve_refresh_token_expires(
                is_mobile_request=is_mobile_request,
                remember_me=remember_me,
            )
            new_refresh_claims: dict[str, object] = {
                "aud": "atmr-api",
                "pwd_hash": pwd_hash_version,
                "token_version": _user_token_version(user),
            }
            if not is_mobile_request:
                new_refresh_claims["remember_me"] = remember_me
            new_refresh_token = create_refresh_token(
                identity=str(user.public_id),
                additional_claims=new_refresh_claims,
                expires_delta=refresh_expires_delta,
            )

            # ✅ PHASE 2: Utiliser RefreshTokenService pour rotation et limitation
            token_service = RefreshTokenService()

            # Mettre a jour le score ZSET Redis pour que la purge evince par last_used_at
            with suppress(Exception):
                token_service.touch_token_score(user.id, refresh_token)

            # ✅ ROTATION SOFT : marquer l'ancien token comme rotate (pas revoque)
            # L'ancien reste valide tant que le nouveau n'a pas ete utilise par le client.
            # Cela evite les pertes de session si le mobile crash avant de sauvegarder le nouveau.
            try:
                mark_token_rotated(refresh_token, new_refresh_token)

                try:
                    from security.security_metrics import tokens_rotation_total

                    tokens_rotation_total.inc()
                except Exception:
                    pass
            except Exception as rotate_error:
                logger.warning(
                    "Soft rotation marking failed (non-blocking): %s",
                    str(rotate_error),
                )

            # ✅ SECURITY: Stocker le nouveau token dans Redis et DB (fail-closed)
            try:
                token_service.store_token(
                    user.id,
                    new_refresh_token,
                    ttl_seconds=int(refresh_expires_delta.total_seconds()),
                )

                refresh_expires_at = datetime.now(UTC) + refresh_expires_delta
                device_id = request.headers.get("X-Device-ID")
                device_name = request.headers.get("X-Device-Name")
                if device_id:
                    revoke_active_tokens_for_device(
                        user.id,
                        device_id,
                        reason="Remplacé par rotation refresh (même appareil)",
                    )
                store_refresh_token(
                    token=new_refresh_token,
                    user_id=user.id,
                    expires_at=refresh_expires_at,
                    device_id=device_id,
                    device_name=device_name,
                )

                max_active_tokens = _resolve_max_active_refresh_tokens(user)
                token_service.limit_active_tokens(user.id, max_active_tokens)
            except Exception as store_error:
                logger.error(
                    "Soft rotation storage failed: %s",
                    str(store_error),
                )
                if refresh_fail_closed_enabled() and not current_app.config.get(
                    "TESTING"
                ):
                    return {
                        "error": "service_unavailable",
                        "message": "Stockage session indisponible. Réessayez.",
                    }, 503

            # 8. ✅ Priorité 7: Audit logging pour token refresh
            try:
                AuditLogger.log_action(
                    action_type="token_refresh",
                    action_category="security",
                    user_id=user.id,
                    user_type=user.role.value if user.role else "unknown",
                    result_status="success",
                    action_details={
                        "rotation_automatic": True,
                        "token_source": (
                            "cookie"
                            if not is_mobile_request
                            and request.cookies.get(
                                current_app.config["COOKIE_REFRESH_TOKEN_NAME"]
                            )
                            else "body"
                            if (request.get_json() or {}).get("refresh_token")
                            else "header"
                        ),
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métrique Prometheus pour token refresh
                security_token_refreshes_total.inc()
            except Exception as audit_error:
                logger.warning("Échec audit logging token_refresh: %s", audit_error)

            # 9. ✅ Migration localStorage → cookies httpOnly
            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "Token refresh success - user_id: %s, trace_id: %s",
                user.id,
                trace_id,
            )

            # Construire la réponse JSON
            response_data = {
                "user": {
                    "public_id": user.public_id,
                    "role": user.role.value,
                    "company_id": _resolve_company_id(user),
                    "driver_id": getattr(user, "driver_id", None),
                },
                "trace_id": trace_id,
            }

            # Lot 1-E : tokens JSON uniquement pour mobile ; web = cookies only
            if is_mobile_request:
                response_data["access_token"] = new_access_token
                response_data["refresh_token"] = new_refresh_token
                response_data["token_type"] = "Bearer"
                response_data["expires_in"] = int(access_expires_delta.total_seconds())

            # Créer la réponse avec make_response pour pouvoir définir les cookies
            response = make_response(response_data, 200)

            # ✅ Définir cookies httpOnly pour web (pas pour mobile)
            if not is_mobile_request and refresh_token_from_cookie:
                # Cookie access_token
                response.set_cookie(
                    current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
                    new_access_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=int(
                        current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()
                    ),
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )

                # Cookie refresh_token (rotation automatique, politique remember_me conservée)
                refresh_cookie_max_age = _refresh_cookie_max_age(
                    remember_me=remember_me,
                    refresh_expires_delta=refresh_expires_delta,
                )
                response.set_cookie(
                    current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
                    new_refresh_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=refresh_cookie_max_age,
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )

            return response

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 3. Obtenir un token "fresh"
# ========================
@auth_ns.route("/fresh-token")
class FreshToken(Resource):
    @auth_ns.expect(fresh_token_request_model)
    @auth_ns.response(200, "Token fresh obtenu avec succès", fresh_token_response_model)
    @auth_ns.response(401, "Mot de passe incorrect ou utilisateur non authentifié")
    @jwt_required()  # Nécessite un token valide (mais pas fresh)
    @limiter.limit("5 per minute")  # Protection contre brute force
    def post(self):
        """Obtient un token 'fresh' en vérifiant le mot de passe de l'utilisateur.

        Permet d'obtenir un token 'fresh' sans se déconnecter complètement.
        Utile pour effectuer des actions sensibles qui nécessitent un token fresh.
        """
        try:
            # 1. Récupérer l'utilisateur actuel
            user_public_id = get_jwt_identity()
            if not user_public_id:
                return auth_error(
                    AuthErrorCodes.MISSING_TOKEN,
                    "Utilisateur non authentifié",
                    401,
                )

            user_dto = user_repo.find_by_public_id(user_public_id)
            if not user_dto or not user_dto.email:
                return auth_error(
                    AuthErrorCodes.INVALID_CREDENTIALS,
                    "Utilisateur non trouvé",
                    401,
                )

            user = user_repo.find_model_by_email(user_dto.email)
            if not user:
                return auth_error(
                    AuthErrorCodes.INVALID_CREDENTIALS,
                    "Utilisateur non trouvé",
                    401,
                )

            # 2. Récupérer le mot de passe depuis la requête
            data = request.get_json(silent=True) or {}
            password = data.get("password")
            if not password:
                return auth_error(
                    "invalid_request",
                    "Mot de passe requis",
                    400,
                )

            # 3. Vérifier le mot de passe
            if not user.check_password(password):
                logger.warning(
                    "[Auth] Échec vérification mot de passe pour fresh token (user: %s)",
                    user_public_id,
                )
                return auth_error(
                    AuthErrorCodes.INVALID_CREDENTIALS,
                    "Mot de passe incorrect",
                    401,
                )

            # 4. Créer un token fresh
            claims = {
                "role": user.role.value
                if hasattr(user.role, "value")
                else str(user.role),
                "company_id": _resolve_company_id(user),
                "driver_id": getattr(user, "driver_id", None),
                "institution_id": getattr(user, "institution_id", None),
                "institution_role": getattr(user, "institution_role", None),
                "aud": "atmr-api",
                "token_version": _user_token_version(user),
            }
            fresh_token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
                expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
                fresh=True,  # ✅ Token fresh
            )

            logger.info(
                "[Auth] Token fresh généré pour user %s (public_id: %s)",
                user.id,
                user_public_id,
            )

            # 5. Retourner le token
            response_data = {
                "access_token": fresh_token,
                "message": "Token fresh obtenu avec succès",
            }

            # Si on utilise des cookies, mettre à jour le cookie
            is_mobile_request = request.headers.get("X-Requested-With") == "Expo"
            if not is_mobile_request:
                response = make_response(response_data)
                response.set_cookie(
                    current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
                    fresh_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=int(
                        current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()
                    ),
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )
                return response

            return response_data, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Auth] Erreur lors de la génération du token fresh: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 4. Logout / Révoquer Token
# ========================
@auth_ns.route("/logout")
class Logout(Resource):
    @auth_ns.doc(
        description=(
            "Révoque le token JWT actuel et l'ajoute à la blacklist. "
            "Après la déconnexion, le token ne pourra plus être utilisé "
            "pour accéder aux endpoints protégés."
        ),
        summary="Déconnexion utilisateur",
    )
    @auth_ns.response(200, "Déconnexion réussie", logout_response_model)
    @auth_ns.response(401, "Token manquant ou invalide")
    @auth_ns.response(500, "Erreur lors de la révocation du token", logout_error_model)
    @jwt_required(optional=True)
    def post(self):
        """Révoque le token JWT actuel (logout) et efface toujours les cookies web."""
        is_mobile_request = _is_mobile_request()
        try:
            from security.token_blacklist import revoke_token

            # ✅ Priorité 7: Récupérer user_id pour audit logging
            current_user_id = get_jwt_identity()
            user = None
            driver_id_for_log = None
            if current_user_id:
                user = user_repo.find_by_public_id(current_user_id)

            # ✅ Migration localStorage → cookies httpOnly
            # Récupérer le refresh token depuis cookie (priorité), body ou header
            refresh_token = None

            # Priorité 1 : Cookie (pour web)
            if not is_mobile_request:
                refresh_token = request.cookies.get(
                    current_app.config["COOKIE_REFRESH_TOKEN_NAME"]
                )

            # Priorité 2 : Body JSON (pour mobile ou fallback)
            if not refresh_token:
                data = request.get_json(silent=True) or {}
                refresh_token = (
                    data.get("refresh_token")
                    or data.get("refreshToken")
                    or data.get("token")
                )

            # Priorité 3 : Header Authorization (rétrocompatibilité)
            if not refresh_token:
                auth_header = request.headers.get("Authorization", "")
                if auth_header and auth_header.startswith("Bearer "):
                    refresh_token = auth_header.split(" ", 1)[1].strip()

            # ✅ PHASE 2: Révoquer tous les tokens via RefreshTokenService
            token_service = RefreshTokenService()
            try:
                if user:
                    # Révoquer tous les tokens de l'utilisateur dans Redis
                    token_service.revoke_all_user_tokens(user.id)

                    # Révoquer aussi dans la DB pour compatibilité
                    revoke_all_user_tokens(user.id, reason="Logout utilisateur")

                # Si un refresh token spécifique est présent, le révoquer aussi
                if refresh_token:
                    token_service.revoke_token(refresh_token)
                    revoke_refresh_token(refresh_token, reason="Logout utilisateur")
                    logger.debug(
                        "Refresh token révoqué lors du logout pour user %s",
                        current_user_id,
                    )
            except Exception as revoke_error:
                # Ne pas bloquer le logout si la révocation échoue
                logger.warning(
                    "Échec révocation refresh tokens lors logout (ignoré): %s",
                    str(revoke_error),
                )

            # Invalidation push ciblée par device_id (multi-appareils : iPhone ≠ iPad)
            try:
                logout_body = request.get_json(silent=True) or {}
                device_id = (
                    logout_body.get("device_id")
                    or logout_body.get("deviceId")
                    or request.headers.get("X-Device-ID")
                )
                if device_id:
                    device_id = str(device_id).strip()
                if user and device_id:
                    from application.notifications.upsert_device_token import (
                        deactivate_device_tokens_for_logout,
                    )

                    tokens_invalidated = 0
                    if user.role == UserRole.driver:
                        from repositories.driver_repository import DriverRepository

                        driver_repo = DriverRepository()
                        driver = driver_repo.find_model_by_user_id(user.id)
                        driver_id_for_log = driver.id if driver else None
                        if driver:
                            tokens_invalidated = deactivate_device_tokens_for_logout(
                                driver_id=driver.id,
                                device_id=device_id,
                            )
                    elif user.role == UserRole.company and user.company:
                        tokens_invalidated = deactivate_device_tokens_for_logout(
                            company_id=int(user.company.id),
                            device_id=device_id,
                        )

                    if tokens_invalidated > 0:
                        db.session.commit()
                        logger.info(
                            "[logout] %d token(s) push invalidé(s) device_id=%s user=%s",
                            tokens_invalidated,
                            device_id[:8] + "…" if len(device_id) > 8 else device_id,
                            current_user_id,
                        )
                        try:
                            from services.monitoring.prometheus import (
                                track_push_token_invalidated,
                            )

                            for _ in range(tokens_invalidated):
                                track_push_token_invalidated(reason="logout")
                        except ImportError:
                            pass
                    else:
                        logger.debug(
                            "[logout] Aucun token push actif pour device_id sur ce compte",
                        )
                elif user and user.role == UserRole.driver:
                    logger.debug(
                        "[logout] device_id absent — pas d'invalidation push (multi-device)",
                    )
            except Exception as device_token_error:
                logger.warning(
                    "Échec invalidation tokens push lors logout (ignoré): %s",
                    str(device_token_error),
                )

            # ✅ PHASE 3: Révoquer l'access token actuel (optionnel mais recommandé)
            try:
                from datetime import UTC, datetime

                from services.security.authentication import AccessTokenService

                access_token_service = AccessTokenService()
                jwt_claims = get_jwt()

                # Obtenir le jti (JWT ID) et l'expiration
                token_jti = jwt_claims.get("jti")
                exp = jwt_claims.get("exp")

                if token_jti and exp:
                    # Calculer le temps restant avant expiration
                    now = datetime.now(UTC).timestamp()
                    ttl = int(exp - now)

                    if ttl > 0:
                        access_token_service.revoke_token(token_jti, ttl)
                        logger.debug(
                            "Access token révoqué lors du logout pour user %s, jti=%s",
                            current_user_id,
                            token_jti,
                        )
            except Exception as access_revoke_error:
                # Ne pas bloquer le logout si la révocation de l'access token échoue
                logger.warning(
                    "Échec révocation access token lors logout (ignoré): %s",
                    str(access_revoke_error),
                )

            if current_user_id and revoke_token():
                # ✅ S3: Métrique Prometheus pour invalidation de token
                try:
                    from security.security_metrics import (
                        security_token_invalidations_total,
                    )

                    security_token_invalidations_total.labels(reason="logout").inc()
                except Exception:
                    pass  # Ne pas bloquer si métriques indisponibles

                # ✅ Priorité 7: Audit logging pour logout réussi
                try:
                    AuditLogger.log_action(
                        action_type="logout",
                        action_category="security",
                        user_id=user.id if user else None,
                        user_type=user.role.value if user and user.role else "unknown",
                        result_status="success",
                        action_details={
                            "refresh_token_revoked": refresh_token is not None,
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                    # ✅ Priorité 7: Métrique Prometheus pour logout
                    security_logout_total.inc()
                except Exception as audit_error:
                    logger.warning("Échec audit logging logout: %s", audit_error)

                # ✅ P0.1: Log structuré logout explicite (corrélation driver_id / device_id / session_diag)
                logger.info(
                    "auth_logout_explicit",
                    extra={
                        "event": "auth_logout_explicit",
                        "driver_id": driver_id_for_log,
                        "user_public_id": current_user_id,
                        "device_id": request.headers.get("X-Device-ID"),
                        "session_diag": request.headers.get("X-Session-Diag"),
                        "trace_id": get_trace_id(),
                    },
                )
            elif current_user_id:
                logger.warning(
                    "Logout: revoke_token a échoué pour user %s — cookies web effacés quand même",
                    current_user_id,
                )

            # Toujours 200 + clear cookies web : sans JWT valide, il faut quand même
            # supprimer Domain=.lirie.ch et les host-only hérités (sinon session fantôme).
            response = make_response({"message": "Déconnexion réussie"}, 200)
            if not is_mobile_request:
                _clear_web_auth_cookies(response)
            return response

        except Exception as e:
            sentry_sdk.capture_exception(e)
            # Même en erreur : tenter d'effacer les cookies web (pas de session fantôme).
            if not is_mobile_request:
                err_response = make_response(
                    {"message": "Déconnexion réussie", "warning": "partial"},
                    200,
                )
                try:
                    _clear_web_auth_cookies(err_response)
                    return err_response
                except Exception:
                    pass
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 4. Bootstrap session & switch-context (Unified Mobile Platform)
# ========================
@auth_ns.route("/bootstrap")
class AuthBootstrap(Resource):
    @jwt_required(optional=True)
    def get(self):
        request_id = get_trace_id() or str(uuid.uuid4())
        identity = get_jwt_identity()

        if not identity:
            return _bootstrap_base_response(
                request_id=request_id,
                is_authenticated=False,
            ), 200

        user = _load_user_for_bootstrap(str(identity))
        if not user:
            return _bootstrap_base_response(
                request_id=request_id,
                is_authenticated=False,
            ), 200

        user = _prepare_user_for_bootstrap(user)

        payload = _bootstrap_base_response(
            request_id=request_id,
            is_authenticated=True,
        )

        is_active, _denial_message = _check_user_profile_active(user)
        account_status = _normalized_account_status(user)
        if not is_active and account_status == "active":
            account_status = "suspended"

        contexts = _build_available_contexts(user)
        requested_context = request.headers.get("X-Active-Context-Id")
        saved_context = _get_saved_active_context(str(user.public_id))
        active_context_id = _resolve_active_context_id(
            contexts=contexts,
            preferred_context_id=requested_context or saved_context,
        )
        if active_context_id:
            _save_active_context(str(user.public_id), active_context_id)

        payload.update(
            {
                "user": {
                    "id": user.id,
                    "public_id": user.public_id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "role": user.role.value,
                    "force_password_change": bool(
                        getattr(user, "force_password_change", False)
                    ),
                    "password_expires_at": user.password_expires_at.isoformat()
                    if getattr(user, "password_expires_at", None)
                    else None,
                },
                "account_status": account_status,
                "onboarding_status": {
                    "status": getattr(user, "account_status", None) or "active",
                    "must_complete_onboarding": _must_complete_onboarding(user),
                    "reasons": _onboarding_reasons(user),
                    "required": _must_complete_onboarding(user),
                },
                "available_contexts": contexts,
                "active_context_id": active_context_id,
            }
        )
        return payload, 200


@auth_ns.route("/switch-context")
class AuthSwitchContext(Resource):
    @jwt_required()
    def post(self):
        request_id = get_trace_id() or str(uuid.uuid4())
        identity = get_jwt_identity()
        user = _load_user_for_bootstrap(str(identity))
        if not user:
            return {
                "error_code": "session_expired",
                "error_message": "Session expirée ou utilisateur introuvable.",
                "action_hint": "logout",
                "retryable": False,
                "request_id": request_id,
            }, 401

        user = _prepare_user_for_bootstrap(user)

        data = request.get_json(silent=True) or {}
        target_context_id = str(
            data.get("target_context_id") or data.get("context_id") or ""
        ).strip()
        if not target_context_id:
            return {
                "error_code": "validation_error",
                "error_message": "Le champ target_context_id est requis.",
                "action_hint": "retry",
                "retryable": True,
                "request_id": request_id,
            }, 400

        contexts = _build_available_contexts(user)
        context_ids = {
            cast("str", context.get("context_id"))
            for context in contexts
            if context.get("context_id")
        }
        if target_context_id not in context_ids:
            return {
                "error_code": "context_invalid",
                "error_message": "Le contexte demandé n'est pas disponible.",
                "action_hint": "open_context_selector",
                "retryable": False,
                "request_id": request_id,
            }, 403

        from_ctx = _context_by_id(
            contexts, _get_saved_active_context(str(user.public_id))
        )
        if from_ctx is None:
            dcid = _default_context_id(contexts)
            from_ctx = _context_by_id(contexts, dcid)
        to_ctx = _context_by_id(contexts, target_context_id)
        if (
            to_ctx
            and from_ctx
            and _is_company_driver_cross_context_switch(from_ctx, to_ctx)
        ):
            if user.role is not UserRole.COMPANY:
                return {
                    "error_code": "context_switch_company_account_only",
                    "error_message": (
                        "Seul le compte entreprise peut basculer entre l'espace entreprise et le chauffeur. "
                        "Un compte chauffeur seul n'a pas acces a la gestion d'entreprise."
                    ),
                    "action_hint": "open_context_selector",
                    "retryable": False,
                    "request_id": request_id,
                }, 403
            # App unifiée (Expo web + natif) : mêmes règles que le mobile, pas d'exclusion web
            if not (
                bool(from_ctx.get("allow_mobile_context_switch"))
                and bool(to_ctx.get("allow_mobile_context_switch"))
            ):
                return {
                    "error_code": "context_switch_transport_only",
                    "error_message": "Cette bascule est reservee aux comptes entreprise avec le dispatch actif.",
                    "action_hint": "open_context_selector",
                    "retryable": False,
                    "request_id": request_id,
                }, 403

        _save_active_context(str(user.public_id), target_context_id)
        return {
            "success": True,
            "active_context_id": target_context_id,
            "available_contexts": contexts,
            "feature_flags": _feature_flags_config(),
            "request_id": request_id,
            "status_dictionary_version": STATUS_DICTIONARY_VERSION,
            "pricing_contract_version": PRICING_CONTRACT_VERSION,
            "canonical_address_contract_version": CANONICAL_ADDRESS_CONTRACT_VERSION,
            "preview_contract_version": PREVIEW_CONTRACT_VERSION,
            "mission_status_version": MISSION_STATUS_VERSION,
            "mission_snapshot_version": MISSION_SNAPSHOT_VERSION,
            "driver_socket_contract_version": DRIVER_SOCKET_CONTRACT_VERSION,
            "driver_tracking_contract_version": DRIVER_TRACKING_CONTRACT_VERSION,
        }, 200


# ========================
# 5. Public mode utilities (mobile)
# ========================
@auth_ns.route("/public/service-area/check")
class PublicServiceAreaCheck(Resource):
    @limiter.limit("120 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        departure = str(data.get("departure") or "").strip()
        destination = str(data.get("destination") or "").strip()
        date = str(data.get("date") or "").strip()
        transport_type = str(data.get("transport_type") or "assis").strip().lower()
        if not departure or not destination or not date:
            return {
                "status": "unavailable",
                "reason_code": "UNKNOWN",
                "message": "Depart, destination et date sont requis.",
                "next_step": "try_later",
            }, 400

        # Source de vérité backend (règles minimales, extensibles sans MAJ mobile).
        concatenated = f"{departure} {destination}".lower()
        if "france" in concatenated or "lyon" in concatenated:
            return {
                "status": "unavailable",
                "reason_code": "OUT_OF_ZONE",
                "message": "Trajet hors zone couverte actuellement.",
                "next_step": "contact_support",
            }, 200
        if transport_type in {"pmr", "fauteuil", "wheelchair"}:
            return {
                "status": "conditional",
                "reason_code": "PMR_LIMITATION",
                "message": "Transport PMR possible sous validation de disponibilite.",
                "next_step": "continue",
            }, 200
        if "institution" in concatenated or "hopital" in concatenated:
            return {
                "status": "conditional",
                "reason_code": "PARTNER_REQUIRED",
                "message": "Trajet possible via reseau partenaire.",
                "next_step": "continue",
            }, 200
        return {
            "status": "available",
            "reason_code": "UNKNOWN",
            "message": "Trajet couvert par Lirie.",
            "next_step": "continue",
        }, 200


@auth_ns.route("/public/pre-request/draft")
class PublicPreRequestDraft(Resource):
    @limiter.limit("200 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        draft_id = str(data.get("draft_id") or "").strip()
        departure = str(data.get("departure") or "").strip()
        destination = str(data.get("destination") or "").strip()
        date = str(data.get("date") or "").strip()
        transport_type = str(data.get("transport_type") or "").strip()
        if (
            not draft_id
            or not departure
            or not destination
            or not date
            or not transport_type
        ):
            return {
                "error": "missing_fields",
                "error_message": "draft_id, departure, destination, date et transport_type sont requis.",
            }, 400
        payload = {
            "draft_id": draft_id,
            "departure": departure,
            "destination": destination,
            "date": date,
            "pickup_time": data.get("pickup_time"),
            "trip_type": data.get("trip_type"),
            "passengers": data.get("passengers"),
            "transport_type": transport_type,
            "special_requirements": data.get("special_requirements"),
            "contact_first_name": data.get("contact_first_name"),
            "contact_last_name": data.get("contact_last_name"),
            "contact_email": data.get("contact_email"),
            "contact_phone": data.get("contact_phone"),
            "service_area_status": data.get("service_area_status"),
            "updated_at": datetime.now(UTC).isoformat(),
            "consumed": False,
        }
        key = _build_public_pre_request_key(draft_id)
        status = "stored"
        if _public_cache_get(key):
            status = "updated"
        _public_cache_setex(key, 60 * 60 * 24 * 7, json.dumps(payload))
        return {
            "draft_id": draft_id,
            "status": status,
            "server_timestamp": datetime.now(UTC).isoformat(),
        }, 200


@auth_ns.route("/public/pre-request/draft/<string:draft_id>")
class PublicPreRequestDraftById(Resource):
    @limiter.limit("300 per hour")
    def get(self, draft_id: str):
        draft_id = str(draft_id or "").strip()
        if not draft_id:
            return {"draft": None}, 200
        key = _build_public_pre_request_key(draft_id)
        raw = _public_cache_get(key)
        if not raw:
            return {"draft": None}, 404
        try:
            parsed = json.loads(raw)
        except Exception:
            return {"draft": None}, 404
        if bool(parsed.get("consumed")):
            return {"draft": None}, 404
        return {"draft": parsed}, 200


@auth_ns.route("/public/pre-request/consume")
class PublicPreRequestConsume(Resource):
    @limiter.limit("200 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        draft_id = str(data.get("draft_id") or "").strip()
        if not draft_id:
            return {"error": "missing_draft_id"}, 400
        key = _build_public_pre_request_key(draft_id)
        raw = _public_cache_get(key)
        if not raw:
            return {"status": "missing"}, 200
        try:
            parsed = json.loads(raw)
        except Exception:
            _public_cache_delete(key)
            return {"status": "missing"}, 200
        parsed["consumed"] = True
        parsed["consumed_at"] = datetime.now(UTC).isoformat()
        _public_cache_setex(key, 60 * 60 * 24, json.dumps(parsed))
        return {"status": "consumed"}, 200


@auth_ns.route("/public/booking-status")
class PublicBookingStatus(Resource):
    @limiter.limit("400 per hour")
    def get(self):
        token = str(request.args.get("token") or "").strip()
        if not token:
            return {"error": "token_missing"}, 401
        booking_id, token_error = _load_booking_status_from_token(token)
        if token_error == "expired":
            return {"error": "token_expired"}, 410
        if token_error == "revoked":
            return {"error": "token_revoked"}, 410
        if token_error:
            return {"error": "token_invalid"}, 401
        if booking_id is None:
            return {"error": "token_invalid"}, 401

        booking = db.session.get(Booking, booking_id)
        if not booking:
            return {"error": "booking_not_found"}, 404

        raw_status = str(
            getattr(booking, "status", None)
            or getattr(booking, "reservation_status", None)
            or ""
        )
        normalized_status, label = _map_public_booking_status(raw_status)
        booking_reference = str(
            getattr(booking, "booking_reference", None)
            or getattr(booking, "id", booking_id)
        )
        updated_at_raw = getattr(booking, "updated_at", None)
        updated_at = (
            updated_at_raw.isoformat() if isinstance(updated_at_raw, datetime) else None
        )
        return {
            "status": normalized_status,
            "label": label,
            "updated_at": updated_at,
            "booking_reference": booking_reference,
        }, 200


@auth_ns.route("/public/guest-booking/preview")
class PublicGuestBookingPreview(Resource):
    @limiter.limit("120 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        departure = str(data.get("departure") or "").strip()
        destination = str(data.get("destination") or "").strip()
        date = str(data.get("date") or "").strip()
        pickup_time = str(data.get("pickup_time") or "").strip()
        trip_type = str(data.get("trip_type") or "one_way").strip().lower()
        if trip_type not in {"one_way", "round_trip"}:
            trip_type = "one_way"
        if not departure or not destination or not date:
            return {
                "error": "missing_fields",
                "error_message": "departure, destination et date sont requis.",
            }, 400
        if not pickup_time:
            return {
                "error": "missing_pickup_time",
                "error_message": "pickup_time est requis pour calculer le trajet.",
            }, 400

        out = compute_public_guest_booking_price(
            departure=departure,
            destination=destination,
            date=date,
            pickup_time=pickup_time,
            trip_type=trip_type,
        )
        if not out.get("ok"):
            code = 422
            err = str(out.get("error") or "pricing_failed")
            if err in {"missing_fields", "invalid_schedule"}:
                code = 400
            return {
                "error": err,
                "error_message": str(out.get("error_message") or "Prix indisponible."),
            }, code

        return {
            "pricing": {
                "amount": float(out["amount"]),
                "currency": str(out.get("currency") or "CHF"),
                "distance_meters": int(out.get("distance_meters") or 0),
                "duration_seconds": int(out.get("duration_seconds") or 0),
                "pricing_profile_id": out.get("pricing_profile_id"),
                "pricing_profile_version_id": out.get("pricing_profile_version_id"),
                "pricing_status": str(out.get("pricing_status") or "confirmed"),
                "breakdown": out.get("breakdown"),
            },
            "workflow": {
                "guest_checkout_enabled": True,
                "payment_required": True,
            },
        }, 200


@auth_ns.route("/public/guest-booking/create")
class PublicGuestBookingCreate(Resource):
    @limiter.limit("40 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        required_fields = [
            "departure",
            "destination",
            "date",
            "pickup_time",
        ]
        missing = [
            field for field in required_fields if not str(data.get(field) or "").strip()
        ]
        if missing:
            return {
                "error": "missing_fields",
                "error_message": f"Champs requis manquants: {', '.join(missing)}",
            }, 400

        guest_booking_id = f"gb_{uuid.uuid4().hex[:14]}"
        serializer = _public_link_serializer()
        status_token = serializer.dumps(
            {"guest_booking_id": guest_booking_id},
            salt="guest-booking-status-link",
        )
        transport_type = str(data.get("transport_type") or "assis").strip().lower()
        trip_type_raw = str(data.get("trip_type") or "one_way").strip().lower()
        if trip_type_raw not in {"one_way", "round_trip"}:
            trip_type_raw = "one_way"

        price_out = compute_public_guest_booking_price(
            departure=str(data.get("departure") or "").strip(),
            destination=str(data.get("destination") or "").strip(),
            date=str(data.get("date") or "").strip(),
            pickup_time=str(data.get("pickup_time") or "").strip(),
            trip_type=trip_type_raw,
        )
        if not price_out.get("ok"):
            return {
                "error": str(price_out.get("error") or "pricing_failed"),
                "error_message": str(
                    price_out.get("error_message") or "Impossible de calculer le prix."
                ),
            }, 422

        server_amount = float(price_out["amount"])
        payload = {
            "guest_booking_id": guest_booking_id,
            "departure": str(data.get("departure") or "").strip(),
            "destination": str(data.get("destination") or "").strip(),
            "date": str(data.get("date") or "").strip(),
            "pickup_time": str(data.get("pickup_time") or "").strip(),
            "transport_type": transport_type,
            "trip_type": trip_type_raw,
            "passengers": int(data.get("passengers") or 1),
            "first_name": str(data.get("first_name") or "").strip() or None,
            "last_name": str(data.get("last_name") or "").strip() or None,
            "email": str(data.get("email") or "").strip() or None,
            "phone": str(data.get("phone") or "").strip() or None,
            "notes": str(data.get("notes") or "").strip() or None,
            "amount": server_amount,
            "pricing_profile_id": price_out.get("pricing_profile_id"),
            "pricing_profile_version_id": price_out.get("pricing_profile_version_id"),
            "price_breakdown": price_out.get("breakdown"),
            "currency": str(price_out.get("currency") or "CHF"),
            # Géo : déjà calculé par le pricing (géocodage) — repousse vers Booking pour le dispatch
            "pickup_lat": price_out.get("pickup_lat"),
            "pickup_lon": price_out.get("pickup_lon"),
            "dropoff_lat": price_out.get("dropoff_lat"),
            "dropoff_lon": price_out.get("dropoff_lon"),
            "pickup_geo_unit_id": price_out.get("pickup_geo_unit_id"),
            "dropoff_geo_unit_id": price_out.get("dropoff_geo_unit_id"),
            "distance_meters": price_out.get("distance_meters"),
            "duration_seconds": price_out.get("duration_seconds"),
            "status": "pending_payment",
            "consumed": False,
            "linked_user_public_id": None,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _public_cache_setex(
            _build_public_guest_booking_key(guest_booking_id),
            _resolve_guest_booking_ttl_seconds(),
            json.dumps(payload),
        )
        return {
            "guest_booking_id": guest_booking_id,
            "status": payload["status"],
            "status_token": status_token,
            "message": "Reservation invitée creee. Paiement et confirmation a finaliser.",
        }, 201


@auth_ns.route("/public/guest-booking/saferpay/initialize")
class PublicGuestSaferpayInitialize(Resource):
    @limiter.limit("30 per hour")
    def post(self):
        from services.guest_saferpay import initialize_guest_saferpay
        from services.saferpay.config import saferpay_configured

        if not saferpay_configured():
            return {
                "error": "payment_unavailable",
                "message": "Paiement Saferpay non configuré sur ce serveur",
            }, 503

        data = request.get_json(silent=True) or {}
        token = str(data.get("status_token") or "").strip()
        guest_booking_id_body = str(data.get("guest_booking_id") or "").strip()
        gid, err = _decode_guest_booking_status_token(token)
        if err or not gid:
            return {"error": err or "token_invalid"}, 401
        if guest_booking_id_body and guest_booking_id_body != gid:
            return {"error": "guest_booking_mismatch"}, 400

        raw = _public_cache_get(_build_public_guest_booking_key(gid))
        if not raw:
            return {"error": "booking_not_found"}, 404
        try:
            payload = json.loads(raw)
        except Exception:
            return {"error": "booking_not_found"}, 404

        return_url = data.get("return_url")
        if isinstance(return_url, str) and return_url.strip():
            logger.debug("Guest Saferpay init return_url: %s", return_url)
        try:
            out = initialize_guest_saferpay(
                guest_booking_id=gid,
                payload=payload,
                return_url_override=return_url if isinstance(return_url, str) else None,
                redis_setex=_public_cache_setex,
                guest_ttl_seconds=_resolve_guest_booking_ttl_seconds(),
            )
        except ValueError as e:
            msg = str(e)
            if (
                "return_url non autoris" in msg
                and isinstance(return_url, str)
                and return_url.strip()
            ):
                logger.debug(
                    "Guest Saferpay init: return_url rejeté par l’allowlist: %s",
                    return_url,
                )
            if msg == "already_promoted":
                raw2 = _public_cache_get(_build_public_guest_booking_key(gid))
                pl2: dict[str, Any] = {}
                if raw2:
                    with suppress(Exception):
                        pl2 = json.loads(raw2)
                bid = pl2.get("promoted_booking_id") or payload.get(
                    "promoted_booking_id"
                )
                logger.info(
                    "Guest Saferpay initialize refus (déjà promu)",
                    extra={
                        "guest_booking_id": gid,
                        "booking_id": bid,
                        "outcome": "initialize_already_promoted",
                    },
                )
                return {
                    "error": "already_promoted",
                    "booking_id": bid,
                    "public_status_token": pl2.get("public_status_token")
                    or payload.get("public_status_token"),
                }, 409
            if msg == "guest_booking_consumed":
                return {"error": "guest_booking_consumed"}, 409
            return {"error": "validation_error", "error_message": msg}, 400
        except RuntimeError as e:
            return {"error": "saferpay_initialize_failed", "error_message": str(e)}, 503

        return out, 200


@auth_ns.route("/public/guest-booking/saferpay/assert")
class PublicGuestSaferpayAssert(Resource):
    @limiter.limit("60 per hour")
    def post(self):
        from services.guest_saferpay import promote_guest_booking_after_saferpay
        from services.saferpay.config import saferpay_configured

        if not saferpay_configured():
            return {
                "error": "payment_unavailable",
                "message": "Paiement Saferpay non configuré sur ce serveur",
            }, 503

        data = request.get_json(silent=True) or {}
        token = str(data.get("status_token") or "").strip()
        guest_booking_id_body = str(data.get("guest_booking_id") or "").strip()
        gid, err = _decode_guest_booking_status_token(token)
        if err or not gid:
            return {"error": err or "token_invalid"}, 401
        if guest_booking_id_body and guest_booking_id_body != gid:
            return {"error": "guest_booking_mismatch"}, 400

        raw = _public_cache_get(_build_public_guest_booking_key(gid))
        if not raw:
            return {"error": "booking_not_found"}, 404
        try:
            payload = json.loads(raw)
        except Exception:
            return {"error": "booking_not_found"}, 404

        try:
            out = promote_guest_booking_after_saferpay(
                guest_booking_id=gid,
                payload=payload,
                redis_setex=_public_cache_setex,
                notify_key=None,
            )
        except ValueError as e:
            return {"error": "validation_error", "error_message": str(e)}, 400
        except Exception:
            db.session.rollback()
            logger.exception("PublicGuestSaferpayAssert guest_booking_id=%s", gid)
            return {"error": "assert_failed"}, 500

        st = str(out.get("status") or "").strip().lower()
        if st == "forbidden":
            return {"error": "forbidden"}, 403

        payment_status = "pending_verification"
        if st in {"completed", "already_completed"}:
            payment_status = "paid"
        elif st in {"payment_failed", "assert_failed", "unexpected_tx_status"}:
            payment_status = "failed"

        body = {
            **out,
            "payment_provider": "saferpay",
            "payment_status": payment_status,
            "pending_verification": payment_status == "pending_verification",
        }
        return body, 200


@auth_ns.route("/public/guest-booking/status")
class PublicGuestBookingStatus(Resource):
    @limiter.limit("250 per hour")
    def get(self):
        token = str(request.args.get("token") or "").strip()
        if not token:
            return {"error": "token_missing"}, 401
        serializer = _public_link_serializer()
        try:
            decoded = serializer.loads(
                token,
                salt="guest-booking-status-link",
                max_age=_resolve_guest_booking_ttl_seconds(),
            )
        except SignatureExpired:
            return {"error": "token_expired"}, 410
        except BadSignature:
            return {"error": "token_invalid"}, 401
        guest_booking_id = str((decoded or {}).get("guest_booking_id") or "").strip()
        if not guest_booking_id:
            return {"error": "token_invalid"}, 401
        raw = _public_cache_get(_build_public_guest_booking_key(guest_booking_id))
        if not raw:
            return {"error": "booking_not_found"}, 404
        with suppress(Exception):
            payload = json.loads(raw)
            if payload.get("promoted_booking_id") and payload.get(
                "public_status_token"
            ):
                return {
                    "guest_booking_id": payload.get("guest_booking_id"),
                    "status": "already_promoted",
                    "booking_id": int(payload["promoted_booking_id"]),
                    "public_status_token": payload.get("public_status_token"),
                    "departure": payload.get("departure"),
                    "destination": payload.get("destination"),
                    "date": payload.get("date"),
                    "pickup_time": payload.get("pickup_time"),
                    "amount": payload.get("amount"),
                    "currency": payload.get("currency", "CHF"),
                    "updated_at": payload.get("updated_at"),
                    "linked_to_account": bool(payload.get("linked_user_public_id")),
                }, 200
            return {
                "guest_booking_id": payload.get("guest_booking_id"),
                "status": payload.get("status", "unknown"),
                "departure": payload.get("departure"),
                "destination": payload.get("destination"),
                "date": payload.get("date"),
                "pickup_time": payload.get("pickup_time"),
                "amount": payload.get("amount"),
                "currency": payload.get("currency", "CHF"),
                "updated_at": payload.get("updated_at"),
                "linked_to_account": bool(payload.get("linked_user_public_id")),
            }, 200
        return {"error": "booking_not_found"}, 404


@auth_ns.route("/public/guest-booking/link")
class PublicGuestBookingLink(Resource):
    @jwt_required()
    @limiter.limit("60 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        token = str(data.get("status_token") or "").strip()
        if not token:
            return {"error": "status_token_missing"}, 400
        serializer = _public_link_serializer()
        try:
            decoded = serializer.loads(
                token,
                salt="guest-booking-status-link",
                max_age=_resolve_guest_booking_ttl_seconds(),
            )
        except SignatureExpired:
            return {"error": "token_expired"}, 410
        except BadSignature:
            return {"error": "token_invalid"}, 401
        guest_booking_id = str((decoded or {}).get("guest_booking_id") or "").strip()
        if not guest_booking_id:
            return {"error": "token_invalid"}, 401
        cache_key = _build_public_guest_booking_key(guest_booking_id)
        raw = _public_cache_get(cache_key)
        if not raw:
            return {"error": "guest_booking_not_found"}, 404
        with suppress(Exception):
            payload = json.loads(raw)
            user_public_id = str(get_jwt_identity() or "").strip()
            if not user_public_id:
                return {"error": "auth_required"}, 401
            payload["linked_user_public_id"] = user_public_id
            payload["updated_at"] = datetime.now(UTC).isoformat()
            _public_cache_setex(
                cache_key,
                _resolve_guest_booking_ttl_seconds(),
                json.dumps(payload),
            )
            return {
                "status": "linked",
                "guest_booking_id": guest_booking_id,
                "linked_user_public_id": user_public_id,
            }, 200
        return {"error": "guest_booking_not_found"}, 404


@auth_ns.route("/passwordless/otp/request")
class PasswordlessOtpRequest(Resource):
    @limiter.limit("40 per hour")
    def post(self):
        if not _passwordless_allowed_in_environment():
            return {"error": "Not Found"}, 404

        try:
            data = validate_request(
                PasswordlessOtpRequestSchema(),
                request.get_json(silent=True) or {},
                strict=False,
            )
        except ValidationError as e:
            body, code = handle_validation_error(e)
            return body, code or 400

        channel = str(data.get("channel") or "").strip().lower()
        identifier = str(data.get("identifier") or "").strip()
        user: User | None = None
        if channel == "email":
            user = User.query.filter(cast("Any", User.email).ilike(identifier)).first()
        else:
            client = (
                Client.query.options(joinedload(Client.user))
                .filter(Client.contact_phone == identifier)
                .first()
            )
            user = client.user if client else None

        if not user:
            return {"error": "identifier_not_found"}, 404
        is_active, _ = _check_user_profile_active(user)
        if not is_active:
            return {"error": "account_inactive"}, 403

        otp_session_id = f"otp_{uuid.uuid4().hex[:18]}"
        code = _create_passwordless_otp_code()
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        payload = {
            "otp_session_id": otp_session_id,
            "user_public_id": str(user.public_id),
            "channel": channel,
            "identifier": identifier,
            "code_hash": code_hash,
            "attempts": 0,
            "max_attempts": 5,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _public_cache_setex(
            _build_passwordless_otp_key(otp_session_id),
            _resolve_passwordless_otp_ttl_seconds(),
            json.dumps(payload),
        )
        response_body: dict[str, Any] = {
            "otp_session_id": otp_session_id,
            "channel": channel,
            "masked_identifier": mask_email(identifier)
            if channel == "email"
            else _mask_phone(identifier),
            "expires_in_seconds": _resolve_passwordless_otp_ttl_seconds(),
        }
        if _passwordless_debug_code_enabled():
            response_body["debug_code"] = code
        return response_body, 200


@auth_ns.route("/passwordless/otp/verify")
class PasswordlessOtpVerify(Resource):
    @limiter.limit("60 per hour")
    def post(self):
        if not _passwordless_allowed_in_environment():
            return {"error": "Not Found"}, 404

        try:
            data = validate_request(
                PasswordlessOtpVerifySchema(),
                request.get_json(silent=True) or {},
                strict=False,
            )
        except ValidationError as e:
            body, code = handle_validation_error(e)
            return body, code or 400

        otp_session_id = str(data.get("otp_session_id") or "").strip()
        code = str(data.get("code") or "").strip()
        cache_key = _build_passwordless_otp_key(otp_session_id)
        raw = _public_cache_get(cache_key)
        if not raw:
            return {"error": "otp_session_expired"}, 410
        with suppress(Exception):
            payload = json.loads(raw)
            attempts = int(payload.get("attempts") or 0)
            max_attempts = int(payload.get("max_attempts") or 5)
            if attempts >= max_attempts:
                return {"error": "too_many_attempts"}, 429
            expected_hash = str(payload.get("code_hash") or "")
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(code_hash, expected_hash):
                payload["attempts"] = attempts + 1
                _public_cache_setex(
                    cache_key,
                    _resolve_passwordless_otp_ttl_seconds(),
                    json.dumps(payload),
                )
                return {"error": "invalid_code"}, 401

            user_public_id = str(payload.get("user_public_id") or "").strip()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                return {"error": "user_not_found"}, 404
            claims = {
                "role": user.role.value,
                "company_id": _resolve_company_id(user),
                "driver_id": getattr(user, "driver_id", None),
                "institution_id": getattr(user, "institution_id", None),
                "institution_role": getattr(user, "institution_role", None),
                "aud": "atmr-api",
                "token_version": _user_token_version(user),
            }
            access_token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
                expires_delta=_resolve_access_token_expires(True),
                fresh=True,
            )
            refresh_token = create_refresh_token(
                identity=str(user.public_id),
                additional_claims={
                    "aud": "atmr-api",
                    "pwd_hash": _get_password_hash_version(user),
                    "token_version": _user_token_version(user),
                },
                expires_delta=current_app.config["JWT_REFRESH_TOKEN_EXPIRES"],
            )
            with suppress(Exception):
                refresh_expires_at = (
                    datetime.now(UTC) + current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
                )
                store_refresh_token(
                    token=refresh_token,
                    user_id=user.id,
                    expires_at=refresh_expires_at,
                    device_id=request.headers.get("X-Device-ID"),
                    device_name=request.headers.get("X-Device-Name"),
                )
            _public_cache_delete(cache_key)
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "auth_mode": "passwordless_otp",
            }, 200
        return {"error": "otp_session_expired"}, 410


# ========================
# 5. Informations Utilisateur
# ========================
@auth_ns.route("/me")
class UserInfo(Resource):
    @jwt_required()
    def get(self):
        """Retourne les informations de l'utilisateur connecté."""
        try:
            from application.auth_bootstrap.get_bootstrap_session_use_case import (
                GetBootstrapSessionUseCase,
            )

            outcome = GetBootstrapSessionUseCase().execute()
            return outcome.body, outcome.status_code

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 5. Inscription
# ========================
@auth_ns.route("/register")
class Register(Resource):
    # ✅ FIX: validate=False pour laisser Marshmallow + PasswordPolicyService
    # gérer la validation avec des messages d'erreur clairs
    @auth_ns.expect(register_model, validate=False)
    @limiter.limit(
        "10 per minute"
    )  # ✅ SECURITY: Rate limiting pour prévenir spam d'inscriptions
    def post(self):
        """Inscrit un nouvel utilisateur avec le rôle 'client' par défaut
        et crée un profil client associé.
        """
        # Gérer la requête OPTIONS pour CORS si nécessaire
        if request.method == "OPTIONS":
            response = make_response("")
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            return response, 204

        try:
            data = request.get_json() or {}
            logger.info(
                "Inscription reçue: email=%s role=%s has_password=%s username_len=%s",
                mask_email(str(data.get("email") or "")),
                data.get("role"),
                bool(data.get("password")),
                len(str(data.get("username") or "")),
            )

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            try:
                validated_data = validate_request(RegisterSchema(), data, strict=False)
            except ValidationError as e:
                # ✅ FIX: Retourner un dict directement (Flask-RESTX le convertit en JSON)
                body, code = handle_validation_error(e)
                return body, code or 400

            logger.info(
                "Inscription validée: email=%s has_password=%s username_len=%s",
                mask_email(str(validated_data.get("email") or "")),
                bool(validated_data.get("password")),
                len(str(validated_data.get("username") or "")),
            )

            # ✅ DDD: Utiliser le use case pour enregistrer l'utilisateur
            username: str = cast("str", validated_data.get("username"))
            password: str = cast("str", validated_data.get("password"))
            email: str = cast("str", validated_data.get("email"))
            phone: str | None = cast("str | None", validated_data.get("phone"))

            if not phone:
                return auth_error(
                    AuthErrorCodes.REGISTRATION_ERROR,
                    "Le numéro de téléphone est requis pour activer le compte.",
                    400,
                )

            # ✅ S3: Validation explicite du mot de passe avec politique renforcée
            from security.password_policy import (
                PasswordPolicyError,
                PasswordPolicyService,
            )

            try:
                # Valider avec la politique stricte (complexité + HIBP)
                PasswordPolicyService.validate_password(
                    password, user_id=None, check_history=False
                )
            except PasswordPolicyError as e:
                # ✅ FIX: Retourner un message d'erreur clair via helper standard
                return auth_error(
                    AuthErrorCodes.PASSWORD_POLICY_ERROR,
                    e.message,
                    400,
                )

            uc = RegisterUserUseCase()
            input_data = RegisterUserInput(
                username=username,
                email=email,
                password=password,
                first_name=validated_data.get("first_name"),
                last_name=validated_data.get("last_name"),
                phone=phone,
                address=validated_data.get("address"),
                birth_date=validated_data.get("birth_date"),
                gender=validated_data.get("gender"),
                profile_image=validated_data.get("profile_image"),
            )
            register_result = uc.execute(input_data)

            if not register_result.success:
                # ✅ FIX: Retourner un dict directement via helper standard
                # Déterminer le code d'erreur approprié
                error_msg = (
                    register_result.error.get("error", "Erreur lors de l'inscription")
                    if register_result.error
                    else "Erreur lors de l'inscription"
                )

                # Mapper les messages aux codes d'erreur spécifiques
                lower_msg = error_msg.lower()
                if "existe" in lower_msg:
                    err_code = (
                        AuthErrorCodes.EMAIL_EXISTS
                        if "email" in lower_msg
                        else AuthErrorCodes.USERNAME_EXISTS
                        if "utilisateur" in lower_msg
                        else AuthErrorCodes.REGISTRATION_ERROR
                    )
                    return auth_error(err_code, error_msg, 409)

                # Code générique pour les autres erreurs
                return auth_error(
                    AuthErrorCodes.REGISTRATION_ERROR,
                    error_msg,
                    register_result.status_code or 400,
                )

            user = register_result.user
            if not user:
                auth_ns.abort(500, "User created but not returned")

            # Création du profil client associé
            # user est garanti non-None après la vérification ci-dessus
            assert user is not None  # Pour le type checker
            client = Client()
            client.user_id = user.id
            client.is_active = False
            client.contact_email = email
            # Inscription portail grand public → marché ouvert aux transporteurs
            client.client_type = ClientType.PORTAL
            db.session.add(client)

            now = datetime.now(UTC)
            activation_session = ActivationSession()
            activation_session.activation_session_id = str(uuid.uuid4())
            activation_session.user_id = user.id
            activation_session.email_token_expires_at = now + timedelta(
                minutes=ACTIVATION_EMAIL_TTL_MINUTES
            )
            activation_session.sms_expires_at = now + timedelta(
                minutes=ACTIVATION_SMS_TTL_MINUTES
            )
            activation_session.sms_attempts = 0
            activation_session.resend_count_email = 0
            activation_session.resend_count_sms = 0

            sms_code = _generate_sms_otp()
            activation_session.sms_code_hash = _hash_plain_value(sms_code)
            db.session.add(activation_session)
            db.session.commit()

            from models.activation_session import EMAIL_DELIVERY_KIND_INITIAL
            from services.notifications.activation_email_delivery import (
                try_enqueue_activation_email,
            )

            environment = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
            # Lot 1 : jeton HMAC dérivé de email_delivery_id (pas itsdangerous)
            enqueue_result = try_enqueue_activation_email(
                activation_session,
                kind=EMAIL_DELIVERY_KIND_INITIAL,
                environment=environment,
                is_testing=bool(current_app.config.get("TESTING")),
            )

            sms_sent = False
            try:
                sms_sent = _send_activation_sms(user, sms_code)
                if sms_sent:
                    activation_session.last_sms_sent_at = datetime.now(UTC)
                    db.session.commit()
            except Exception as sms_err:
                logger.warning("[Activation] Echec envoi SMS activation: %s", sms_err)

            logger.info("Client créé : user_id=%s, client_id=%s", user.id, client.id)

            if enqueue_result.get("require_502"):
                body, code = _activation_email_send_failed_body(
                    activation_session_id=activation_session.activation_session_id,
                    user=user,
                )
                if enqueue_result.get("debug_activation_link"):
                    body["debug_activation_link"] = enqueue_result[
                        "debug_activation_link"
                    ]
                return body, code

            response_body: dict[str, object] = {
                "message": "Inscription créée. Activez votre compte via email et SMS.",
                "user_id": user.public_id,
                "username": user.username,
                "activation_session_id": activation_session.activation_session_id,
                "masked_email": mask_email(user.email or ""),
                "masked_phone": _mask_phone(user.phone),
                "email_sent": None,
                "activation_email_queued": bool(enqueue_result.get("queued")),
                "sms_sent": sms_sent,
            }
            if enqueue_result.get("debug_activation_link"):
                response_body["debug_activation_link"] = enqueue_result[
                    "debug_activation_link"
                ]
            return response_body, 201

        except ValidationError as e:
            logger.error("Erreur de validation : %s", e.messages)
            auth_ns.abort(400, "Validation failed")
        except Exception as e:
            # Flask-RESTX abort() lève une HTTPException (ex: 400/409).
            # Ne pas transformer ces erreurs attendues en 500.
            from werkzeug.exceptions import (
                HTTPException,
            )

            if isinstance(e, HTTPException):
                raise
            sentry_sdk.capture_exception(e)
            # Utiliser repr() pour éviter les problèmes de formatage avec %
            exception_message = repr(e) if "%" in str(e) else str(e)
            logger.exception(
                "❌ ERREUR register_user: %s - %s", type(e).__name__, exception_message
            )
            auth_ns.abort(500, "Une erreur interne est survenue.")


@auth_ns.route("/activation/verify-email")
class VerifyActivationEmail(Resource):
    @limiter.limit("20 per hour")
    def post(self):
        """Valide le lien email d'une session d'activation client (F-03)."""
        try:
            data = request.get_json() or {}
            validated_data = validate_request(VerifyEmailActivationSchema(), data)
            token = cast("str", validated_data.get("token", ""))

            from models.activation_email_delivery import ActivationEmailDelivery
            from services.notifications.activation_email_delivery import (
                get_activation_session_for_update,
            )
            from services.notifications.activation_token import (
                hash_activation_token,
                verify_activation_token,
            )
            from services.security.activation_legacy import is_legacy_acceptance_active

            token_hash = hash_activation_token(token)
            matches = (
                ActivationEmailDelivery.query.filter_by(email_token_hash=token_hash)
                .limit(2)
                .all()
            )

            # --- Branche HMAC moderne ---
            if matches:
                if len(matches) != 1:
                    logger.warning(
                        "activation_email_verify_rejected reason=duplicate_hash"
                    )
                    return auth_error(
                        AuthErrorCodes.TOKEN_INVALID,
                        "Lien d'activation invalide.",
                        400,
                    )
                delivery = matches[0]
                if not verify_activation_token(
                    token,
                    delivery.email_delivery_id,
                    key_version=int(delivery.token_key_version or 1),
                ):
                    logger.warning(
                        "activation_email_verify_rejected reason=invalid "
                        "email_delivery_id=%s",
                        delivery.email_delivery_id,
                    )
                    return auth_error(
                        AuthErrorCodes.TOKEN_INVALID,
                        "Lien d'activation invalide.",
                        400,
                    )

                locked = get_activation_session_for_update(
                    delivery.activation_session_pk
                )
                delivery = (
                    ActivationEmailDelivery.query.filter_by(
                        email_delivery_id=delivery.email_delivery_id
                    )
                    .populate_existing()
                    .with_for_update()
                    .one()
                )
                if delivery.activation_session_pk != locked.id:
                    db.session.rollback()
                    logger.warning(
                        "activation_email_verify_rejected reason=invalid "
                        "email_delivery_id=%s",
                        delivery.email_delivery_id,
                    )
                    return auth_error(
                        AuthErrorCodes.TOKEN_INVALID,
                        "Lien d'activation invalide.",
                        400,
                    )

                now = datetime.now(UTC)
                if (
                    locked.email_delivery_id != delivery.email_delivery_id
                    or delivery.superseded_at is not None
                ):
                    db.session.rollback()
                    logger.warning(
                        "activation_email_verify_rejected reason=superseded "
                        "email_delivery_id=%s",
                        delivery.email_delivery_id,
                    )
                    return auth_error(
                        AuthErrorCodes.TOKEN_EXPIRED,
                        "Ce lien a été remplacé. Utilisez le dernier email reçu.",
                        400,
                    )
                if delivery.token_expires_at is None:
                    db.session.rollback()
                    logger.warning(
                        "activation_email_verify_rejected reason=invalid "
                        "email_delivery_id=%s",
                        delivery.email_delivery_id,
                    )
                    return auth_error(
                        AuthErrorCodes.TOKEN_INVALID,
                        "Lien d'activation invalide.",
                        400,
                    )
                expires = delivery.token_expires_at
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=UTC)
                if now >= expires:
                    db.session.rollback()
                    logger.warning(
                        "activation_email_verify_rejected reason=expired "
                        "email_delivery_id=%s",
                        delivery.email_delivery_id,
                    )
                    return auth_error(
                        AuthErrorCodes.TOKEN_EXPIRED,
                        "Le lien email a expiré. Demandez un nouvel envoi.",
                        400,
                    )

                if locked.email_verified_at:
                    db.session.commit()
                    user = User.query.get(locked.user_id)
                    return {
                        "message": "Email déjà confirmé.",
                        "activation_session_id": locked.activation_session_id,
                        "masked_email": mask_email(user.email or "") if user else None,
                        "masked_phone": _mask_phone(user.phone) if user else None,
                        "activation_status": _build_activation_status(locked),
                    }, 200

                locked.email_verified_at = now
                db.session.commit()
                user = User.query.get(locked.user_id)
                return {
                    "message": "Email confirmé.",
                    "activation_session_id": locked.activation_session_id,
                    "masked_email": mask_email(user.email or "") if user else None,
                    "masked_phone": _mask_phone(user.phone) if user else None,
                    "activation_status": _build_activation_status(locked),
                }, 200

            # --- Branche legacy itsdangerous (bornée F-03) ---
            if not is_legacy_acceptance_active():
                logger.warning(
                    "activation_email_verify_rejected reason=legacy_disabled"
                )
                return auth_error(
                    AuthErrorCodes.TOKEN_INVALID,
                    "Lien d'activation invalide.",
                    400,
                )
            try:
                payload = _activation_serializer().loads(
                    token,
                    salt="activation-email-salt",
                    max_age=1800,
                )
            except SignatureExpired:
                return auth_error(
                    AuthErrorCodes.TOKEN_EXPIRED,
                    "Le lien email a expiré. Demandez un nouvel envoi.",
                    400,
                )
            except BadSignature:
                return auth_error(
                    AuthErrorCodes.TOKEN_INVALID,
                    "Lien d'activation invalide.",
                    400,
                )
            session_id = payload.get("sid")
            if not session_id:
                return auth_error(
                    AuthErrorCodes.TOKEN_INVALID,
                    "Lien d'activation invalide.",
                    400,
                )
            activation_session = ActivationSession.query.filter_by(
                activation_session_id=session_id
            ).first()
            if not activation_session:
                return auth_error(
                    AuthErrorCodes.TOKEN_INVALID,
                    "Lien d'activation invalide.",
                    400,
                )
            if activation_session.email_delivery_id is not None:
                return auth_error(
                    AuthErrorCodes.TOKEN_INVALID,
                    "Lien d'activation invalide.",
                    400,
                )
            if (
                ActivationEmailDelivery.query.filter_by(
                    activation_session_pk=activation_session.id
                ).first()
                is not None
            ):
                return auth_error(
                    AuthErrorCodes.TOKEN_INVALID,
                    "Lien d'activation invalide.",
                    400,
                )
            if not hmac.compare_digest(
                _hash_plain_value(token),
                activation_session.email_token_hash or "",
            ):
                return auth_error(
                    AuthErrorCodes.TOKEN_INVALID,
                    "Lien d'activation invalide.",
                    400,
                )
            now = datetime.now(UTC)
            if (
                not activation_session.email_token_expires_at
                or activation_session.email_token_expires_at <= now
            ):
                return auth_error(
                    AuthErrorCodes.TOKEN_EXPIRED,
                    "Le lien email a expiré. Demandez un nouvel envoi.",
                    400,
                )
            if activation_session.email_verified_at:
                user = User.query.get(activation_session.user_id)
                return {
                    "message": "Email déjà confirmé.",
                    "activation_session_id": activation_session.activation_session_id,
                    "masked_email": mask_email(user.email or "") if user else None,
                    "masked_phone": _mask_phone(user.phone) if user else None,
                    "activation_status": _build_activation_status(activation_session),
                }, 200
            activation_session.email_verified_at = now
            db.session.commit()
            user = User.query.get(activation_session.user_id)
            return {
                "message": "Email confirmé.",
                "activation_session_id": activation_session.activation_session_id,
                "masked_email": mask_email(user.email or "") if user else None,
                "masked_phone": _mask_phone(user.phone) if user else None,
                "activation_status": _build_activation_status(activation_session),
            }, 200
        except ValidationError as e:
            return handle_validation_error(e)
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/activation/verify-sms")
class VerifyActivationSms(Resource):
    @limiter.limit("30 per hour")
    def post(self):
        """Valide le code OTP SMS pour une session d'activation client."""
        try:
            data = request.get_json() or {}
            validated_data = validate_request(VerifySmsActivationSchema(), data)
            session_id = cast("str", validated_data.get("activation_session_id"))
            code = cast("str", validated_data.get("code"))

            activation_session = ActivationSession.query.filter_by(
                activation_session_id=session_id
            ).first()
            if not activation_session:
                return {"error": "Session d'activation introuvable."}, 404

            if activation_session.phone_verified_at:
                return {
                    "message": "Téléphone déjà confirmé.",
                    "activation_status": _build_activation_status(activation_session),
                }, 200

            now = datetime.now(UTC)
            if (
                activation_session.sms_locked_until
                and activation_session.sms_locked_until > now
            ):
                retry_after = int(
                    (activation_session.sms_locked_until - now).total_seconds()
                )
                return auth_error(
                    AuthErrorCodes.ACCOUNT_LOCKED,
                    "Trop d'essais SMS. Réessayez plus tard.",
                    429,
                    details={"retry_after_seconds": retry_after},
                )

            if (
                activation_session.sms_expires_at
                and activation_session.sms_expires_at < now
            ):
                return auth_error(
                    AuthErrorCodes.TOKEN_EXPIRED,
                    "Le code SMS a expiré. Demandez un nouveau code.",
                    400,
                )

            code_hash = _hash_plain_value(code)
            if not hmac.compare_digest(
                code_hash, activation_session.sms_code_hash or ""
            ):
                attempts = int(activation_session.sms_attempts or 0) + 1
                activation_session.sms_attempts = attempts
                if attempts >= ACTIVATION_SMS_MAX_ATTEMPTS:
                    activation_session.sms_attempts = 0
                    activation_session.sms_locked_until = now + timedelta(
                        minutes=ACTIVATION_SMS_LOCK_MINUTES
                    )
                    db.session.commit()
                    return auth_error(
                        AuthErrorCodes.ACCOUNT_LOCKED,
                        "Trop d'essais SMS. Réessayez plus tard.",
                        429,
                        details={
                            "retry_after_seconds": ACTIVATION_SMS_LOCK_MINUTES * 60
                        },
                    )
                db.session.commit()
                return auth_error(
                    AuthErrorCodes.INVALID_CREDENTIALS,
                    "Code SMS invalide.",
                    400,
                    details={
                        "remaining_attempts": ACTIVATION_SMS_MAX_ATTEMPTS - attempts
                    },
                )

            activation_session.phone_verified_at = now
            activation_session.sms_attempts = 0
            activation_session.sms_locked_until = None
            db.session.commit()
            return {
                "message": "Téléphone confirmé.",
                "activation_status": _build_activation_status(activation_session),
            }, 200
        except ValidationError as e:
            return handle_validation_error(e)
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/activation/finalize")
class FinalizeActivation(Resource):
    @limiter.limit("20 per hour")
    def post(self):
        """Active définitivement le compte après validation email + SMS."""
        try:
            data = request.get_json() or {}
            validated_data = validate_request(FinalizeActivationSchema(), data)
            session_id = cast("str", validated_data.get("activation_session_id"))

            activation_session = ActivationSession.query.filter_by(
                activation_session_id=session_id
            ).first()
            if not activation_session:
                return {"error": "Session d'activation introuvable."}, 404

            if activation_session.consumed_at:
                return {
                    "message": "Compte déjà activé.",
                    "activation_status": _build_activation_status(activation_session),
                }, 200

            if (
                not activation_session.email_verified_at
                or not activation_session.phone_verified_at
            ):
                return auth_error(
                    AuthErrorCodes.EMAIL_NOT_VERIFIED,
                    "Validation incomplète: confirmez l'email et le SMS avant d'activer le compte.",
                    400,
                    details=_build_activation_status(activation_session),
                )

            user = User.query.get(activation_session.user_id)
            if not user:
                return {"error": "Utilisateur introuvable."}, 404

            user.account_status = "active"
            for client in user.clients:
                client.is_active = True
            activation_session.consumed_at = datetime.now(UTC)
            db.session.commit()

            return {
                "message": "Compte activé avec succès.",
                "user_id": user.public_id,
                "activation_status": _build_activation_status(activation_session),
            }, 200
        except ValidationError as e:
            return handle_validation_error(e)
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/activation/resend-email")
class ResendActivationEmail(Resource):
    @limiter.limit("10 per hour")
    def post(self):
        """Renvoyer un email d'activation (asynchrone via Celery)."""
        try:
            data = request.get_json() or {}
            validated_data = validate_request(ResendActivationSchema(), data)
            session_id = cast("str", validated_data.get("activation_session_id"))

            activation_session = ActivationSession.query.filter_by(
                activation_session_id=session_id
            ).first()
            if not activation_session:
                return {"error": "Session d'activation introuvable."}, 404

            if activation_session.email_verified_at:
                return {"message": "Email déjà confirmé."}, 200

            from models.activation_session import EMAIL_DELIVERY_KIND_RESEND
            from services.notifications.activation_email_delivery import (
                can_start_new_delivery_snapshot,
                try_enqueue_activation_email,
            )
            from services.notifications.activation_email_policy import (
                enforce_resend_policy,
                is_same_utc_day,
            )

            # Précontrôles indicatifs (non mutatifs) — autorité = service sous verrou
            can_send, block_reason = can_start_new_delivery_snapshot(activation_session)
            if not can_send:
                return auth_error(
                    AuthErrorCodes.RATE_LIMITED,
                    "Envoi déjà en cours. Veuillez patienter.",
                    429,
                    details={
                        "retry_after_seconds": ACTIVATION_RESEND_COOLDOWN_SECONDS,
                        "reason": block_reason,
                    },
                )

            now = datetime.now(UTC)
            daily_count = int(activation_session.resend_count_email or 0)
            if activation_session.last_email_sent_at and not is_same_utc_day(
                activation_session.last_email_sent_at, now
            ):
                daily_count = 0

            allowed, policy_error, retry_after = enforce_resend_policy(
                last_sent_at=activation_session.last_email_sent_at,
                resend_count=daily_count,
            )
            if not allowed:
                message = (
                    "Veuillez patienter avant de renvoyer l'email."
                    if policy_error == "cooldown"
                    else "Limite journalière de renvoi email atteinte."
                )
                return auth_error(
                    AuthErrorCodes.RATE_LIMITED,
                    message,
                    429,
                    details={"retry_after_seconds": retry_after},
                )

            user = User.query.get(activation_session.user_id)
            if not user:
                return {"error": "Utilisateur introuvable."}, 404

            environment = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
            enqueue_result = try_enqueue_activation_email(
                activation_session,
                kind=EMAIL_DELIVERY_KIND_RESEND,
                environment=environment,
                is_testing=bool(current_app.config.get("TESTING")),
            )
            if enqueue_result.get("error") == "email_delivery_in_progress":
                return auth_error(
                    AuthErrorCodes.RATE_LIMITED,
                    "Envoi déjà en cours. Veuillez patienter.",
                    429,
                    details={
                        "retry_after_seconds": ACTIVATION_RESEND_COOLDOWN_SECONDS,
                        "reason": "email_delivery_in_progress",
                    },
                )
            if enqueue_result.get("error") in {"cooldown", "daily_limit"}:
                return auth_error(
                    AuthErrorCodes.RATE_LIMITED,
                    (
                        "Veuillez patienter avant de renvoyer l'email."
                        if enqueue_result.get("error") == "cooldown"
                        else "Limite journalière de renvoi email atteinte."
                    ),
                    429,
                    details={"retry_after_seconds": retry_after},
                )

            if enqueue_result.get("require_502"):
                body, code = _activation_email_send_failed_body(
                    activation_session_id=activation_session.activation_session_id,
                    user=user,
                )
                if enqueue_result.get("debug_activation_link"):
                    body["debug_activation_link"] = enqueue_result[
                        "debug_activation_link"
                    ]
                return body, code

            response_body: dict[str, object] = {
                "message": (
                    "Email d'activation en cours d'envoi."
                    if enqueue_result.get("queued")
                    else "Préparation de l'email d'activation."
                ),
                "activation_email_queued": bool(enqueue_result.get("queued")),
                "email_sent": None,
                "activation_status": _build_activation_status(activation_session),
            }
            if enqueue_result.get("debug_activation_link"):
                response_body["debug_activation_link"] = enqueue_result[
                    "debug_activation_link"
                ]
                response_body["message"] = (
                    "Service email indisponible en local. "
                    "Utilisez le lien de secours ci-dessous pour continuer l'activation."
                )
            return response_body, 200
        except ValidationError as e:
            return handle_validation_error(e)
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/activation/resend-sms")
class ResendActivationSms(Resource):
    @limiter.limit("15 per hour")
    def post(self):
        """Renvoyer un code SMS d'activation."""
        try:
            data = request.get_json() or {}
            validated_data = validate_request(ResendActivationSchema(), data)
            session_id = cast("str", validated_data.get("activation_session_id"))

            activation_session = ActivationSession.query.filter_by(
                activation_session_id=session_id
            ).first()
            if not activation_session:
                return {"error": "Session d'activation introuvable."}, 404

            if activation_session.phone_verified_at:
                return {"message": "Téléphone déjà confirmé."}, 200

            now = datetime.now(UTC)
            daily_count = int(activation_session.resend_count_sms or 0)
            if activation_session.last_sms_sent_at and not _is_same_utc_day(
                activation_session.last_sms_sent_at, now
            ):
                daily_count = 0

            allowed, policy_error, retry_after = _enforce_resend_policy(
                last_sent_at=activation_session.last_sms_sent_at,
                resend_count=daily_count,
            )
            if not allowed:
                message = (
                    "Veuillez patienter avant de renvoyer le SMS."
                    if policy_error == "cooldown"
                    else "Limite journalière de renvoi SMS atteinte."
                )
                return auth_error(
                    AuthErrorCodes.RATE_LIMITED,
                    message,
                    429,
                    details={"retry_after_seconds": retry_after},
                )

            sms_code = _generate_sms_otp()
            activation_session.sms_code_hash = _hash_plain_value(sms_code)
            activation_session.sms_expires_at = now + timedelta(
                minutes=ACTIVATION_SMS_TTL_MINUTES
            )
            activation_session.sms_attempts = 0
            activation_session.sms_locked_until = None
            activation_session.last_sms_sent_at = now
            activation_session.resend_count_sms = daily_count + 1

            user = User.query.get(activation_session.user_id)
            if not user:
                return {"error": "Utilisateur introuvable."}, 404
            if not _send_activation_sms(user, sms_code):
                environment = (
                    str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
                )
                is_local_dev = environment == "development" and not bool(
                    current_app.config.get("TESTING")
                )
                if is_local_dev:
                    db.session.commit()
                    logger.warning(
                        "[Activation] Fallback dev resend-sms (session=%s): provider indisponible",
                        activation_session.activation_session_id,
                    )
                    return {
                        "message": (
                            "Service SMS indisponible en local. "
                            "Utilisez le code de secours ci-dessous."
                        ),
                        "sms_sent": False,
                        "debug_sms_code": sms_code,
                    }, 200

                db.session.rollback()
                return {"error": "Echec envoi SMS. Réessayez plus tard."}, 502
            db.session.commit()
            return {"message": "Code SMS renvoyé."}, 200
        except ValidationError as e:
            return handle_validation_error(e)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/activation/update-phone")
class UpdateActivationPhone(Resource):
    @limiter.limit("10 per hour")
    def post(self):
        """Met à jour le numéro de téléphone puis renvoie un nouveau code SMS."""
        try:
            data = request.get_json() or {}
            validated_data = validate_request(UpdateActivationPhoneSchema(), data)
            session_id = cast("str", validated_data.get("activation_session_id"))
            new_phone = cast("str", validated_data.get("phone", "")).strip()

            activation_session = ActivationSession.query.filter_by(
                activation_session_id=session_id
            ).first()
            if not activation_session:
                return {"error": "Session d'activation introuvable."}, 404

            if activation_session.consumed_at:
                return {"error": "Session d'activation déjà finalisée."}, 409

            if activation_session.phone_verified_at:
                return {"error": "Téléphone déjà confirmé."}, 409

            user = User.query.get(activation_session.user_id)
            if not user:
                return {"error": "Utilisateur introuvable."}, 404

            now = datetime.now(UTC)
            daily_count = int(activation_session.resend_count_sms or 0)
            if activation_session.last_sms_sent_at and not _is_same_utc_day(
                activation_session.last_sms_sent_at, now
            ):
                daily_count = 0

            allowed, policy_error, retry_after = _enforce_resend_policy(
                last_sent_at=activation_session.last_sms_sent_at,
                resend_count=daily_count,
            )
            if not allowed:
                message = (
                    "Veuillez patienter avant de renvoyer le SMS."
                    if policy_error == "cooldown"
                    else "Limite journalière de renvoi SMS atteinte."
                )
                return auth_error(
                    AuthErrorCodes.RATE_LIMITED,
                    message,
                    429,
                    details={"retry_after_seconds": retry_after},
                )

            user.phone = new_phone
            sms_code = _generate_sms_otp()
            activation_session.sms_code_hash = _hash_plain_value(sms_code)
            activation_session.sms_expires_at = now + timedelta(
                minutes=ACTIVATION_SMS_TTL_MINUTES
            )
            activation_session.sms_attempts = 0
            activation_session.sms_locked_until = None
            activation_session.last_sms_sent_at = now
            activation_session.resend_count_sms = daily_count + 1

            if not _send_activation_sms(user, sms_code):
                environment = (
                    str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
                )
                is_local_dev = environment == "development" and not bool(
                    current_app.config.get("TESTING")
                )
                if is_local_dev:
                    db.session.commit()
                    logger.warning(
                        "[Activation] Fallback dev update-phone SMS (session=%s): provider indisponible",
                        activation_session.activation_session_id,
                    )
                    return {
                        "message": (
                            "Numéro mis à jour. Service SMS indisponible en local, "
                            "utilisez le code de secours ci-dessous."
                        ),
                        "masked_phone": _mask_phone(user.phone),
                        "debug_sms_code": sms_code,
                        "sms_sent": False,
                        "activation_status": _build_activation_status(
                            activation_session
                        ),
                    }, 200

                db.session.rollback()
                return {"error": "Echec envoi SMS. Réessayez plus tard."}, 502

            db.session.commit()
            return {
                "message": "Numéro mis à jour. Nouveau code SMS envoyé.",
                "masked_phone": _mask_phone(user.phone),
                "activation_status": _build_activation_status(activation_session),
            }, 200
        except ValidationError as e:
            return handle_validation_error(e)
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/activation/status")
class ActivationStatus(Resource):
    @limiter.limit("60 per hour")
    def get(self):
        """Retourne l'état courant de la session d'activation."""
        session_id = (request.args.get("activation_session_id") or "").strip()
        if not session_id:
            return {"error": "activation_session_id requis."}, 400
        activation_session = ActivationSession.query.filter_by(
            activation_session_id=session_id
        ).first()
        if not activation_session:
            return {"error": "Session d'activation introuvable."}, 404
        user = User.query.get(activation_session.user_id)
        return {
            "activation_session_id": activation_session.activation_session_id,
            "masked_email": mask_email(user.email or "") if user else None,
            "masked_phone": _mask_phone(user.phone) if user else None,
            "activation_status": _build_activation_status(activation_session),
        }, 200


# ========================
# 5. Mot de Passe Oublié
# ========================
@auth_ns.route("/forgot-password")
class ForgotPassword(Resource):
    # ✅ S2: Rate limiting strict pour forgot-password (protection brute force)
    @limiter.limit("3 per minute")
    def post(self):
        """Envoie un email de réinitialisation de mot de passe."""
        try:
            data = request.get_json() or {}
            email = str(data.get("email") or "").strip()
            if not email:
                return APIErrorHandler.handle_validation_error(
                    "Email is required",
                    field="email",
                    logger_instance=logger,
                )

            user = user_repo.find_by_email(email)
            if not user:
                return {
                    "message": "Si cet email existe, un lien de réinitialisation a été envoyé.",
                    "reason": "forgot_password_email_queued",
                    "outcome_class": "success",
                    "retryable": False,
                }, 200

            # Accéder explicitement à la configuration via current_app
            secret_key = current_app.config.get("SECRET_KEY")
            if not secret_key:
                return APIErrorHandler.handle_exception(
                    Exception("Configuration error: SECRET_KEY not set"),
                    logger,
                )

            serializer = URLSafeTimedSerializer(secret_key)
            reset_token = serializer.dumps(user.email, salt="password-reset-salt")
            web_link = _build_reset_password_web_link(reset_token)
            mobile_link = _build_reset_password_deep_link(reset_token)

            send_result = send_email_notification(
                email=email,
                subject="Réinitialisation de votre mot de passe",
                body=(
                    "<p>Bonjour,</p>"
                    "<p>Cliquez sur ce lien pour réinitialiser votre mot de passe :</p>"
                    f'<p><a href="{web_link}">{web_link}</a></p>'
                    "<p>Si l'application mobile Lirie est installée, vous pouvez aussi ouvrir :</p>"
                    f"<p>{mobile_link}</p>"
                ),
                notification_type="forgot_password",
                html=True,
                from_name=os.getenv("ACTIVATION_EMAIL_FROM_NAME", "LIRIE"),
                from_email=os.getenv("ACTIVATION_EMAIL_FROM", "noreply@lirie.ch"),
                reply_to=os.getenv("ACTIVATION_EMAIL_REPLY_TO", "support@lirie.ch"),
            )
            if not send_result.get("ok"):
                environment = (
                    str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
                )
                is_local_dev = environment == "development" and not bool(
                    current_app.config.get("TESTING")
                )
                response_payload = {
                    "message": "Si cet email existe, un lien de réinitialisation a été envoyé.",
                    "reason": "forgot_password_email_unavailable",
                    "outcome_class": "success",
                    "retryable": False,
                }
                if is_local_dev:
                    logger.warning(
                        "[Auth] Fallback dev forgot-password (email=%s): %s",
                        mask_email(email),
                        send_result.get("error"),
                    )
                    response_payload = {
                        "message": (
                            "Service email indisponible en local. "
                            "Utilisez le lien de secours ci-dessous pour continuer."
                        ),
                        "reason": "forgot_password_email_local_fallback",
                        "outcome_class": "success",
                        "retryable": False,
                        "debug_reset_link": web_link,
                    }
                else:
                    logger.warning(
                        "[Auth] forgot-password email send failed (email=%s): %s",
                        mask_email(email),
                        send_result.get("error"),
                    )
                return response_payload, 200
            return {
                "message": "Si cet email existe, un lien de réinitialisation a été envoyé.",
                "reason": "forgot_password_email_queued",
                "outcome_class": "success",
                "retryable": False,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 6. Réinitialisation via Lien
# ========================


@auth_ns.route("/reset-password/<string:public_id>")
class ResetPassword(Resource):
    # ✅ S2: Rate limiting strict pour reset-password (protection brute force)
    @limiter.limit("3 per minute")
    def post(self, public_id):  # noqa: ARG002
        """Ancien reset par public_id — retiré (Lot 0 SEC-02).

        Utiliser POST /auth/change-password (session JWT) ou
        POST /auth/reset-password (token signé forgot-password).
        """
        return {
            "error": "endpoint_removed",
            "message": "Utilisez POST /auth/change-password avec votre session.",
            "reason": "reset_password_by_public_id_removed",
            "outcome_class": "terminal_error",
            "retryable": False,
        }, 410


@auth_ns.route("/change-password")
class ChangePassword(Resource):
    """Changement de mot de passe authentifié (force-reset ou volontaire)."""

    @jwt_required()
    @limiter.limit("5 per minute")
    def post(self):
        """Change le mot de passe de l'utilisateur connecté.

        Autorisé si force_password_change=True, ou si current_password est valide.
        Après succès : token_version incrémenté, sessions révoquées, re-login requis.
        """
        try:
            user = User.query.filter_by(public_id=get_jwt_identity()).first()
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    get_jwt_identity(),
                    logger,
                )

            data = request.get_json() or {}
            new_password = data.get("new_password")
            confirm_password = data.get("confirm_password")
            current_password = data.get("current_password")

            if not new_password:
                return APIErrorHandler.handle_validation_error(
                    "Un nouveau mot de passe est requis.",
                    field="new_password",
                    logger_instance=logger,
                )
            if confirm_password is not None and confirm_password != new_password:
                return APIErrorHandler.handle_validation_error(
                    "Les mots de passe ne correspondent pas.",
                    field="confirm_password",
                    logger_instance=logger,
                )

            forced = bool(getattr(user, "force_password_change", False))
            if not forced and (
                not current_password or not user.check_password(current_password)
            ):
                return auth_error(
                    AuthErrorCodes.INVALID_CREDENTIALS,
                    "Mot de passe actuel incorrect.",
                    401,
                )

            return _reset_user_password_with_policy(user, new_password)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/reset-password")
class ResetPasswordByToken(Resource):
    @limiter.limit("5 per minute")
    def post(self):
        """Réinitialise le mot de passe via token signé (web/mobile)."""
        try:
            data = request.get_json() or {}
            token = str(data.get("token") or "").strip()
            new_password = str(data.get("new_password") or "")
            missing_fields = []
            if not token:
                missing_fields.append("token")
            if not new_password:
                missing_fields.append("new_password")
            if missing_fields:
                return {
                    "error": "Payload invalide pour la réinitialisation du mot de passe.",
                    "reason": "password_reset_payload_invalid",
                    "outcome_class": "terminal_error",
                    "retryable": False,
                    "details": {"missing_fields": missing_fields},
                }, 400

            serializer = _activation_serializer()
            try:
                email_value = serializer.loads(
                    token,
                    salt="password-reset-salt",
                    max_age=RESET_PASSWORD_TOKEN_TTL_SECONDS,
                )
            except (SignatureExpired, BadSignature) as parse_error:
                if isinstance(parse_error, SignatureExpired):
                    return _password_reset_terminal_error(
                        reason="password_reset_token_expired",
                        message="Le lien de réinitialisation a expiré.",
                    )
                return _password_reset_terminal_error(
                    reason="password_reset_token_invalid",
                    message="Lien de réinitialisation invalide.",
                )

            email = str(email_value or "").strip()
            user = User.query.filter_by(email=email).first()
            if not email or not user:
                return _password_reset_terminal_error(
                    reason="password_reset_token_invalid",
                    message="Lien de réinitialisation invalide.",
                )

            return _reset_user_password_with_policy(user, new_password)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint admin : Révoquer toutes les sessions d'un utilisateur
# ========================
@auth_ns.route("/revoke-all-sessions/<int:user_id>")
class RevokeAllSessions(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @auth_ns.response(
        200, "Sessions révoquées avec succès", revoke_sessions_response_model
    )
    @auth_ns.response(401, "Non autorisé")
    @auth_ns.response(403, "Accès refusé (admin uniquement)")
    @auth_ns.response(404, "Utilisateur non trouvé")
    @auth_ns.response(500, "Erreur interne")
    # ✅ S2: Rate limiting strict pour revoke-all-sessions (action sensible)
    @limiter.limit("10 per hour")
    def post(self, user_id: int):
        """Révoque toutes les sessions actives d'un utilisateur (admin uniquement).

        Cette action déconnecte l'utilisateur de tous ses appareils en révoquant
        tous ses refresh tokens actifs dans la base de données.
        """
        try:
            # 1. Vérifier que l'utilisateur existe
            user = user_repo.find_by_id(user_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    user_id,
                    logger,
                )

            # 2. Récupérer l'admin qui effectue l'action (pour audit logging)
            admin_public_id = get_jwt_identity()
            admin_user = user_repo.find_by_public_id(admin_public_id)

            # 3. Révoquer tous les tokens de l'utilisateur
            # ✅ S3: Métrique Prometheus pour invalidation de tokens
            try:
                from security.security_metrics import (
                    security_token_invalidations_total,
                )

                security_token_invalidations_total.labels(reason="admin_revoke").inc()
            except Exception:
                pass  # Ne pas bloquer si métriques indisponibles

            count = revoke_all_user_tokens(
                user_id=user_id, reason="Révoqué par l'admin"
            )

            # 4. Audit logging
            try:
                AuditLogger.log_action(
                    action_type="revoke_all_sessions",
                    action_category="security",
                    user_id=admin_user.id if admin_user else None,
                    user_type=admin_user.role.value
                    if admin_user and admin_user.role
                    else "admin",
                    result_status="success",
                    action_details={
                        "target_user_id": user_id,
                        "target_username": user.username,
                        "target_email": mask_email(user.email) if user.email else None,
                        "sessions_revoked": count,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                # Ne pas bloquer l'action si l'audit logging échoue
                logger.warning(
                    "Échec audit logging revoke_all_sessions: %s", audit_error
                )

            # 5. Retourner la réponse
            logger.info(
                "Admin %s a révoqué %d session(s) pour user_id=%d",
                admin_public_id,
                count,
                user_id,
            )
            return {
                "message": f"{count} session(s) révoquée(s)",
                "sessions_revoked": count,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint : Lister les sessions actives de l'utilisateur
# ========================
@auth_ns.route("/sessions")
class ListSessions(Resource):
    @jwt_required()
    @auth_ns.response(
        200, "Sessions récupérées avec succès", list_sessions_response_model
    )
    @auth_ns.response(401, "Non autorisé")
    @auth_ns.response(404, "Utilisateur non trouvé")
    @auth_ns.response(500, "Erreur interne")
    # ✅ S2: Rate limiting pour sessions (endpoint utilisateur)
    @limiter.limit("50 per hour")
    def get(self):
        """Liste les sessions actives de l'utilisateur connecté.

        Retourne toutes les sessions (refresh tokens) actives de l'utilisateur
        actuellement connecté, incluant les informations sur les appareils et
        les dates de création/expiration/dernière utilisation.
        """
        try:
            # 1. Récupérer l'utilisateur connecté
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)

            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    current_user_public_id,
                    logger,
                )

            # 2. Récupérer les sessions actives
            sessions = get_user_active_sessions(user.id)

            # 3. Déterminer la session courante via le refresh token cookie (G6)
            refresh_cookie = request.cookies.get("refresh_token_cookie")
            current_hash = (
                _hash_refresh_token(refresh_cookie) if refresh_cookie else None
            )

            # 4. Sérialiser avec IP masquée et is_current
            sessions_data = [
                s.serialize_masked(current_token_hash=current_hash) for s in sessions
            ]

            # 5. Retourner la réponse
            return {
                "sessions": sessions_data,
                "count": len(sessions_data),
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint : Révoquer une session spécifique
# ========================
@auth_ns.route("/sessions/<int:session_id>")
class RevokeSession(Resource):
    @jwt_required()
    @limiter.limit("30 per hour")
    def delete(self, session_id: int):
        """Révoque une session spécifique de l'utilisateur courant.

        Protection IDOR : seul le propriétaire de la session peut la révoquer.
        Stratégie refresh-boundary (G9) : l'access token reste valide jusqu'à expiration.
        """
        try:
            from models.refresh_token import RefreshToken

            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur", current_user_public_id, logger
                )

            session = RefreshToken.query.get(session_id)
            if not session or session.user_id != user.id:
                return {"error": "Session non trouvée"}, 404

            if session.is_revoked:
                return {"error": "Session déjà révoquée"}, 409

            session.is_revoked = True
            session.revoked_at = datetime.now(UTC)
            session.revoked_reason = "Session révoquée manuellement"
            db.session.commit()

            from shared.audit_helpers import audit_log

            audit_log(
                "session_revoked",
                "security",
                resource_type="session",
                resource_id=session_id,
            )

            return "", 204

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint : Révoquer toutes les sessions sauf la courante
# ========================
@auth_ns.route("/sessions/revoke-others")
class RevokeOtherSessions(Resource):
    @jwt_required()
    @limiter.limit("10 per hour")
    def post(self):
        """Révoque toutes les sessions de l'utilisateur sauf la session courante.

        La session courante est identifiée via le refresh token cookie (G6).
        Stratégie refresh-boundary (G9).
        """
        try:
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur", current_user_public_id, logger
                )

            refresh_cookie = request.cookies.get("refresh_token_cookie")
            current_hash = (
                _hash_refresh_token(refresh_cookie) if refresh_cookie else None
            )

            now = datetime.now(UTC)
            sessions = get_user_active_sessions(user.id)
            revoked_count = 0

            for s in sessions:
                if current_hash and s.token_hash == current_hash:
                    continue
                s.is_revoked = True
                s.revoked_at = now
                s.revoked_reason = "Toutes les autres sessions révoquées"
                revoked_count += 1

            db.session.commit()

            from shared.audit_helpers import audit_log

            audit_log(
                "sessions_bulk_revoked",
                "security",
                action_details={"revoked_count": revoked_count},
            )

            return {"revoked_count": revoked_count}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint : Obtenir un token CSRF
# ========================
TOTP_CODE_LENGTH = 6
RECOVERY_CODE_LENGTH = 8
MAX_2FA_FAILURES = 10


# ========================
# Endpoints TOTP 2FA (Sprint 2)
# ========================
@auth_ns.route("/totp/setup")
class TOTPSetup(Resource):
    @jwt_required(fresh=True)
    @limiter.limit("5 per 15 minutes")
    def post(self):
        """Génère un secret TOTP et retourne le QR code + URI.

        Gardé par feature flag SECURITY_2FA_ENABLED.
        """
        if os.environ.get("SECURITY_2FA_ENABLED", "false") != "true":
            return {"error": "2FA non disponible"}, 403

        try:
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur", current_user_public_id, logger
                )

            if user.totp_enabled:
                return {"error": "2FA déjà activée. Désactivez d'abord."}, 409

            from security.totp_service import generate_totp_secret

            result = generate_totp_secret(user.email or user.username)

            user.totp_secret_encrypted = result["secret_encrypted"]
            db.session.commit()

            return {
                "provisioning_uri": result["provisioning_uri"],
                "qr_code_base64": result["qr_code_base64"],
                "secret_display": result["secret_display"],
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/totp/verify")
class TOTPVerify(Resource):
    @jwt_required(fresh=True)
    @limiter.limit("5 per 15 minutes")
    def post(self):
        """Vérifie un code TOTP et active le 2FA. Retourne les recovery codes."""
        if os.environ.get("SECURITY_2FA_ENABLED", "false") != "true":
            return {"error": "2FA non disponible"}, 403

        try:
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur", current_user_public_id, logger
                )

            data = request.get_json() or {}
            code = str(data.get("code", "")).strip()
            if not code or len(code) != TOTP_CODE_LENGTH:
                return {"error": "Code à 6 chiffres requis"}, 400

            if not user.totp_secret_encrypted:
                return {"error": "Appelez /totp/setup d'abord"}, 400

            from security.totp_service import (
                generate_recovery_codes,
                verify_totp_code,
            )

            if not verify_totp_code(user.totp_secret_encrypted, code):
                return {"error": "Code invalide"}, 401

            codes, hashes_json = generate_recovery_codes()
            user.totp_enabled = True
            user.totp_enabled_at = datetime.now(UTC)
            user.recovery_codes_hash = hashes_json
            user.recovery_codes_remaining = len(codes)
            db.session.commit()

            from shared.audit_helpers import audit_log

            audit_log("totp_enabled", "security")

            return {
                "message": "Validation en deux étapes activée.",
                "recovery_codes": codes,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/totp/disable")
class TOTPDisable(Resource):
    @jwt_required(fresh=True)
    @limiter.limit("5 per 15 minutes")
    def post(self):
        """Désactive le 2FA. Requiert le mot de passe actuel."""
        if os.environ.get("SECURITY_2FA_ENABLED", "false") != "true":
            return {"error": "2FA non disponible"}, 403

        try:
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur", current_user_public_id, logger
                )

            data = request.get_json() or {}
            password = data.get("password", "")
            if not password or not user.check_password(password):
                return {"error": "Mot de passe incorrect"}, 401

            user.totp_enabled = False
            user.totp_secret_encrypted = None
            user.totp_enabled_at = None
            user.recovery_codes_hash = None
            user.recovery_codes_remaining = 0
            db.session.commit()

            from shared.audit_helpers import audit_log

            audit_log("totp_disabled", "security")

            return {"message": "Validation en deux étapes désactivée."}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/totp/status")
class TOTPStatus(Resource):
    @jwt_required()
    def get(self):
        """Retourne le statut 2FA de l'utilisateur courant."""
        try:
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur", current_user_public_id, logger
                )

            return {
                "enabled": bool(user.totp_enabled),
                "enabled_at": user.totp_enabled_at.isoformat()
                if user.totp_enabled_at
                else None,
                "recovery_codes_remaining": user.recovery_codes_remaining or 0,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/totp/recovery-codes")
class TOTPRecoveryCodes(Resource):
    @jwt_required(fresh=True)
    @limiter.limit("5 per 15 minutes")
    def post(self):
        """Régénère les codes de secours. Requiert un code TOTP valide."""
        if os.environ.get("SECURITY_2FA_ENABLED", "false") != "true":
            return {"error": "2FA non disponible"}, 403

        try:
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur", current_user_public_id, logger
                )

            if not user.totp_enabled or not user.totp_secret_encrypted:
                return {"error": "2FA non activée"}, 400

            data = request.get_json() or {}
            code = str(data.get("code", "")).strip()

            from security.totp_service import generate_recovery_codes, verify_totp_code

            if not verify_totp_code(user.totp_secret_encrypted, code):
                return {"error": "Code TOTP invalide"}, 401

            codes, hashes_json = generate_recovery_codes()
            user.recovery_codes_hash = hashes_json
            user.recovery_codes_remaining = len(codes)
            db.session.commit()

            from shared.audit_helpers import audit_log

            audit_log("recovery_codes_regenerated", "security")

            return {
                "message": "Codes de secours régénérés.",
                "recovery_codes": codes,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@auth_ns.route("/totp/challenge")
class TOTPChallenge(Resource):
    @limiter.limit("5 per 15 minutes")
    def post(self):
        """Vérifie un code TOTP après login (temp_token anti-replay G7).

        Reçoit temp_token + code TOTP, retourne les vrais tokens si valide.
        """
        if os.environ.get("SECURITY_2FA_ENABLED", "false") != "true":
            return {"error": "2FA non disponible"}, 403

        try:
            data = request.get_json() or {}
            temp_token = data.get("temp_token", "")
            code = str(data.get("code", "")).strip()

            if not temp_token or not code:
                return {"error": "temp_token et code requis"}, 400

            decoded = decode_token(temp_token)
            if decoded.get("purpose") != "2fa_challenge":
                return {"error": "Token invalide"}, 401

            jti = decoded.get("jti")
            user_public_id = decoded.get("sub")

            if not jti or not user_public_id:
                return {"error": "Token invalide"}, 401

            from security.totp_service import (
                check_2fa_lockout,
                consume_2fa_challenge_jti,
                record_2fa_failure,
                reset_2fa_failures,
                verify_recovery_code,
                verify_totp_code,
            )

            if not consume_2fa_challenge_jti(jti):
                return {"error": "Token déjà utilisé ou expiré"}, 401

            user = user_repo.find_by_public_id(user_public_id)
            if not user:
                return {"error": "Utilisateur non trouvé"}, 404

            if check_2fa_lockout(user.id):
                return {"error": "Trop de tentatives. Réessayez dans 30 minutes."}, 429

            is_valid = False
            if len(code) == TOTP_CODE_LENGTH and code.isdigit():
                is_valid = verify_totp_code(user.totp_secret_encrypted, code)
            elif len(code) == RECOVERY_CODE_LENGTH and code.isdigit():
                is_valid, updated_hashes = verify_recovery_code(
                    user.recovery_codes_hash or "[]", code
                )
                if is_valid:
                    user.recovery_codes_hash = updated_hashes
                    user.recovery_codes_remaining = max(
                        0, (user.recovery_codes_remaining or 0) - 1
                    )

            if not is_valid:
                failures = record_2fa_failure(user.id)
                from shared.audit_helpers import audit_log

                audit_log(
                    "totp_challenge_failed", "security", user=user, result="failure"
                )
                if failures >= MAX_2FA_FAILURES:
                    return {
                        "error": "Trop de tentatives. Réessayez dans 30 minutes."
                    }, 429
                return {"error": "Code invalide"}, 401

            reset_2fa_failures(user.id)

            # Charger le modèle User pour token_version
            db_user = User.query.filter_by(public_id=user.public_id).first() or user
            additional_claims = {
                "role": user.role.value if user.role else "unknown",
                "aud": "atmr-api",
                "token_version": _user_token_version(db_user),
            }
            if user.company:
                additional_claims["company_id"] = user.company.id

            access_token = create_access_token(
                identity=user.public_id,
                additional_claims=additional_claims,
                fresh=True,
            )
            refresh_token = create_refresh_token(
                identity=user.public_id,
                additional_claims={
                    "aud": "atmr-api",
                    "token_version": _user_token_version(db_user),
                },
            )

            device_id = request.headers.get("X-Device-Id")
            from security.refresh_token_service import store_refresh_token
            from shared.security_helpers import parse_device

            store_refresh_token(
                token=refresh_token,
                user_id=user.id,
                expires_at=datetime.now(UTC) + timedelta(days=30),
                device_id=device_id,
                device_name=parse_device(request.headers.get("User-Agent")),
            )

            db.session.commit()

            from shared.audit_helpers import audit_log

            audit_log("user_login", "security", user=user)

            resp = make_response(
                {
                    "message": "2FA validée",
                    "user": {
                        "id": user.id,
                        "public_id": user.public_id,
                        "username": user.username,
                        "email": user.email,
                        "role": user.role.value if user.role else None,
                    },
                    "token": access_token,
                    "refresh_token": refresh_token,
                }
            )

            from services.security.csrf import generate_csrf_token

            csrf_token = generate_csrf_token()
            resp.set_cookie(
                "csrf_token",
                csrf_token,
                httponly=False,
                samesite="Lax",
                secure=False,
                path="/",
            )
            resp.set_cookie(
                "access_token_cookie",
                access_token,
                httponly=True,
                samesite="Lax",
                secure=False,
                path="/",
            )
            resp.set_cookie(
                "refresh_token_cookie",
                refresh_token,
                httponly=True,
                samesite="Lax",
                secure=False,
                path="/api",
            )

            return resp

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint : Obtenir un token CSRF
# ========================
@auth_ns.route("/csrf-token")
@auth_ns.doc(
    security=None,  # Endpoint public (pas besoin d'authentification)
    description="Récupère un token CSRF pour protéger les requêtes mutantes",
)
class CSRFTokenResource(Resource):
    """Endpoint pour obtenir un token CSRF."""

    @auth_ns.marshal_with(csrf_token_response_model)
    @auth_ns.response(200, "Token CSRF généré avec succès")
    @auth_ns.response(500, "Erreur interne")
    # ✅ S2: Rate limiting plus strict pour endpoint CSRF (protection contre abus)
    @limiter.limit("50 per hour")
    def get(self):
        """Génère et retourne un token CSRF.

        Le token doit être inclus dans toutes les requêtes mutantes (POST/PUT/PATCH/DELETE)
        via le header `X-CSRF-Token` ou dans le body JSON avec la clé `csrf_token`.

        Returns:
            Dict avec le token CSRF et sa durée de vie
        """
        import os

        try:
            # Récupérer user_id depuis JWT si disponible (optionnel)
            user_id = None
            try:
                jwt_identity = get_jwt_identity()
                if jwt_identity:
                    if isinstance(jwt_identity, dict):
                        # Extraire user_id depuis les claims JWT
                        from flask_jwt_extended import (
                            get_jwt,
                        )

                        jwt_claims = get_jwt()
                        user_id = jwt_claims.get("user_id") or jwt_claims.get("id")
                        if not user_id:
                            user_id = jwt_identity.get("user_id") or jwt_identity.get(
                                "id"
                            )
                    elif isinstance(jwt_identity, (int, str)):
                        with suppress(ValueError, TypeError):
                            user_id = int(jwt_identity)
            except Exception:
                # JWT non disponible, générer token sans user_id
                pass

            # Générer le token CSRF
            csrf_token = generate_csrf_token(user_id=user_id)

            # TTL du token (en secondes)
            csrf_ttl = int(os.getenv("CSRF_TOKEN_TTL_SECONDS", "3600"))

            logger.debug("[CSRF] Token généré pour user_id=%s", user_id)

            return {
                "csrf_token": csrf_token,
                "ttl": csrf_ttl,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# INVITATION - Endpoints publics (pas de JWT requis)
# ============================================================================


@auth_ns.route("/invite/<string:token>")
class InviteVerify(Resource):
    """Endpoint public pour vérifier un token d'invitation."""

    @auth_ns.doc(description="Vérifie la validité d'un token d'invitation.")
    @auth_ns.response(200, "Token valide")
    @auth_ns.response(400, "Token invalide ou expiré")
    @limiter.limit("20 per hour")
    def get(self, token):
        """Vérifie un token d'invitation (public, pas de JWT).

        Retourne les infos de base si le token est valide.
        """
        from application.institutions.invitation_service import hash_token

        try:
            token_hash = hash_token(token)

            user = User.query.filter_by(invite_token_hash=token_hash).first()

            # Message générique pour ne pas fuiter d'info
            generic_error = "Ce lien d'invitation est invalide ou a expiré."

            if not user:
                return {"error": generic_error, "code": "invalid_token"}, 400

            # Vérifier expiration
            if user.invite_expires_at and user.invite_expires_at < datetime.now(UTC):
                return {
                    "error": "Ce lien d'invitation a expiré. Demandez à votre administrateur d'en envoyer un nouveau.",
                    "code": "expired",
                }, 400

            # Vérifier que l'utilisateur est bien en statut "invited"
            if user.account_status not in ("invited", None):
                return {
                    "error": "Ce compte a déjà été activé.",
                    "code": "already_activated",
                }, 400

            # Retourner les infos de base (pas de données sensibles)
            institution_name = None
            if user.institution_id:
                from models.institution import Institution

                inst = Institution.query.get(user.institution_id)
                institution_name = inst.name if inst else None

            return {
                "valid": True,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "institution_name": institution_name,
                "role": user.institution_role,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Auth] Erreur vérification invite token: %s", e)
            return {"error": "Erreur lors de la vérification"}, 500


@auth_ns.route("/activate-account")
class ActivateAccount(Resource):
    """Endpoint public pour activer un compte via invitation."""

    @auth_ns.doc(description="Active un compte invité en définissant le mot de passe.")
    @auth_ns.response(200, "Compte activé avec succès")
    @auth_ns.response(400, "Token invalide, expiré ou données manquantes")
    @limiter.limit("10 per hour")
    def post(self):
        """Active un compte invité (public, pas de JWT).

        Body JSON:
            token (str): Token d'invitation brut
            password (str): Nouveau mot de passe (min 8 caractères)
        """
        from application.institutions.invitation_service import hash_token

        try:
            data = request.get_json() or {}
            token = data.get("token", "").strip()
            password = data.get("password", "")

            min_password_length = 8
            if not token or not password or len(password) < min_password_length:
                error_msg = (
                    "Token manquant"
                    if not token
                    else f"Le mot de passe doit contenir au moins {min_password_length} caractères"
                )
                return {"error": error_msg}, 400

            token_hash = hash_token(token)
            user = User.query.filter_by(invite_token_hash=token_hash).first()

            generic_error = "Ce lien d'invitation est invalide ou a expiré."

            if not user:
                return {"error": generic_error, "code": "invalid_token"}, 400

            # Vérifier expiration
            if user.invite_expires_at and user.invite_expires_at < datetime.now(UTC):
                return {
                    "error": "Ce lien d'invitation a expiré. Demandez à votre administrateur d'en envoyer un nouveau.",
                    "code": "expired",
                }, 400

            # Vérifier statut
            if user.account_status not in ("invited", None):
                return {
                    "error": "Ce compte a déjà été activé.",
                    "code": "already_activated",
                }, 400

            # Activer le compte
            from security.password_policy import (
                PasswordPolicyError,
                PasswordPolicyService,
            )

            try:
                PasswordPolicyService.validate_password(
                    password, user_id=user.id, check_history=True
                )
            except PasswordPolicyError as e:
                return {"error": str(e), "code": "weak_password"}, 400

            user.set_password(  # nosemgrep: python.django.security.audit.unvalidated-password.unvalidated-password
                password
            )
            user.account_status = "active"
            user.force_password_change = False
            if (
                hasattr(user, "first_login_completed_at")
                and not user.first_login_completed_at
                and user.institution_id
                and getattr(user, "authentication_method", "email") == "username"
            ):
                user.first_login_completed_at = datetime.now(UTC)
            # Invalider le token (one-time use)
            user.invite_token_hash = None
            user.invite_expires_at = None

            db.session.commit()

            # Audit log
            try:
                from security.audit_log import AuditLogger

                AuditLogger.log_action(
                    action_type="institution_account_activated",
                    action_category="authentication",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=user.institution_id,
                    result_status="success",
                    action_details={
                        "email": user.email,
                        "institution_role": user.institution_role,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[Auth] Audit log activate failed: %s", audit_err)

            logger.info(
                "[Auth] Compte activé: user_id=%s, email=%s",
                user.id,
                user.email,
            )

            # Auto-login : générer un JWT pour connexion immédiate
            claims = {
                "user_id": user.id,
                "role": user.role.value
                if hasattr(user.role, "value")
                else str(user.role),
                "company_id": _resolve_company_id(user),
                "driver_id": getattr(user, "driver_id", None),
                "institution_id": getattr(user, "institution_id", None),
                "institution_role": getattr(user, "institution_role", None),
                "aud": "atmr-api",
                "token_version": _user_token_version(user),
            }
            access_token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
                fresh=True,
            )
            refresh_token = create_refresh_token(
                identity=str(user.public_id),
                additional_claims={
                    "aud": "atmr-api",
                    "token_version": _user_token_version(user),
                },
            )

            return {
                "message": "Compte activé avec succès.",
                "email": user.email,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": user.serialize,
            }, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[Auth] Erreur activation compte: %s", e)
            return {"error": "Erreur lors de l'activation"}, 500
