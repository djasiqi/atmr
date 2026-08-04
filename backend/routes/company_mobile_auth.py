"""Routes d'authentification mobile pour les entreprises (dispatch)."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    cast,
)

import jwt
import sentry_sdk
from flask import current_app, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (
    Namespace,
    Resource,
    fields,
)
from marshmallow import (
    Schema,
    ValidationError,
    validate,
)
from marshmallow import fields as ma_fields

from ext import limiter, redis_client
from models import Company, User, UserRole
from models.enums import DriverType
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import (
    get_current_user_via_use_case,
)

logger = logging.getLogger(__name__)


MOBILE_AUDIENCE = "atmr-mobile-enterprise"
MFA_CHALLENGE_TTL = 300  # 5 minutes
MFA_CHALLENGE_PREFIX = "company_mobile:mfa:challenge:"
FAILED_LOGIN_PREFIX = "company_mobile:failed_login:"
MAX_FAILED_ATTEMPTS = 5
FAILED_LOGIN_TTL = 900  # 15 minutes
DEFAULT_SCOPES = ["enterprise.dispatch:read", "enterprise.dispatch:write"]

# Constantes pour sanitization des logs
TOKEN_MIN_LENGTH_FOR_MASKING = 50
EMAIL_USERNAME_MIN_LENGTH = 2
LOG_MAX_LENGTH = 100

# Codes HTTP pour gestion des erreurs
HTTP_INTERNAL_ERROR = 500
HTTP_UNAUTHORIZED = 401

# Whitelist des providers OIDC autorisés (configurable via env)
ALLOWED_OIDC_PROVIDERS = set(
    os.getenv("ALLOWED_OIDC_PROVIDERS", "").split(",")
    if os.getenv("ALLOWED_OIDC_PROVIDERS")
    else []
)
OIDC_JWKS_URL = os.getenv("OIDC_JWKS_URL", "").strip()
try:
    OIDC_JWKS_BY_ISSUER = json.loads(os.getenv("OIDC_JWKS_BY_ISSUER", "{}") or "{}")
except (TypeError, ValueError):
    OIDC_JWKS_BY_ISSUER = {}


company_mobile_auth_ns = Namespace(
    "company_mobile_auth", description="Authentification mobile entreprise (dispatch)"
)


# ====== Modèles Swagger ======
login_model = company_mobile_auth_ns.model(
    "EnterpriseMobileLogin",
    {
        "method": fields.String(
            required=False,
            default="password",
            enum=["password", "oidc"],
            description="Méthode d'authentification (password ou oidc)",
        ),
        "email": fields.String(
            description="Email entreprise (pour login/mot de passe)"
        ),
        "password": fields.String(description="Mot de passe (si method=password)"),
        "id_token": fields.String(description="ID token OIDC (si method=oidc)"),
        "provider": fields.String(description="Identifiant fournisseur OIDC/SAML"),
        "mfa_code": fields.String(description="Code MFA TOTP"),
        "device_id": fields.String(description="Identifiant appareil (MDM)"),
    },
)

mfa_verify_model = company_mobile_auth_ns.model(
    "EnterpriseMobileMfaVerify",
    {
        "challenge_id": fields.String(
            required=True, description="Identifiant du challenge MFA"
        ),
        "code": fields.String(required=True, description="Code TOTP à 6 chiffres"),
        "device_id": fields.String(description="Identifiant appareil (optionnel)"),
    },
)

refresh_model = company_mobile_auth_ns.model(
    "EnterpriseMobileRefresh",
    {
        "refresh_token": fields.String(
            required=True, description="Refresh token valide"
        ),
    },
)


# ====== Schemas Marshmallow ======
class EnterpriseLoginSchema(Schema):
    method = ma_fields.String(
        load_default="password", validate=validate.OneOf(["password", "oidc"])
    )
    email = ma_fields.Email(load_default=None)
    password = ma_fields.String(load_default=None)
    id_token = ma_fields.String(load_default=None)
    provider = ma_fields.String(load_default=None)
    mfa_code = ma_fields.String(load_default=None)
    device_id = ma_fields.String(load_default=None)


class EnterpriseMfaVerifySchema(Schema):
    challenge_id = ma_fields.UUID(required=True)
    code = ma_fields.String(required=True, validate=validate.Length(min=4, max=10))
    device_id = ma_fields.String(load_default=None)


class EnterpriseRefreshSchema(Schema):
    refresh_token = ma_fields.String(required=True)


# ====== Helpers ======
def _get_company_security(company: Company | None) -> Dict[str, Any]:
    if not company or not company.autonomous_config:
        return {}
    try:
        payload = json.loads(company.autonomous_config)
    except (ValueError, TypeError):
        warning_msg = (
            "[AUTH][Enterprise] Impossible de parser autonomous_config "
            + "pour company_id=%s"
        )
        logger.warning(
            warning_msg,
            getattr(company, "id", None),
        )
        return {}
    return payload.get("security", {}) or {}


def _company_requires_mfa(company: Company | None) -> bool:
    if _is_reviewer_company(company):
        logger.info(
            "[AUTH][Enterprise] MFA bypass activé pour company reviewer (company_id=%s)",
            getattr(company, "id", None),
        )
        return False
    security = _get_company_security(company)
    policy = security.get("mobile_mfa") or {}
    return bool(policy.get("required", False))


def _is_reviewer_company(company: Company | None) -> bool:
    if not company:
        return False
    reviewer_company_id = os.getenv("REVIEWER_COMPANY_ID", "").strip()
    reviewer_company_name = os.getenv("REVIEWER_COMPANY_NAME", "").strip().lower()

    if reviewer_company_id and str(getattr(company, "id", "")) == reviewer_company_id:
        return True
    return bool(
        reviewer_company_name
        and str(getattr(company, "name", "")).strip().lower() == reviewer_company_name
    )


def _get_totp_secret(company: Company | None) -> str | None:
    security = _get_company_security(company)
    policy = security.get("mobile_mfa") or {}
    secret = policy.get("totp_secret")
    if isinstance(secret, str) and secret.strip():
        return secret.strip()
    return None


def _sanitize_log_data(data: Any) -> str:
    """Sanitise les données pour les logs (évite fuite d'infos sensibles)."""
    if isinstance(data, str):
        # Masquer les tokens (longue chaîne alphanumérique)
        if len(data) > TOKEN_MIN_LENGTH_FOR_MASKING and re.match(
            r"^[A-Za-z0-9._-]+$", data
        ):
            return f"{data[:10]}...{data[-5:]}"
        # Masquer les emails partiellement
        if "@" in data:
            parts = data.split("@")
            EMAIL_PARTS_COUNT = 2
            if len(parts) == EMAIL_PARTS_COUNT:
                username = parts[0]
                domain = parts[1]
                if len(username) > EMAIL_USERNAME_MIN_LENGTH:
                    return f"{username[:EMAIL_USERNAME_MIN_LENGTH]}***@{domain}"
                return f"***@{domain}"
    return str(data)[:LOG_MAX_LENGTH]  # Limiter la longueur


def _check_failed_login_attempts(email: str) -> Tuple[bool, int]:
    """Vérifie le nombre de tentatives de connexion échouées.

    Returns:
        Tuple[bool, int]: (dépassé la limite, nombre de tentatives)
    """
    if not redis_client:
        return False, 0
    key = f"{FAILED_LOGIN_PREFIX}{email.lower()}"
    attempts = redis_client.get(key)
    if attempts:
        # Convertir bytes en int si nécessaire
        attempts_str = (
            attempts.decode("utf-8") if isinstance(attempts, bytes) else str(attempts)
        )
        count = int(attempts_str)
        return count >= MAX_FAILED_ATTEMPTS, count
    return False, 0


def _increment_failed_login(email: str) -> int:
    """Incrémente le compteur de tentatives échouées.

    Returns:
        Nombre total de tentatives après incrément
    """
    if not redis_client:
        return 0
    key = f"{FAILED_LOGIN_PREFIX}{email.lower()}"
    count_result = redis_client.incr(key)
    # Convertir en int si nécessaire (Redis peut retourner bytes ou int)
    if isinstance(count_result, bytes):
        count = int(count_result.decode("utf-8"))
    elif isinstance(count_result, (int, str)):
        count = int(count_result)
    else:
        # Fallback pour types inattendus
        count = int(str(count_result))
    if count == 1:  # Première tentative, définir TTL
        redis_client.expire(key, FAILED_LOGIN_TTL)
    return count


def _reset_failed_login(email: str) -> None:
    """Réinitialise le compteur de tentatives échouées."""
    if redis_client:
        key = f"{FAILED_LOGIN_PREFIX}{email.lower()}"
        redis_client.delete(key)


def _validate_device_id(device_id: str | None) -> bool:
    """Valide le format du device_id (UUID ou format MDM valide).

    Args:
        device_id: Identifiant de l'appareil

    Returns:
        True si valide, False sinon
    """
    if not device_id:
        return True  # Optionnel
    # UUID format ou format MDM (alphanumérique avec tirets)
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    mdm_pattern = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
    return bool(uuid_pattern.match(device_id) or mdm_pattern.match(device_id))


def _verify_totp_code(company: Company | None, code: str) -> bool:
    secret = _get_totp_secret(company)
    if not secret:
        logger.warning(
            "[AUTH][Enterprise] Aucun secret TOTP configuré pour company_id=%s",
            getattr(company, "id", None),
        )
        return False
    try:
        import pyotp  # type: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover - dépendance optionnelle
        logger.error(
            "[AUTH][Enterprise] pyotp requis pour vérifier le code MFA (company_id=%s)",
            getattr(company, "id", None),
        )
        return False
    totp = pyotp.TOTP(secret)
    return bool(totp.verify(code, valid_window=1))


def _issue_tokens(
    user: User,
    company: Company,
    device_id: str | None = None,
    extra_scopes: List[str] | None = None,
    session_id: str | None = None,
) -> Dict[str, Any]:
    scopes = list(DEFAULT_SCOPES)
    if extra_scopes:
        for scope in extra_scopes:
            if scope not in scopes:
                scopes.append(scope)

    session_identifier = session_id or str(uuid.uuid4())

    additional_claims = {
        "role": user.role.value,
        "company_id": company.id,
        "aud": MOBILE_AUDIENCE,
        "scopes": scopes,
        "session_id": session_identifier,
    }
    if device_id:
        additional_claims["device_id"] = device_id

    access_expires = current_app.config.get(
        "JWT_MOBILE_ACCESS_TOKEN_EXPIRES",
        current_app.config.get("JWT_ACCESS_TOKEN_EXPIRES", timedelta(hours=1)),
    )
    access_token = create_access_token(
        identity=str(user.public_id),
        additional_claims=additional_claims,
        expires_delta=access_expires,
    )
    refresh_expires = current_app.config.get(
        "JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=90)
    )
    refresh_token = create_refresh_token(
        identity=str(user.public_id),
        additional_claims={
            "aud": MOBILE_AUDIENCE,
            "session_id": session_identifier,
        },
        expires_delta=refresh_expires,
    )

    return {
        "token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "id": user.id,
            "public_id": user.public_id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role.value,
        },
        "company": {
            "id": company.id,
            "name": company.name,
            "dispatch_mode": company.dispatch_mode.value,
        },
        "scopes": scopes,
        "session_id": session_identifier,
    }


def _store_mfa_challenge(
    user: User,
    company: Company,
    method: str,
    device_id: str | None,
) -> str:
    challenge_id = str(uuid.uuid4())
    payload = {
        "user_public_id": user.public_id,
        "company_id": company.id,
        "method": method,
        "device_id": device_id,
    }
    if redis_client:
        redis_client.setex(
            f"{MFA_CHALLENGE_PREFIX}{challenge_id}",
            MFA_CHALLENGE_TTL,
            json.dumps(payload),
        )
    else:  # pragma: no cover - fallback
        warning_msg = (
            "[AUTH][Enterprise] Redis indisponible, "
            + "impossible de stocker le challenge MFA."
        )
        logger.warning(warning_msg)
    return challenge_id


def _consume_mfa_challenge(challenge_id: str) -> Dict[str, Any] | None:
    key = f"{MFA_CHALLENGE_PREFIX}{challenge_id}"
    if not redis_client:
        logger.error("[AUTH][Enterprise] Redis requis pour vérifier le challenge MFA.")
        return None
    data = redis_client.get(key)
    if not data:
        return None
    redis_client.delete(key)
    try:
        # Convert bytes to str if necessary
        data_str = data.decode("utf-8") if isinstance(data, bytes) else str(data)
        return json.loads(data_str)
    except (ValueError, TypeError):
        logger.error("[AUTH][Enterprise] Challenge MFA corrompu (%s).", challenge_id)
        return None


def _find_company_user_by_email(email: str) -> Tuple[User | None, Company | None]:
    # Utiliser le repository pour récupérer l'utilisateur
    from repositories.user_repository import UserRepository

    user_repo = UserRepository()
    # ADMIN plateforme exclu : ne peut pas se connecter à l'espace entreprise
    user = user_repo.find_by_email_with_role_filter(email, (UserRole.COMPANY,))
    if not user:
        return None, None
    if user.role == UserRole.ADMIN:
        return None, None
    company = user.company
    if not company:
        return None, None
    return user, company


def _handle_oidc_login(
    id_token: str, provider: str | None
) -> Tuple[User | None, Company | None]:
    """Gère la connexion OIDC avec validation de sécurité renforcée.

    Args:
        id_token: Token OIDC ID
        provider: Nom du provider OIDC

    Returns:
        Tuple[User, Company] si succès

    Raises:
        ValueError: Si le token est invalide ou non autorisé
    """
    if not id_token:
        raise ValueError("Token OIDC manquant.")

    # Étape 1: lecture non vérifiée minimale pour récupérer issuer/provider
    try:
        unverified_claims = jwt.decode(
            id_token,
            options={
                "verify_signature": False,
                "verify_aud": False,
                "verify_exp": False,
            },
        )
        unverified_header = jwt.get_unverified_header(id_token)
    except jwt.PyJWTError as exc:
        logger.warning(
            "[AUTH][Enterprise] Echec décodage token OIDC: %s",
            _sanitize_log_data(str(exc)),
        )
        raise ValueError("ID token invalide.") from exc

    issuer = unverified_claims.get("iss")
    jwks_url = ""
    if isinstance(OIDC_JWKS_BY_ISSUER, dict) and issuer in OIDC_JWKS_BY_ISSUER:
        jwks_url = str(OIDC_JWKS_BY_ISSUER.get(issuer) or "").strip()
    if not jwks_url:
        jwks_url = OIDC_JWKS_URL
    if not jwks_url:
        logger.error(
            "[AUTH][Enterprise] OIDC_JWKS_URL/OIDC_JWKS_BY_ISSUER non configuré"
        )
        raise ValueError("Configuration OIDC incomplète (JWKS manquant).")

    alg = str(unverified_header.get("alg") or "")
    if alg not in {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}:
        logger.warning("[AUTH][Enterprise] Algorithme OIDC non autorisé: %s", alg)
        raise ValueError("ID token invalide (algorithme non autorisé).")

    # Étape 2: validation cryptographique via JWKS
    try:
        jwk_client = jwt.PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(id_token)
        decoded = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=[alg],
            options={"verify_aud": False},
        )
    except jwt.PyJWTError as exc:
        logger.warning(
            "[AUTH][Enterprise] Signature OIDC invalide: %s",
            _sanitize_log_data(str(exc)),
        )
        raise ValueError("ID token invalide (signature).") from exc

    # Vérifier l'issuer (provider)
    if provider and ALLOWED_OIDC_PROVIDERS:
        if provider not in ALLOWED_OIDC_PROVIDERS:
            logger.warning(
                "[AUTH][Enterprise] Provider OIDC non autorisé: %s",
                _sanitize_log_data(provider),
            )
            raise ValueError("Provider OIDC non autorisé.")
    elif issuer and ALLOWED_OIDC_PROVIDERS:
        # Vérifier si l'issuer est dans la whitelist
        issuer_allowed = any(
            issuer.startswith(allowed) for allowed in ALLOWED_OIDC_PROVIDERS
        )
        if not issuer_allowed:
            logger.warning(
                "[AUTH][Enterprise] Issuer OIDC non autorisé: %s",
                _sanitize_log_data(issuer),
            )
            raise ValueError("Issuer OIDC non autorisé.")

    # Vérifier l'audience si présente
    aud = decoded.get("aud")
    if aud and aud != MOBILE_AUDIENCE:
        logger.warning(
            "[AUTH][Enterprise] Audience OIDC incorrecte: %s (attendu: %s)",
            _sanitize_log_data(str(aud)),
            MOBILE_AUDIENCE,
        )
        # Ne pas bloquer, juste logger (certains providers n'incluent pas aud)

    email = decoded.get("email")
    if not email:
        raise ValueError("ID token ne contient pas d'email.")

    user, company = _find_company_user_by_email(email)
    if not user or not company:
        raise ValueError("Utilisateur OIDC non autorisé pour l'app entreprise.")

    logger.info(
        "[AUTH][Enterprise] Connexion OIDC réussie pour user_id=%s provider=%s",
        user.id,
        _sanitize_log_data(provider or issuer or "unknown"),
    )
    return user, company


def _ensure_company_operator_driver_for_user(user: Any) -> None:
    """Crée une seule fois le profil chauffeur lié au compte entreprise dispatch."""
    user_model = user if isinstance(user, User) else None
    if user_model is None:
        public_id = getattr(user, "public_id", None)
        user_id = getattr(user, "id", None)
        if public_id:
            user_model = User.query.filter_by(public_id=str(public_id)).first()
        elif user_id is not None:
            user_model = User.query.filter_by(id=int(user_id)).first()
    if user_model is None:
        return

    from application.companies.drivers.ensure_company_operator_driver import (
        EnsureCompanyOperatorDriverUseCase,
    )
    from ext import db

    result = EnsureCompanyOperatorDriverUseCase().execute(user_model)
    if result.created:
        db.session.commit()


# ====== Resources ======
@company_mobile_auth_ns.route("/login")
class EnterpriseMobileLogin(Resource):
    @company_mobile_auth_ns.expect(login_model, validate=True)
    @limiter.limit("10/minute")
    def post(self):
        payload = request.get_json() or {}
        try:
            data = EnterpriseLoginSchema().load(payload)
        except ValidationError as exc:
            return APIErrorHandler.handle_exception(exc, logger)

        # Type assertion: load() returns a dict
        assert isinstance(data, dict), "Schema load should return a dict"
        method: str = data["method"]
        email = data.get("email")
        password = data.get("password")
        id_token = data.get("id_token")
        mfa_code = data.get("mfa_code")
        device_id = data.get("device_id")

        user: User | None = None
        company: Company | None = None
        result: Tuple[Dict[str, Any], int] = (
            {"error": "Erreur interne."},
            HTTP_INTERNAL_ERROR,
        )

        if method == "password":
            if not email or not password:
                result = ({"error": "Email et mot de passe requis."}, 400)
            else:
                # Vérifier les tentatives échouées
                blocked, attempts = _check_failed_login_attempts(email)
                if blocked:
                    logger.warning(
                        (
                            "[AUTH][Enterprise] Trop de tentatives échouées pour %s "
                            "(%d tentatives)"
                        ),
                        _sanitize_log_data(email),
                        attempts,
                    )
                    result = (
                        {
                            "error": (
                                "Trop de tentatives de connexion échouées. "
                                "Réessayez dans 15 minutes."
                            ),
                            "retry_after": FAILED_LOGIN_TTL,
                        },
                        429,
                    )
                else:
                    user, company = _find_company_user_by_email(email)
                    if not user or not company or not user.check_password(password):
                        # Incrémenter le compteur d'échecs
                        new_count = _increment_failed_login(email)
                        logger.warning(
                            (
                                "[AUTH][Enterprise] Tentative de connexion échouée "
                                "pour %s (tentative %d/%d)"
                            ),
                            _sanitize_log_data(email),
                            new_count,
                            MAX_FAILED_ATTEMPTS,
                        )
                        result = ({"error": "Identifiants invalides."}, 401)
                    else:
                        # Réinitialiser le compteur en cas de succès
                        _reset_failed_login(email)
        elif method == "oidc":
            if not id_token:
                result = ({"error": "ID token requis pour OIDC."}, 400)
            else:
                try:
                    user, company = _handle_oidc_login(id_token, data.get("provider"))
                except ValueError as exc:
                    result = ({"error": str(exc)}, 401)

        # Vérifier si une erreur s'est produite
        if result[1] != HTTP_INTERNAL_ERROR:
            return result

        if not user or not company:
            result = ({"error": "Accès refusé."}, 403)
        elif device_id and not _validate_device_id(device_id):
            logger.warning(
                "[AUTH][Enterprise] Device ID invalide pour user_id=%s: %s",
                user.id if user else "unknown",
                _sanitize_log_data(device_id),
            )
            result = ({"error": "Format device_id invalide."}, 400)
        else:
            requires_mfa = _company_requires_mfa(company)
            if requires_mfa:
                if mfa_code:
                    if not _verify_totp_code(company, mfa_code):
                        result = ({"error": "Code MFA invalide."}, 401)
                    else:
                        # MFA vérifié, continuer avec l'émission de tokens
                        _ensure_company_operator_driver_for_user(user)
                        response = _issue_tokens(user, company, device_id)
                        response["mfa_required"] = False
                        result = (response, 200)
                else:
                    challenge_id = _store_mfa_challenge(
                        user, company, method, device_id
                    )
                    result = (
                        {
                            "message": "MFA requis",
                            "mfa_required": True,
                            "challenge_id": challenge_id,
                            "methods": ["totp"],
                            "ttl": MFA_CHALLENGE_TTL,
                        },
                        202,
                    )
            else:
                # Pas de MFA requis, émettre les tokens directement
                _ensure_company_operator_driver_for_user(user)
                response = _issue_tokens(user, company, device_id)
                response["mfa_required"] = False
                result = (response, 200)

        return result


@company_mobile_auth_ns.route("/mfa/verify")
class EnterpriseMobileMfaVerify(Resource):
    @company_mobile_auth_ns.expect(mfa_verify_model, validate=True)
    @limiter.limit("15/minute")
    def post(self):
        payload = request.get_json() or {}
        try:
            data = EnterpriseMfaVerifySchema().load(payload)
        except ValidationError as exc:
            return APIErrorHandler.handle_exception(exc, logger)

        # Type assertion: load() returns a dict
        assert isinstance(data, dict), "Schema load should return a dict"
        challenge = _consume_mfa_challenge(str(data["challenge_id"]))
        if not challenge:
            return APIErrorHandler.handle_validation_error(
                "Challenge MFA expiré ou invalide.",
                field="mfa_challenge",
                logger_instance=logger,
            )

        # Utiliser le repository pour récupérer l'utilisateur et l'entreprise
        from repositories.company_repository import CompanyRepository
        from repositories.user_repository import UserRepository

        user_repo = UserRepository()
        company_repo = CompanyRepository()
        user = user_repo.find_by_public_id(challenge["user_public_id"])
        if not user:
            return APIErrorHandler.handle_not_found(
                "User",
                None,
                logger,
            )
        company = company_repo.find_model_by_id(challenge["company_id"])
        if not company:
            return APIErrorHandler.handle_not_found(
                "Company",
                None,
                logger,
            )

        if not _verify_totp_code(company, data["code"]):
            return APIErrorHandler.handle_permission_error(
                "Code MFA invalide.",
                logger_instance=logger,
            )

        _ensure_company_operator_driver_for_user(user)
        response = _issue_tokens(
            user,  # pyright: ignore[reportArgumentType]
            company,
            device_id=data.get("device_id") or challenge.get("device_id"),
        )
        response["mfa_required"] = False
        return response, 200


@company_mobile_auth_ns.route("/refresh")
class EnterpriseMobileRefresh(Resource):
    @company_mobile_auth_ns.expect(refresh_model, validate=True)
    @limiter.limit("20/minute")
    def post(self):
        payload = request.get_json() or {}
        normalized_payload = dict(payload)
        if not normalized_payload.get("refresh_token"):
            normalized_payload["refresh_token"] = payload.get(
                "refreshToken"
            ) or payload.get("token")
        try:
            data = EnterpriseRefreshSchema().load(normalized_payload)
        except ValidationError as exc:
            return APIErrorHandler.handle_exception(exc, logger)

        # Type assertion: load() returns a dict
        assert isinstance(data, dict), "Schema load should return a dict"

        # ✅ SECURITE: Vérifier la signature du refresh token
        # avec JWT_SECRET_KEY (contrairement à OIDC, nos tokens sont signés)
        decoded: Dict[str, Any] | None = None

        def _refresh_reject(
            reason: str, status: int = HTTP_UNAUTHORIZED
        ) -> Tuple[Dict[str, Any], int]:
            error = (
                "refresh_rejected"
                if status == HTTP_UNAUTHORIZED
                else "access_forbidden"
            )
            return (
                {
                    "error": error,
                    "reason": reason,
                },
                status,
            )

        result: Tuple[Dict[str, Any], int] = (
            {"error": "Erreur interne."},
            HTTP_INTERNAL_ERROR,
        )

        try:
            secret_key = current_app.config.get("JWT_SECRET_KEY")
            if not secret_key:
                logger.error(
                    "[AUTH][Enterprise] JWT_SECRET_KEY manquant dans la config"
                )
                return APIErrorHandler.handle_validation_error(
                    "Erreur de configuration serveur",
                    logger_instance=logger,
                )

            # verify_aud=False : PyJWT valide l'aud par défaut et rejette les tokens
            # avec aud != attendu. On désactive pour décoder puis vérifier manuellement,
            # afin d'avoir des logs explicites (ex. "atmr-api" au lieu de "Invalid audience").
            decoded = jwt.decode(
                data["refresh_token"],
                secret_key,
                algorithms=["HS256"],
                leeway=current_app.config.get("JWT_DECODE_LEEWAY", 0),
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": False,
                },
            )

            # Vérifier manuellement l'audience (isolation driver vs enterprise)
            aud = decoded.get("aud") if decoded else None
            if aud and aud != MOBILE_AUDIENCE:
                logger.warning(
                    "[AUTH][Enterprise] Refresh token avec audience incorrecte (attendu=%s, reçu=%s)",
                    MOBILE_AUDIENCE,
                    _sanitize_log_data(str(aud)),
                )
                result = _refresh_reject("refresh_invalid", 401)

        except jwt.ExpiredSignatureError:
            logger.warning("[AUTH][Enterprise] Refresh token expiré")
            result = _refresh_reject("refresh_expired", 401)
        except jwt.PyJWTError as exc:
            logger.warning(
                "[AUTH][Enterprise] Refresh token invalide: %s",
                _sanitize_log_data(str(exc)),
            )
            result = _refresh_reject("refresh_invalid", 401)

        # Si erreur de décodage, retourner immédiatement
        if result[1] != HTTP_INTERNAL_ERROR:
            return result

        if not decoded:
            result = _refresh_reject("refresh_invalid", 401)
        else:
            public_id = decoded.get("sub")
            session_id = decoded.get("session_id")
            if not public_id:
                result = _refresh_reject("refresh_invalid", 401)
            else:
                # Utiliser le repository pour récupérer l'utilisateur
                from repositories.user_repository import UserRepository

                user_repo = UserRepository()

                user = user_repo.find_by_public_id(str(public_id))
                if not user or user.role != UserRole.COMPANY:
                    result = _refresh_reject("tenant_access_revoked", 403)
                else:
                    # Récupérer l'entreprise via le modèle User directement car user est un DTO
                    user_model = User.query.filter_by(public_id=str(public_id)).first()
                    if user_model and getattr(user_model, "active", True) is False:
                        return _refresh_reject("account_disabled", 403)
                    company = user_model.company if user_model else None
                    if not company:
                        result = _refresh_reject("tenant_access_revoked", 403)
                    else:
                        _ensure_company_operator_driver_for_user(user_model or user)
                        response = _issue_tokens(user, company, session_id=session_id)  # pyright: ignore[reportArgumentType]
                        response["mfa_required"] = False
                        result = (response, 200)

        return result


@company_mobile_auth_ns.route("/me/driver-account")
class MyDriverAccount(Resource):
    @jwt_required()
    @company_mobile_auth_ns.doc(
        description="Vérifie si l'utilisateur entreprise a aussi un compte chauffeur"
    )
    def get(self):
        """Vérifie si l'utilisateur entreprise a aussi un compte chauffeur."""
        user_public_id = get_jwt_identity()
        # Utiliser le repository pour récupérer l'utilisateur
        from repositories.user_repository import UserRepository

        user_repo = UserRepository()
        user = user_repo.find_by_public_id(user_public_id)

        if not user:
            return APIErrorHandler.handle_not_found(
                "User",
                None,
                logger,
            )

        logger.info(
            "[MyDriverAccount] Recherche compte driver pour user_id=%s, email=%s, role=%s",
            user.id,
            user.email,
            user.role,
        )
        _ensure_company_operator_driver_for_user(user)
        # 1. Vérifier si ce user a directement un compte driver (user_id)
        # Utiliser le repository pour récupérer le driver
        from repositories.company_repository import CompanyRepository
        from repositories.driver_repository import DriverRepository
        from repositories.user_repository import UserRepository

        driver_repo = DriverRepository()
        user_repo = UserRepository()
        company_repo = CompanyRepository()
        driver = driver_repo.find_model_by_user_id(user.id)
        if driver:
            logger.info(
                "[MyDriverAccount] Driver trouvé directement par user_id: driver_id=%s",
                driver.id,
            )
        else:
            logger.debug(
                "[MyDriverAccount] Aucun driver trouvé directement par user_id"
            )

        # 2. Si pas trouvé, chercher par email (même email = même personne)
        if not driver and user.email:
            driver_user = user_repo.find_by_email_and_role(user.email, UserRole.DRIVER)
            if driver_user:
                driver = driver_repo.find_model_by_user_id(driver_user.id)
                if driver:
                    logger.info(
                        "[MyDriverAccount] Driver trouvé par email: driver_id=%s, driver_user_id=%s",
                        driver.id,
                        driver_user.id,
                    )

        # 3. Si toujours pas trouvé, chercher par company_id (chauffeur d'urgence de la même entreprise)
        if not driver:
            company = company_repo.find_model_by_user_id(user.id)
            if company:
                logger.info(
                    "[MyDriverAccount] Company trouvée: company_id=%s, cherchant drivers d'urgence",
                    company.id,
                )

                # Chercher un driver d'urgence de la même entreprise avec le même email
                if user.email:
                    driver_user = user_repo.find_by_email_with_driver_join(
                        user.email, company.id, DriverType.EMERGENCY
                    )
                    if driver_user:
                        driver = driver_repo.find_model_by_user_id(driver_user.id)
                        if driver:
                            logger.info(
                                "[MyDriverAccount] Driver d'urgence trouvé par email: driver_id=%s",
                                driver.id,
                            )

                # Si toujours pas trouvé, chercher n'importe quel driver d'urgence de la même entreprise
                # (au cas où l'email serait différent)
                if not driver:
                    emergency_driver = driver_repo.find_model_by_company_and_type(
                        company.id, DriverType.EMERGENCY
                    )
                    if emergency_driver:
                        logger.info(
                            "[MyDriverAccount] Driver d'urgence trouvé dans l'entreprise: driver_id=%s, user_id=%s",
                            emergency_driver.id,
                            emergency_driver.user_id,
                        )
                        # Vérifier si c'est le même utilisateur (par email ou username)
                        emergency_user = user_repo.find_by_id(
                            cast(int, emergency_driver.user_id)
                        )
                        if emergency_user:
                            logger.info(
                                "[MyDriverAccount] User du driver: email=%s, username=%s",
                                emergency_user.email,
                                emergency_user.username,
                            )
                            # Si email ou username correspond, ou si c'est le seul driver d'urgence de l'entreprise
                            # on considère qu'il peut switcher
                            if (user.email and emergency_user.email == user.email) or (
                                user.username
                                and emergency_user.username == user.username
                            ):
                                driver = emergency_driver
                                logger.info(
                                    "[MyDriverAccount] Driver d'urgence associé par email/username"
                                )
                            else:
                                # Si c'est le seul driver d'urgence de l'entreprise, on l'associe quand même
                                # (cas où l'entreprise a un seul chauffeur d'urgence)
                                emergency_count = driver_repo.count_by_company_and_type(
                                    company.id, DriverType.EMERGENCY
                                )
                                if emergency_count == 1:
                                    driver = emergency_driver
                                    logger.info(
                                        "[MyDriverAccount] Driver d'urgence unique de l'entreprise associé"
                                    )
            else:
                logger.warning(
                    "[MyDriverAccount] Aucune company trouvée pour user_id=%s", user.id
                )

        if driver:
            driver_type = (
                driver.driver_type.value
                if hasattr(driver.driver_type, "value")
                else str(driver.driver_type)
            )
            logger.info(
                "[MyDriverAccount] Retourne has_driver_account=True: driver_id=%s, type=%s",
                driver.id,
                driver_type,
            )
            return {
                "has_driver_account": True,
                "driver_id": driver.id,
                "driver_type": driver_type,
                "is_active": driver.is_active,
                "is_available": driver.is_available,
            }

        logger.info(
            "[MyDriverAccount] Aucun driver trouvé, retourne has_driver_account=False"
        )
        return {"has_driver_account": False}


@company_mobile_auth_ns.route("/me/switch-to-driver")
class SwitchToDriver(Resource):
    @jwt_required()
    @company_mobile_auth_ns.doc(
        description="Génère un token driver à partir du token entreprise (si l'utilisateur a aussi un compte driver)"
    )
    def post(self):
        """Génère un token driver pour permettre le switch automatique."""
        logger.info("[SwitchToDriver] Endpoint appelé")
        from flask import current_app
        from flask_jwt_extended import (
            create_access_token,
            create_refresh_token,
        )

        from routes.auth import store_refresh_token

        user_public_id = get_jwt_identity()
        logger.info("[SwitchToDriver] user_public_id=%s", user_public_id)
        # ✅ DDD: Utilise use-case au lieu de service directement
        user = get_current_user_via_use_case()
        if not user:
            return APIErrorHandler.handle_not_found(
                "User",
                None,
                logger,
            )

        _ensure_company_operator_driver_for_user(user)
        # Utiliser la même logique que MyDriverAccount pour trouver le driver
        # 1. Vérifier si ce user a directement un compte driver (user_id)
        # Utiliser le repository pour récupérer le driver
        from repositories.company_repository import CompanyRepository
        from repositories.driver_repository import DriverRepository
        from repositories.user_repository import UserRepository

        driver_repo = DriverRepository()
        user_repo = UserRepository()
        company_repo = CompanyRepository()
        driver = driver_repo.find_model_by_user_id(user.id)
        # 2. Si pas trouvé, chercher par email (même email = même personne)
        if not driver and user.email:
            driver_user = user_repo.find_by_email_and_role(user.email, UserRole.DRIVER)
            if driver_user:
                driver = driver_repo.find_model_by_user_id(driver_user.id)

        # 3. Si toujours pas trouvé, chercher par company_id (chauffeur d'urgence de la même entreprise)
        if not driver:
            company = company_repo.find_model_by_user_id(user.id)
            if company:
                # Chercher un driver d'urgence de la même entreprise avec le même email
                if user.email:
                    driver_user = user_repo.find_by_email_with_driver_join(
                        user.email, company.id, DriverType.EMERGENCY
                    )
                    if driver_user:
                        driver = driver_repo.find_model_by_user_id(driver_user.id)

                # Si toujours pas trouvé, chercher n'importe quel driver d'urgence de la même entreprise
                if not driver:
                    emergency_driver = driver_repo.find_model_by_company_and_type(
                        company.id, DriverType.EMERGENCY
                    )
                    if emergency_driver:
                        # Vérifier si c'est le même utilisateur (par email ou username)
                        emergency_user = user_repo.find_by_id(
                            cast(int, emergency_driver.user_id)
                        )
                        if emergency_user:
                            # Si email ou username correspond, ou si c'est le seul driver d'urgence de l'entreprise
                            if (user.email and emergency_user.email == user.email) or (
                                user.username
                                and emergency_user.username == user.username
                            ):
                                driver = emergency_driver
                            else:
                                # Si c'est le seul driver d'urgence de l'entreprise, on l'associe quand même
                                emergency_count = driver_repo.count_by_company_and_type(
                                    company.id, DriverType.EMERGENCY
                                )
                                if emergency_count == 1:
                                    driver = emergency_driver
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Driver",
                None,
                logger,
            )

        # Vérifier que le driver est actif
        if not bool(driver.is_active):
            return APIErrorHandler.handle_permission_error(
                "Le compte chauffeur n'est pas actif",
                logger_instance=logger,
            )

        # Récupérer le User associé au driver (peut être différent de l'user entreprise)
        driver_user = user_repo.find_by_id(cast(int, driver.user_id))
        if not driver_user:
            return APIErrorHandler.handle_not_found(
                "User",
                None,
                logger,
            )

        logger.info(
            "[SwitchToDriver] Driver trouvé: driver_id=%s, driver_user_id=%s, driver_user_role=%s, driver_user_public_id=%s",
            driver.id,
            driver_user.id,
            driver_user.role,
            driver_user.public_id,
        )

        # Créer un token driver avec l'audience "atmr-api" (pour l'API driver)
        # Utiliser le public_id du driver_user (pas celui de l'user entreprise)
        claims = {
            "role": "driver",  # Forcer le rôle driver
            "company_id": driver.company_id,
            "driver_id": driver.id,
            "aud": "atmr-api",  # Audience pour l'API driver
        }

        access_token = create_access_token(
            identity=str(driver_user.public_id),  # Utiliser le public_id du driver_user
            additional_claims=claims,
            expires_delta=current_app.config.get(
                "JWT_ACCESS_TOKEN_EXPIRES", timedelta(hours=1)
            ),
        )

        # Créer un refresh token
        refresh_token = create_refresh_token(
            identity=str(driver_user.public_id),  # Utiliser le public_id du driver_user
            additional_claims={
                "aud": "atmr-api",
                "role": "driver",
            },
            # ✅ PHASE 4 : Augmentation de la durée du refresh token à 90 jours
            expires_delta=current_app.config.get(
                "JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=90)
            ),
        )

        # Stocker le refresh token dans la DB (comme pour le login normal)
        try:
            # ✅ PHASE 4 : Augmentation de la durée du refresh token à 90 jours
            refresh_expires_delta = current_app.config.get(
                "JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=90)
            )
            refresh_expires_at = datetime.now(UTC) + refresh_expires_delta
            store_refresh_token(
                token=refresh_token,
                user_id=driver_user.id,  # Utiliser le user_id du driver_user
                expires_at=refresh_expires_at,
                device_id=request.headers.get("X-Device-ID"),
                device_name=request.headers.get("X-Device-Name"),
            )
        except Exception as store_error:
            logger.warning("Échec stockage refresh token driver: %s", store_error)

        return (
            {
                "token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "public_id": driver_user.public_id,  # Utiliser le public_id du driver_user
                    "email": driver_user.email,
                    "first_name": driver_user.first_name,
                    "last_name": driver_user.last_name,
                    "role": "driver",
                },
                "driver": {
                    "id": driver.id,
                    "driver_type": (
                        driver.driver_type.value
                        if hasattr(driver.driver_type, "value")
                        else str(driver.driver_type)
                    ),
                },
                "mfa_required": False,
            },
            200,
        )


@company_mobile_auth_ns.route("/session")
class EnterpriseMobileSession(Resource):
    @jwt_required()
    def get(self):
        try:
            claims = get_jwt()
            # ✅ Validation manuelle de l'audience pour les tokens entreprise mobile
            aud = claims.get("aud")
            if aud and aud != MOBILE_AUDIENCE:
                logger.warning(
                    "[AUTH][Enterprise] Token avec audience incorrecte: %s (attendu: %s)",
                    aud,
                    MOBILE_AUDIENCE,
                )
                return APIErrorHandler.handle_permission_error(
                    "Token invalide (audience incorrecte).",
                    logger_instance=logger,
                )

            # ✅ DDD: Utilise use-case au lieu de service directement
            try:
                user = get_current_user_via_use_case()
            except Exception:
                raise
            if not user or user.role != UserRole.COMPANY:
                return APIErrorHandler.handle_permission_error(
                    "Accès refusé.",
                    logger_instance=logger,
                )

            _ensure_company_operator_driver_for_user(user)
            # Récupérer l'entreprise
            # ✅ Fix: UserDTO n'a pas d'attribut company, utiliser CompanyRepository
            from repositories.company_repository import CompanyRepository

            company_repo = CompanyRepository()
            # Si user est un UserDTO, utiliser find_model_by_user_id
            # Si user est un User, essayer d'abord user.company puis fallback
            if hasattr(user, "company"):
                # user est un modèle User SQLAlchemy
                company = user.company
            else:
                # user est un UserDTO, récupérer via repository
                company = company_repo.find_model_by_user_id(user.id)
            # Valider que l'entreprise existe
            if not company:
                return APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )

            # Succès : construire la réponse
            response_data = {
                "user": {
                    "id": user.id,
                    "public_id": user.public_id,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "role": user.role.value,
                },
                "company": {
                    "id": company.id,
                    "name": company.name,
                    "dispatch_mode": company.dispatch_mode.value,
                },
                "scopes": claims.get("scopes", []),
                "session_id": claims.get("session_id"),
                "aud": claims.get("aud"),
            }
            return response_data, 200

        except (
            jwt.exceptions.ExpiredSignatureError,
            jwt.exceptions.InvalidAudienceError,
        ) as jwt_error:
            error_msg = (
                "Token expiré. Veuillez vous reconnecter."
                if isinstance(jwt_error, jwt.exceptions.ExpiredSignatureError)
                else "Token invalide (audience incorrecte)."
            )
            log_msg = (
                "Token expiré pour /auth/session"
                if isinstance(jwt_error, jwt.exceptions.ExpiredSignatureError)
                else "Token avec audience invalide pour /auth/session"
            )
            logger.warning(log_msg)
            return APIErrorHandler.handle_permission_error(
                error_msg,
                logger_instance=logger,
            )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception(
                "❌ ERREUR /auth/session: %s - %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
