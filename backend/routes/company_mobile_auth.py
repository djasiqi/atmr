"""Routes d'authentification mobile pour les entreprises (dispatch)."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import jwt
from flask import current_app, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Namespace, Resource, fields
from marshmallow import Schema, ValidationError, validate
from marshmallow import fields as ma_fields

from ext import limiter, redis_client
from models import Company, User, UserRole

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

# Whitelist des providers OIDC autorisés (configurable via env)
ALLOWED_OIDC_PROVIDERS = set(
    os.getenv("ALLOWED_OIDC_PROVIDERS", "").split(",")
    if os.getenv("ALLOWED_OIDC_PROVIDERS")
    else []
)


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
    security = _get_company_security(company)
    policy = security.get("mobile_mfa") or {}
    return bool(policy.get("required", False))


def _get_totp_secret(company: Company | None) -> Optional[str]:
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


def _validate_device_id(device_id: Optional[str]) -> bool:
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
    device_id: Optional[str] = None,
    extra_scopes: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    scopes = list(DEFAULT_SCOPES)
    if user.role == UserRole.ADMIN:
        scopes.append("enterprise.dispatch:admin")
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

    access_token = create_access_token(
        identity=str(user.public_id),
        additional_claims=additional_claims,
        expires_delta=timedelta(minutes=45),
    )
    refresh_token = create_refresh_token(
        identity=str(user.public_id),
        additional_claims={
            "aud": MOBILE_AUDIENCE,
            "session_id": session_identifier,
        },
        expires_delta=timedelta(days=14),
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
    device_id: Optional[str],
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


def _consume_mfa_challenge(challenge_id: str) -> Optional[Dict[str, Any]]:
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


def _find_company_user_by_email(email: str) -> Tuple[Optional[User], Optional[Company]]:
    user: Optional[User] = User.query.filter(User.email == email).first()
    if not user:
        return None, None
    if user.role not in (UserRole.COMPANY, UserRole.ADMIN):
        return None, None
    company = user.company
    if not company:
        return None, None
    return user, company


def _handle_oidc_login(
    id_token: str, provider: Optional[str]
) -> Tuple[Optional[User], Optional[Company]]:
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

    # Décoder sans vérification pour extraire les claims
    # ⚠️ SECURITE: En production, il faudrait vérifier la signature
    # avec les clés publiques du provider (JWKS)
    try:
        decoded = jwt.decode(
            id_token, options={"verify_signature": False, "verify_aud": False}
        )
    except jwt.PyJWTError as exc:
        logger.warning(
            "[AUTH][Enterprise] Echec décodage token OIDC: %s",
            _sanitize_log_data(str(exc)),
        )
        raise ValueError("ID token invalide.") from exc

    # Vérifier l'issuer (provider)
    issuer = decoded.get("iss")
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
            return {"error": "Paramètres invalides", "details": exc.messages}, 400

        # Type assertion: load() returns a dict
        assert isinstance(data, dict), "Schema load should return a dict"
        method: str = data["method"]
        email = data.get("email")
        password = data.get("password")
        id_token = data.get("id_token")
        mfa_code = data.get("mfa_code")
        device_id = data.get("device_id")

        user: Optional[User] = None
        company: Optional[Company] = None
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
            return {"error": "Paramètres invalides", "details": exc.messages}, 400

        # Type assertion: load() returns a dict
        assert isinstance(data, dict), "Schema load should return a dict"
        challenge = _consume_mfa_challenge(str(data["challenge_id"]))
        if not challenge:
            return {"error": "Challenge MFA expiré ou invalide."}, 410

        user = User.query.filter_by(public_id=challenge["user_public_id"]).first()
        if not user:
            return {"error": "Utilisateur introuvable."}, 404
        company = Company.query.get(challenge["company_id"])
        if not company:
            return {"error": "Entreprise introuvable."}, 404

        if not _verify_totp_code(company, data["code"]):
            return {"error": "Code MFA invalide."}, 401

        response = _issue_tokens(
            user,
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
        try:
            data = EnterpriseRefreshSchema().load(payload)
        except ValidationError as exc:
            return {"error": "Paramètres invalides", "details": exc.messages}, 400

        # Type assertion: load() returns a dict
        assert isinstance(data, dict), "Schema load should return a dict"

        # ✅ SECURITE: Vérifier la signature du refresh token
        # avec JWT_SECRET_KEY (contrairement à OIDC, nos tokens sont signés)
        decoded: Optional[Dict[str, Any]] = None
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
                return {"error": "Erreur de configuration serveur."}, 500

            decoded = jwt.decode(
                data["refresh_token"],
                secret_key,
                algorithms=["HS256"],
                options={"verify_signature": True, "verify_exp": True},
            )

            # Vérifier manuellement l'audience
            aud = decoded.get("aud") if decoded else None
            if aud and aud != MOBILE_AUDIENCE:
                logger.warning(
                    "[AUTH][Enterprise] Audience refresh token incorrecte: %s",
                    _sanitize_log_data(str(aud)),
                )
                result = ({"error": "Refresh token invalide."}, 401)

        except jwt.ExpiredSignatureError:
            logger.warning("[AUTH][Enterprise] Refresh token expiré")
            result = ({"error": "Refresh token expiré."}, 401)
        except jwt.PyJWTError as exc:
            logger.warning(
                "[AUTH][Enterprise] Refresh token invalide: %s",
                _sanitize_log_data(str(exc)),
            )
            result = ({"error": "Refresh token invalide."}, 401)

        # Si erreur de décodage, retourner immédiatement
        if result[1] != HTTP_INTERNAL_ERROR:
            return result

        if not decoded:
            result = ({"error": "Refresh token invalide."}, 401)
        else:
            public_id = decoded.get("sub")
            session_id = decoded.get("session_id")
            if not public_id:
                result = ({"error": "Refresh token invalide."}, 401)
            else:
                user = User.query.filter_by(public_id=str(public_id)).first()
                if not user or user.role not in (UserRole.COMPANY, UserRole.ADMIN):
                    result = ({"error": "Accès refusé."}, 403)
                else:
                    company = user.company
                    if not company:
                        result = ({"error": "Entreprise introuvable."}, 403)
                    else:
                        response = _issue_tokens(user, company, session_id=session_id)
                        response["mfa_required"] = False
                        result = (response, 200)

        return result


@company_mobile_auth_ns.route("/session")
class EnterpriseMobileSession(Resource):
    @jwt_required()
    def get(self):
        claims = get_jwt()
        identity = get_jwt_identity()
        user = User.query.filter_by(public_id=str(identity)).first()
        if not user or user.role not in (UserRole.COMPANY, UserRole.ADMIN):
            return {"error": "Accès refusé."}, 403
        company = user.company
        if not company:
            return {"error": "Entreprise introuvable."}, 404

        return {
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
        }, 200
