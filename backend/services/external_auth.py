"""Connexion Google et Apple.

Le fournisseur authentifie la personne. LIRIE décide seule des portails.
Une adresse e-mail identique ne rattache pas un compte existant.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import jwt
from flask import current_app
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from sqlalchemy import func, text
from sqlalchemy.exc import IntegrityError

from ext import db
from models import Client, PortalGrant, User, UserRole
from models.activation_session import ActivationSession
from models.enums import ClientType
from models.external_auth_transaction import ExternalAuthTransaction
from models.user_external_identity import UserExternalIdentity

logger = logging.getLogger(__name__)

PROVIDERS = ("google", "apple")
TRANSACTION_TTL = timedelta(minutes=10)
GOOGLE_ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})
APPLE_ISSUER = "https://appleid.apple.com"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"
APPLE_AUTH = "https://appleid.apple.com/auth/authorize"
APPLE_TOKEN = "https://appleid.apple.com/auth/token"
APPLE_JWKS = "https://appleid.apple.com/auth/keys"
_JWKS_CACHE: dict[str, tuple[datetime, dict[str, Any]]] = {}


class ExternalAuthRejected(Exception):
    """Refus explicite, sans session et sans habilitation."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.intent: str | None = None
        self.public_id: str | None = None
        self.role: str | None = None


def provider_status() -> dict[str, Any]:
    """Un bouton n'est utilisable que si la configuration est complète."""
    return {
        "google": {"enabled": _configured("google")},
        "apple": {
            "enabled": _configured("apple"),
            "native": bool(_setting("EXTERNAL_AUTH_APPLE_BUNDLE_ID")),
        },
    }


def begin(
    provider: str,
    *,
    intent: str,
    user_id: int | None,
) -> dict[str, str]:
    """Ouvre une redirection Authorization Code + PKCE."""
    name = _provider(provider)
    if not _configured(name):
        raise ExternalAuthRejected(
            "fournisseur_indisponible",
            "Ce fournisseur n'est pas configuré.",
            status=503,
        )
    if intent not in {"login", "link"}:
        raise ExternalAuthRejected("intention_refusee", "Intention inconnue.")
    if intent == "link" and user_id is None:
        raise ExternalAuthRejected(
            "authentification_requise",
            "Connectez-vous avant d'ajouter cette méthode.",
            status=401,
        )
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    binding = secrets.token_urlsafe(32)
    row = ExternalAuthTransaction(
        state_hash=_digest(state),
        provider=name,
        nonce_hash=_digest(nonce),
        code_verifier=verifier,
        intent=intent,
        user_id=user_id,
        status="pending",
        # Empreinte du cookie de navigateur. Remplacée plus tard par le
        # ticket de consentement, seulement après cette vérification.
        ticket_hash=_digest(binding),
        expires_at=datetime.now(UTC) + TRANSACTION_TTL,
    )
    db.session.add(row)
    _persist()
    return {
        "authorization_url": _authorize_url(name, state, nonce, verifier),
        "state": state,
        "browser_binding": binding,
    }


def consume_callback(
    code: str,
    state: str,
    *,
    apple_names: dict[str, str | None] | None = None,
    browser_binding: str | None = None,
) -> dict[str, Any]:
    """Échange le code, vérifie le jeton, puis ouvre la session."""
    row = _lock_pending(state)
    if not _browser_binding_ok(row, browser_binding):
        raise ExternalAuthRejected(
            "origine_invalide",
            "Cette tentative n'appartient pas à ce navigateur.",
            status=403,
        )
    intent = row.intent
    actor_id = row.user_id
    try:
        return _consume_verified(row, code, apple_names=apple_names)
    except ExternalAuthRejected as error:
        _remember_actor(error, intent, actor_id)
        raise


def _consume_verified(
    row: ExternalAuthTransaction,
    code: str,
    *,
    apple_names: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Le profil transmis par le navigateur LIRIE n'est jamais lu ici.

    Apple joint le nom, une seule fois, au retour ``form_post``. Ce bloc
    n'est pas signé : il ne sert qu'à remplir un nom vide. L'adresse vient
    uniquement du jeton vérifié.
    """
    name = row.provider
    verifier = row.code_verifier or ""
    row.code_verifier = None
    row.status = "used"
    row.consumed_at = datetime.now(UTC)
    _persist()
    try:
        token = exchange_code(name, code, verifier)
        claims = verify_id_token(
            token,
            provider=name,
            audience=_client_id(name),
            nonce_hash=row.nonce_hash,
            jwks=fetch_jwks(name),
        )
    except ExternalAuthRejected:
        _fail(row, "jeton")
        raise
    if name == "apple":
        _apply_apple_names(claims, apple_names)
    subject = str(claims["sub"])
    email, verified = _email_claims(claims)
    _advisory_lock(f"ext-sub:{name}:{subject}")
    if email:
        _advisory_lock(f"ext-email:{email}")
    identity = _identity_for(name, subject)
    if row.intent == "link":
        user = _link_existing(row, identity, claims, email, verified)
        return {"kind": "session", "user": user, "intent": row.intent}
    if identity is not None:
        account = db.session.get(User, int(identity.user_id))
        if account is None:
            raise ExternalAuthRejected(
                "identite_incomplete",
                "Le compte lié à cette identité est introuvable.",
            )
        _assert_account_open(account)
        _note_email(identity, email, verified)
        _touch_names(account, claims, only_if_empty=True)
        _persist()
        return {"kind": "session", "user": account, "intent": row.intent}
    if email and verified and _user_by_email(email) is not None:
        raise ExternalAuthRejected(
            "compte_existant",
            "Un compte existe déjà pour cette adresse. "
            "Connectez-vous avec votre mot de passe, puis ajoutez cette méthode.",
            status=409,
        )
    if not email or not verified:
        raise ExternalAuthRejected(
            "identite_incomplete",
            "Le fournisseur n'a pas confirmé d'adresse e-mail "
            "pour une première connexion.",
        )
    ticket = secrets.token_urlsafe(32)
    row.status = "ready"
    row.ticket_hash = _digest(ticket)
    row.provider_sub = subject
    row.email = email
    row.email_verified = True
    row.first_name = _clip(claims.get("given_name"), 100)
    row.last_name = _clip(claims.get("family_name"), 100)
    row.expires_at = datetime.now(UTC) + TRANSACTION_TTL
    _persist()
    return {"kind": "consent", "ticket": ticket}


def register_from_ticket(ticket: str, *, accept_terms: bool) -> User:
    """Crée un client seulement après un consentement explicite."""
    if not accept_terms:
        raise ExternalAuthRejected(
            "consentement_requis",
            "La création du compte client exige d'accepter les conditions.",
        )
    row = (
        ExternalAuthTransaction.query.filter_by(
            ticket_hash=_digest(ticket), status="ready"
        )
        .with_for_update()
        .one_or_none()
    )
    if row is None or _expired(row):
        raise ExternalAuthRejected(
            "inscription_expiree",
            "Cette création de compte n'est plus valable. Recommencez.",
        )
    if row.provider_sub is None or not row.email:
        raise ExternalAuthRejected("identite_incomplete", "Identité incomplète.")
    _advisory_lock(f"ext-sub:{row.provider}:{row.provider_sub}")
    _advisory_lock(f"ext-email:{row.email}")
    existing = _identity_for(row.provider, row.provider_sub)
    if existing is not None:
        return _resume_linked_account(row, existing)
    if _user_by_email(row.email) is not None:
        raise ExternalAuthRejected(
            "compte_existant",
            "Un compte existe déjà pour cette adresse. "
            "Connectez-vous avec votre mot de passe, puis ajoutez cette méthode.",
            status=409,
        )
    nested = db.session.begin_nested()
    try:
        user = _create_client(row)
        nested.commit()
    except IntegrityError as error:
        nested.rollback()
        raced = _identity_for(row.provider, row.provider_sub)
        if raced is not None:
            return _resume_linked_account(row, raced)
        raise ExternalAuthRejected(
            "compte_existant",
            "Un compte existe déjà pour cette adresse. "
            "Connectez-vous avec votre mot de passe, puis ajoutez cette méthode.",
            status=409,
        ) from error
    except Exception:
        nested.rollback()
        raise
    row.status = "used"
    row.ticket_hash = None
    _persist()
    return user


def list_identities(user: User) -> dict[str, Any]:
    """État des méthodes liées. Le `sub` fournisseur n'est pas renvoyé."""
    linked = {
        row.provider
        for row in UserExternalIdentity.query.filter_by(user_id=int(user.id)).all()
    }
    password_login = str(getattr(user, "authentication_method", "") or "") != "sso"
    return {
        "google": {"linked": "google" in linked},
        "apple": {"linked": "apple" in linked},
        "password_login": password_login,
    }


def unlink(user: User, provider: str) -> None:
    """Retire une méthode seulement s'il reste un autre moyen de récupération."""
    name = _provider(provider)
    row = UserExternalIdentity.query.filter_by(
        user_id=int(user.id), provider=name
    ).one_or_none()
    if row is None:
        raise ExternalAuthRejected(
            "methode_absente", "Cette méthode n'est pas liée.", status=404
        )
    others = (
        UserExternalIdentity.query.filter_by(user_id=int(user.id))
        .filter(UserExternalIdentity.provider != name)
        .count()
    )
    password_login = str(getattr(user, "authentication_method", "") or "") != "sso"
    if others < 1 and not password_login:
        raise ExternalAuthRejected(
            "derniere_methode",
            "Conservez au moins une méthode de connexion.",
            status=409,
        )
    db.session.delete(row)
    _persist()


def exchange_code(provider: str, code: str, verifier: str) -> str:
    """Échange le code auprès du fournisseur. Aucun secret n'est journalisé."""
    secret = code.strip()
    if not secret or not verifier:
        raise ExternalAuthRejected("code_refuse", "Le code d'autorisation est absent.")
    body = {
        "grant_type": "authorization_code",
        "code": secret,
        "redirect_uri": _redirect_uri(provider),
        "client_id": _client_id(provider),
        "code_verifier": verifier,
    }
    if provider == "google":
        body["client_secret"] = _required(
            f"EXTERNAL_AUTH_{provider.upper()}_CLIENT_SECRET"
        )
        url = GOOGLE_TOKEN
    else:
        body["client_secret"] = _apple_client_secret()
        url = APPLE_TOKEN
    payload = _post_form(url, body)
    token = str(payload.get("id_token") or "")
    if not token:
        raise ExternalAuthRejected(
            "jeton_absent",
            "Le fournisseur n'a pas renvoyé de jeton d'identité.",
        )
    return token


def fetch_jwks(provider: str) -> dict[str, Any]:
    url = GOOGLE_JWKS if provider == "google" else APPLE_JWKS
    cached = _JWKS_CACHE.get(url)
    now = datetime.now(UTC)
    if cached is not None and cached[0] > now:
        return cached[1]
    try:
        # JWKS URL constante fournisseur (GOOGLE_JWKS/APPLE_JWKS), hors contrôle utilisateur.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urlopen(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        logger.warning("jwks_indisponible provider=%s", provider)
        raise ExternalAuthRejected(
            "fournisseur_indisponible",
            "Le fournisseur est indisponible.",
            status=503,
        ) from error
    _JWKS_CACHE[url] = (now + timedelta(hours=1), data)
    return data


def verify_id_token(
    token: str,
    *,
    provider: str,
    audience: str,
    nonce_hash: str,
    jwks: dict[str, Any],
) -> dict[str, Any]:
    """Vérifie signature, issuer, audience, expiration et nonce."""
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as error:
        raise ExternalAuthRejected(
            "jeton_invalide", "Jeton d'identité illisible."
        ) from error
    algorithm = str(header.get("alg") or "")
    if algorithm not in {"RS256", "ES256"}:
        raise ExternalAuthRejected("jeton_invalide", "Algorithme de signature refusé.")
    key = _jwk(jwks, str(header.get("kid") or ""))
    issuer = GOOGLE_ISSUERS if provider == "google" else APPLE_ISSUER
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[algorithm],
            audience=audience,
            issuer=issuer,
            leeway=60,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.PyJWTError as error:
        raise ExternalAuthRejected(
            "jeton_invalide",
            "Le jeton d'identité n'est pas accepté.",
        ) from error
    nonce = str(claims.get("nonce") or "")
    if not nonce or _digest(nonce) != nonce_hash:
        raise ExternalAuthRejected(
            "nonce_invalide", "Cette tentative n'est plus valable."
        )
    if not str(claims.get("sub") or "").strip():
        raise ExternalAuthRejected("jeton_invalide", "Identifiant fournisseur absent.")
    authorized_party = str(claims.get("azp") or "")
    if provider == "google" and authorized_party and authorized_party != audience:
        raise ExternalAuthRejected(
            "jeton_invalide", "Le jeton d'identité n'est pas accepté."
        )
    return claims


def _link_existing(
    row: ExternalAuthTransaction,
    identity: UserExternalIdentity | None,
    claims: dict[str, Any],
    email: str | None,
    verified: bool,
) -> User:
    owner = db.session.get(User, int(row.user_id or 0))
    if owner is None:
        raise ExternalAuthRejected(
            "authentification_requise",
            "La session qui ajoute la méthode n'est plus valable.",
            status=401,
        )
    _assert_account_open(owner)
    subject = str(claims["sub"])
    if email and verified:
        other = _user_by_email(email)
        if other is not None and int(other.id) != int(owner.id):
            raise ExternalAuthRejected(
                "adresse_autre_compte",
                "Cette adresse appartient à un autre compte. "
                "Connectez-vous à ce compte pour ajouter cette méthode.",
                status=409,
            )
    if identity is not None and int(identity.user_id) != int(owner.id):
        raise ExternalAuthRejected(
            "methode_deja_liee",
            "Cette méthode est déjà liée à un autre compte.",
            status=409,
        )
    if identity is None:
        taken = UserExternalIdentity.query.filter_by(
            user_id=int(owner.id), provider=row.provider
        ).one_or_none()
        if taken is not None:
            raise ExternalAuthRejected(
                "methode_deja_liee",
                "Cette méthode est déjà liée à votre compte.",
                status=409,
            )
        identity = UserExternalIdentity(
            user_id=int(owner.id),
            provider=row.provider,
            provider_sub=subject,
            email_at_link=email if verified else None,
        )
        db.session.add(identity)
    else:
        _note_email(identity, email, verified)
    _persist()
    return owner


def _create_client(row: ExternalAuthTransaction) -> User:
    """Compte client. Aucune habilitation professionnelle n'est écrite."""
    user = User()
    user.username = f"ext_{uuid.uuid4().hex[:12]}"
    user.first_name = row.first_name
    user.last_name = row.last_name
    user.email = row.email
    user.role = UserRole.CLIENT
    user.public_id = str(uuid.uuid4())
    user.account_status = "active"
    user.authentication_method = "sso"
    user.force_password_change = False
    user.set_password(secrets.token_urlsafe(48), force_change=False)
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.is_active = True
    client.contact_email = row.email
    client.client_type = ClientType.PORTAL
    db.session.add(client)
    activation = ActivationSession()
    activation.activation_session_id = str(uuid.uuid4())
    activation.user_id = user.id
    activation.email_verified_at = datetime.now(UTC)
    activation.portal_terms_required = True
    activation.sms_attempts = 0
    activation.resend_count_email = 0
    activation.resend_count_sms = 0
    db.session.add(activation)
    db.session.flush()
    from services.legal.record_terms_acceptance import record_portal_terms_acceptance

    record_portal_terms_acceptance(user, client)
    activation.portal_terms_required = False
    activation.consumed_at = datetime.now(UTC)
    db.session.add(
        UserExternalIdentity(
            user_id=int(user.id),
            provider=row.provider,
            provider_sub=str(row.provider_sub),
            email_at_link=row.email,
        )
    )
    db.session.flush()
    if PortalGrant.query.filter_by(user_id=int(user.id)).count() != 0:
        raise ExternalAuthRejected(
            "habilitation_interdite",
            "La connexion externe ne crée pas d'habilitation.",
            status=409,
        )
    return user


def _identity_for(provider: str, subject: str) -> UserExternalIdentity | None:
    return UserExternalIdentity.query.filter_by(
        provider=provider, provider_sub=subject
    ).one_or_none()


def _user_by_email(email: str) -> User | None:
    return User.query.filter(func.lower(User.email) == email).one_or_none()


def _note_email(
    identity: UserExternalIdentity, email: str | None, verified: bool
) -> None:
    """Mémorise l'adresse vue chez le fournisseur sans modifier le compte.

    Une adresse déjà portée par un autre utilisateur n'est pas recopiée.
    """
    if not email or not verified:
        return
    other = _user_by_email(email)
    if other is not None and int(other.id) != int(identity.user_id):
        return
    identity.email_at_link = email


def _assert_account_open(user: User) -> None:
    """Une identité fournisseur ne rouvre pas un compte fermé."""
    if getattr(user, "archived_at", None) or getattr(user, "disabled_at", None):
        raise ExternalAuthRejected(
            "compte_indisponible",
            "Ce compte n'est pas disponible.",
            status=403,
        )
    status = str(getattr(user, "account_status", "") or "")
    if status in {"disabled", "pending_activation", "invited"}:
        raise ExternalAuthRejected(
            "compte_indisponible",
            "Ce compte n'est pas disponible.",
            status=403,
        )
    if (
        user.role == UserRole.CLIENT
        and list(user.clients or [])
        and not any(getattr(item, "is_active", False) for item in user.clients)
    ):
        raise ExternalAuthRejected(
            "compte_indisponible",
            "Ce compte n'est pas disponible.",
            status=403,
        )


def _resume_linked_account(
    row: ExternalAuthTransaction, identity: UserExternalIdentity
) -> User:
    """Même `sub` : reprendre le compte déjà lié, sans en créer un second."""
    account = db.session.get(User, int(identity.user_id))
    if account is None:
        raise ExternalAuthRejected(
            "identite_incomplete",
            "Le compte lié à cette identité est introuvable.",
        )
    _assert_account_open(account)
    row.status = "used"
    row.ticket_hash = None
    _persist()
    return account


def _remember_actor(
    error: ExternalAuthRejected, intent: str | None, actor_id: int | None
) -> None:
    error.intent = error.intent or intent
    if not actor_id or error.public_id:
        return
    owner = db.session.get(User, int(actor_id))
    if owner is None:
        return
    error.public_id = str(owner.public_id)
    error.role = owner.role.value


def _advisory_lock(name: str) -> None:
    """Sérialise deux créations concurrentes du même sub ou du même e-mail."""
    if db.engine.dialect.name != "postgresql":
        return
    key = int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "big", signed=True
    )
    db.session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


def apple_names_from_form(raw: str | None) -> dict[str, str | None] | None:
    """Extrait le nom Apple. L'adresse du même bloc est ignorée."""
    text = str(raw or "").strip()
    if not text or len(text) > 2048:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, dict):
        return None
    given = _clip(name.get("firstName"), 100)
    family = _clip(name.get("lastName"), 100)
    if not given and not family:
        return None
    return {"given_name": given, "family_name": family}


def _apply_apple_names(
    claims: dict[str, Any], names: dict[str, str | None] | None
) -> None:
    if not names:
        return
    if names.get("given_name") and not claims.get("given_name"):
        claims["given_name"] = names["given_name"]
    if names.get("family_name") and not claims.get("family_name"):
        claims["family_name"] = names["family_name"]


def _touch_names(user: User, claims: dict[str, Any], *, only_if_empty: bool) -> None:
    if only_if_empty and (user.first_name or user.last_name):
        return
    given = _clip(claims.get("given_name"), 100)
    family = _clip(claims.get("family_name"), 100)
    if given and not user.first_name:
        user.first_name = given
    if family and not user.last_name:
        user.last_name = family


def _email_claims(claims: dict[str, Any]) -> tuple[str | None, bool]:
    raw = str(claims.get("email") or "").strip().lower()
    flag = claims.get("email_verified")
    verified = flag is True or str(flag).lower() == "true"
    if not raw or "@" not in raw:
        return None, False
    return raw, verified


def _browser_binding_ok(row: ExternalAuthTransaction, presented: str | None) -> bool:
    """Le state seul ne prouve pas le navigateur qui a ouvert la tentative."""
    secret = str(presented or "").strip()
    stored = str(row.ticket_hash or "")
    if not secret or not stored:
        return False
    return hmac.compare_digest(_digest(secret), stored)


def _lock_pending(state: str) -> ExternalAuthTransaction:
    secret = str(state or "").strip()
    if not secret:
        raise ExternalAuthRejected(
            "state_invalide", "Cette tentative n'est plus valable."
        )
    row = (
        ExternalAuthTransaction.query.filter_by(state_hash=_digest(secret))
        .with_for_update()
        .one_or_none()
    )
    if row is None or row.status != "pending" or _expired(row):
        raise ExternalAuthRejected(
            "rejeu"
            if row is not None and row.status != "pending"
            else "state_invalide",
            "Cette tentative n'est plus valable.",
        )
    return row


def _fail(row: ExternalAuthTransaction, reason: str) -> None:
    row.status = "failed"
    row.code_verifier = None
    _persist()
    logger.info("connexion_externe_refusee raison=%s", reason)


def _expired(row: ExternalAuthTransaction) -> bool:
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


def _authorize_url(provider: str, state: str, nonce: str, verifier: str) -> str:
    challenge = hashlib.sha256(verifier.encode("utf-8")).digest()
    import base64

    code_challenge = base64.urlsafe_b64encode(challenge).rstrip(b"=").decode("ascii")
    query = {
        "client_id": _client_id(provider),
        "redirect_uri": _redirect_uri(provider),
        "response_type": "code",
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if provider == "google":
        query["scope"] = "openid email profile"
        query["prompt"] = "select_account"
        return f"{GOOGLE_AUTH}?{urlencode(query)}"
    query["scope"] = "openid email name"
    query["response_mode"] = "form_post"
    return f"{APPLE_AUTH}?{urlencode(query)}"


def _apple_client_secret() -> str:
    key = _required("EXTERNAL_AUTH_APPLE_PRIVATE_KEY").replace("\\n", "\n")
    team = _required("EXTERNAL_AUTH_APPLE_TEAM_ID")
    key_id = _required("EXTERNAL_AUTH_APPLE_KEY_ID")
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "iss": team,
            "iat": now,
            "exp": now + timedelta(minutes=10),
            "aud": "https://appleid.apple.com",
            "sub": _client_id("apple"),
        },
        key,
        algorithm="ES256",
        headers={"kid": key_id},
    )


def _post_form(url: str, body: dict[str, str]) -> dict[str, Any]:
    data = urlencode(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        # Token endpoint constant (GOOGLE_TOKEN/APPLE_TOKEN), hors contrôle utilisateur.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        logger.warning("echange_code_impossible")
        raise ExternalAuthRejected(
            "fournisseur_indisponible",
            "Le fournisseur est indisponible.",
            status=503,
        ) from error
    if not isinstance(payload, dict):
        raise ExternalAuthRejected("jeton_absent", "Réponse fournisseur illisible.")
    return payload


def _jwk(jwks: dict[str, Any], kid: str) -> Any:
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        raise ExternalAuthRejected("jeton_invalide", "Clés du fournisseur illisibles.")
    chosen = next((item for item in keys if str(item.get("kid") or "") == kid), None)
    if not isinstance(chosen, dict):
        raise ExternalAuthRejected("jeton_invalide", "Clé de signature introuvable.")
    encoded = json.dumps(chosen)
    if chosen.get("kty") == "EC":
        return ECAlgorithm.from_jwk(encoded)
    return RSAAlgorithm.from_jwk(encoded)


def _configured(provider: str) -> bool:
    if not _client_id(provider) or not _redirect_uri(provider):
        return False
    if provider == "google":
        return bool(_setting("EXTERNAL_AUTH_GOOGLE_CLIENT_SECRET"))
    return bool(
        _setting("EXTERNAL_AUTH_APPLE_TEAM_ID")
        and _setting("EXTERNAL_AUTH_APPLE_KEY_ID")
        and _setting("EXTERNAL_AUTH_APPLE_PRIVATE_KEY")
    )


def _client_id(provider: str) -> str:
    return _setting(f"EXTERNAL_AUTH_{provider.upper()}_CLIENT_ID")


def _redirect_uri(provider: str) -> str:
    return _setting(f"EXTERNAL_AUTH_{provider.upper()}_REDIRECT_URI")


def _required(name: str) -> str:
    value = _setting(name)
    if not value:
        raise ExternalAuthRejected(
            "fournisseur_indisponible",
            "Ce fournisseur n'est pas configuré.",
            status=503,
        )
    return value


def _setting(name: str) -> str:
    """La configuration Flask prime. L'environnement sert au déploiement réel."""
    try:
        value = current_app.config.get(name)
    except RuntimeError:
        value = None
    if value is None or not str(value).strip():
        import os

        value = os.environ.get(name)
    return str(value or "").strip()


def _provider(provider: str) -> str:
    name = str(provider or "").strip().lower()
    if name not in PROVIDERS:
        raise ExternalAuthRejected("fournisseur_inconnu", "Fournisseur inconnu.")
    return name


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clip(value: Any, limit: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:limit]


def _persist() -> None:
    db.session.flush()
    if not current_app.config.get("TESTING"):
        db.session.commit()
