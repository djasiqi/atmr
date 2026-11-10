"""Relie un jeton Apple natif vérifié au compte LIRIE.

Le jeton Web n'est pas accepté. Aucune adresse envoyée par l'application
n'est lue. La session émise ensuite est le contrat mobile, pas un cookie.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from sqlalchemy.exc import IntegrityError

import services.external_auth as external_auth_module
from ext import db
from models import User
from models.external_auth_transaction import ExternalAuthTransaction
from services.apple_native_token import (
    NativeAppleTokenRejected,
    verify_native_apple_identity_token,
)
from services.external_auth import (
    ExternalAuthRejected,
    _advisory_lock,
    _assert_account_open,
    _clip,
    _digest,
    _email_claims,
    _identity_for,
    _link_existing,
    _note_email,
    _persist,
    _setting,
    _touch_names,
    _user_by_email,
)

_TRANSACTION_TTL = timedelta(minutes=10)


def open_native_apple_session(
    token: str,
    raw_nonce: str,
    *,
    intent: str,
    actor_id: int | None,
    names: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Vérifie le jeton, refuse le rejeu, puis retrouve ou prépare le compte."""
    bundle_id = _setting("EXTERNAL_AUTH_APPLE_BUNDLE_ID")
    if not bundle_id:
        raise ExternalAuthRejected(
            "fournisseur_indisponible",
            "Ce fournisseur n'est pas configuré.",
            status=503,
        )
    if intent not in {"login", "link"}:
        raise ExternalAuthRejected("intention_refusee", "Intention inconnue.")
    if intent == "link" and actor_id is None:
        raise ExternalAuthRejected(
            "authentification_requise",
            "Connectez-vous avant d'ajouter cette méthode.",
            status=401,
        )
    try:
        claims = verify_native_apple_identity_token(
            token,
            bundle_id=bundle_id,
            raw_nonce=raw_nonce,
            jwks=external_auth_module.fetch_jwks("apple"),
        )
    except NativeAppleTokenRejected as error:
        raise ExternalAuthRejected(
            error.code,
            "Le jeton d'identité n'est pas accepté.",
            status=400,
        ) from error
    _apply_unsigned_names(claims, names)
    _consume_token(token, raw_nonce, claims)
    subject = str(claims["sub"])
    email, verified = _email_claims(claims)
    _advisory_lock(f"ext-sub:apple:{subject}")
    if email:
        _advisory_lock(f"ext-email:{email}")
    identity = _identity_for("apple", subject)
    if intent == "link":
        holder = SimpleNamespace(user_id=actor_id, provider="apple")
        user = _link_existing(holder, identity, claims, email, verified)
        return {"kind": "session", "user": user}
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
        return {"kind": "session", "user": account}
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
    ready = ExternalAuthTransaction(
        state_hash=_digest(f"ticket:{ticket}"),
        provider="apple",
        nonce_hash=_digest(raw_nonce),
        intent="login",
        status="ready",
        ticket_hash=_digest(ticket),
        provider_sub=subject,
        email=email,
        email_verified=True,
        first_name=_clip(claims.get("given_name"), 100),
        last_name=_clip(claims.get("family_name"), 100),
        expires_at=datetime.now(UTC) + _TRANSACTION_TTL,
    )
    db.session.add(ready)
    _persist()
    return {"kind": "consent", "ticket": ticket}


def _apply_unsigned_names(
    claims: dict[str, Any], names: dict[str, str | None] | None
) -> None:
    if not names:
        return
    if names.get("given_name") and not claims.get("given_name"):
        claims["given_name"] = names["given_name"]
    if names.get("family_name") and not claims.get("family_name"):
        claims["family_name"] = names["family_name"]


def _consume_token(token: str, raw_nonce: str, claims: dict[str, Any]) -> None:
    expires = claims.get("exp")
    if isinstance(expires, datetime):
        expires_at = expires if expires.tzinfo else expires.replace(tzinfo=UTC)
    else:
        expires_at = datetime.fromtimestamp(int(expires or 0), UTC)
    row = ExternalAuthTransaction(
        state_hash=_digest(token),
        provider="apple",
        nonce_hash=_digest(raw_nonce),
        intent="login",
        status="used",
        consumed_at=datetime.now(UTC),
        expires_at=expires_at,
    )
    try:
        with db.session.begin_nested():
            db.session.add(row)
            db.session.flush()
    except IntegrityError as error:
        raise ExternalAuthRejected(
            "rejeu",
            "Cette tentative n'est plus valable.",
            status=409,
        ) from error
