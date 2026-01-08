"""Service de gestion des refresh tokens server-side.

Permet de stocker, vérifier et révoquer les refresh tokens dans la base de données
pour permettre la déconnexion forcée par l'admin.
"""

import hashlib
import logging
from datetime import UTC, datetime

from flask import request  # pyright: ignore[reportMissingImports]
from sqlalchemy import and_

from ext import db
from models import RefreshToken

logger = logging.getLogger(__name__)


def _hash_refresh_token(token: str) -> str:
    """Génère un hash SHA256 du token.

    Args:
        token: Le refresh token JWT en clair

    Returns:
        Hash SHA256 du token (64 caractères hexadécimaux)
    """
    return hashlib.sha256(token.encode()).hexdigest()


def store_refresh_token(
    token: str,
    user_id: int,
    expires_at: datetime,
    device_id: str | None = None,
    device_name: str | None = None,
) -> RefreshToken:
    """Stocke un refresh token dans la base de données.

    Args:
        token: Le refresh token JWT en clair
        user_id: ID de l'utilisateur propriétaire du token
        expires_at: Date d'expiration du token
        device_id: ID de l'appareil (optionnel)
        device_name: Nom de l'appareil (optionnel)

    Returns:
        L'objet RefreshToken créé
    """
    token_hash = _hash_refresh_token(token)

    refresh_token = RefreshToken()
    refresh_token.user_id = user_id
    refresh_token.token_hash = token_hash
    if device_id:
        refresh_token.device_id = device_id
    if device_name:
        refresh_token.device_name = device_name
    if request:
        refresh_token.user_agent = request.headers.get("User-Agent")
        refresh_token.ip_address = request.remote_addr
    refresh_token.expires_at = expires_at
    refresh_token.is_revoked = False

    db.session.add(refresh_token)
    db.session.commit()
    logger.debug(
        "Refresh token stocké pour user_id=%d (device_id=%s)", user_id, device_id
    )
    return refresh_token


def is_token_revoked(token: str) -> bool:
    """Vérifie si un refresh token est révoqué ou expiré.

    Args:
        token: Le refresh token JWT en clair

    Returns:
        True si le token est révoqué ou expiré, False sinon
    """
    token_hash = _hash_refresh_token(token)

    token_record = RefreshToken.query.filter_by(token_hash=token_hash).first()

    if not token_record:
        logger.debug("Token non trouvé dans la DB (hash: %s)", token_hash[:8])
        return True  # Token non trouvé = considéré comme révoqué

    if token_record.is_revoked:
        logger.debug(
            "Token révoqué (user_id=%d, reason=%s)",
            token_record.user_id,
            token_record.revoked_reason,
        )
        return True

    # Vérifier expiration
    now = datetime.now(UTC)
    if token_record.expires_at < now:
        logger.debug(
            "Token expiré (user_id=%d, expires_at=%s)",
            token_record.user_id,
            token_record.expires_at,
        )
        return True

    return False


def revoke_refresh_token(token: str, reason: str | None = None) -> bool:
    """Révoque un refresh token.

    Args:
        token: Le refresh token JWT en clair
        reason: Raison de la révocation (optionnel)

    Returns:
        True si le token a été révoqué, False s'il n'existe pas ou est déjà révoqué
    """
    token_hash = _hash_refresh_token(token)

    token_record = RefreshToken.query.filter_by(
        token_hash=token_hash, is_revoked=False
    ).first()

    if token_record:
        token_record.is_revoked = True
        token_record.revoked_at = datetime.now(UTC)
        token_record.revoked_reason = reason
        db.session.commit()
        logger.info(
            "Refresh token révoqué (user_id=%d, reason=%s)",
            token_record.user_id,
            reason,
        )
        return True

    logger.warning("Tentative de révocation d'un token inexistant ou déjà révoqué")
    return False


def revoke_all_user_tokens(user_id: int, reason: str | None = None) -> int:
    """Révoque tous les refresh tokens actifs d'un utilisateur.

    Args:
        user_id: ID de l'utilisateur
        reason: Raison de la révocation (optionnel, défaut: "Révoqué par l'admin")

    Returns:
        Nombre de tokens révoqués
    """
    now = datetime.now(UTC)
    revoked_reason = reason or "Révoqué par l'admin"

    # Trouver tous les tokens actifs et non expirés
    active_tokens = RefreshToken.query.filter(
        and_(
            RefreshToken.user_id == user_id,
            ~RefreshToken.is_revoked,  # not is_revoked
            RefreshToken.expires_at > now,
        )
    ).all()

    count = len(active_tokens)

    if count > 0:
        for token in active_tokens:
            token.is_revoked = True
            token.revoked_at = now
            token.revoked_reason = revoked_reason

        db.session.commit()
        logger.info(
            "%d refresh token(s) révoqué(s) pour user_id=%d (reason=%s)",
            count,
            user_id,
            revoked_reason,
        )

    return count


def get_user_active_sessions(user_id: int) -> list[RefreshToken]:
    """Récupère toutes les sessions actives d'un utilisateur.

    Args:
        user_id: ID de l'utilisateur

    Returns:
        Liste des RefreshToken actifs (non révoqués et non expirés), triés
        par date de création décroissante
    """
    now = datetime.now(UTC)

    return (
        RefreshToken.query.filter(
            and_(
                RefreshToken.user_id == user_id,
                ~RefreshToken.is_revoked,  # not is_revoked
                RefreshToken.expires_at > now,
            )
        )
        .order_by(RefreshToken.created_at.desc())
        .all()
    )


def update_token_last_used(token: str) -> None:
    """Met à jour la date de dernière utilisation d'un token.

    Args:
        token: Le refresh token JWT en clair
    """
    token_hash = _hash_refresh_token(token)

    token_record = RefreshToken.query.filter_by(token_hash=token_hash).first()
    if token_record:
        token_record.last_used_at = datetime.now(UTC)
        db.session.commit()
