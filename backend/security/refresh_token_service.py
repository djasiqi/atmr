"""Service de gestion des refresh tokens server-side.

Permet de stocker, vérifier et révoquer les refresh tokens dans la base de données
pour permettre la déconnexion forcée par l'admin.
"""

import hashlib
import logging
from datetime import UTC, datetime

from flask import request
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


ROTATION_GRACE_WINDOW_SECONDS = 300  # 5 minutes (mobile-safe)


def mark_token_rotated(old_token: str, new_token: str) -> bool:
    """Marque un token comme ayant ete rotate vers un nouveau token.

    L'ancien token reste valide (pas revoque) tant que le nouveau n'a pas
    ete utilise. Cela rend la rotation non-destructrice cote client.

    Args:
        old_token: L'ancien refresh token JWT en clair
        new_token: Le nouveau refresh token JWT en clair

    Returns:
        True si le marquage a reussi, False sinon
    """
    old_hash = _hash_refresh_token(old_token)
    new_hash = _hash_refresh_token(new_token)

    token_record = RefreshToken.query.filter_by(
        token_hash=old_hash, is_revoked=False
    ).first()

    if not token_record:
        logger.warning("mark_token_rotated: old token not found (hash: %s)", old_hash[:8])
        return False

    token_record.rotated_to_hash = new_hash
    token_record.rotated_at = datetime.now(UTC)
    db.session.commit()

    logger.info(
        "refresh_soft_rotated user_id=%d old=%s new=%s",
        token_record.user_id,
        old_hash[:8],
        new_hash[:8],
    )
    return True


def _supersede_old_token(token_record: RefreshToken, *, commit: bool = True) -> None:
    """Revoque un ancien token dont le successeur a ete utilise."""
    token_record.is_revoked = True
    token_record.revoked_at = datetime.now(UTC)
    token_record.revoked_reason = "Superseded (new token used)"
    if commit:
        db.session.commit()
    logger.info(
        "refresh_superseded user_id=%d hash=%s cause=new_used",
        token_record.user_id,
        token_record.token_hash[:8],
    )


def is_token_revoked(token: str, grace_window: bool = False) -> bool:
    """Verifie si un refresh token est revoque, expire, ou victime de reuse.

    Rotation soft :
    - Si le token a ete rotate (rotated_to_hash set) mais que le nouveau
      n'a pas encore ete utilise → ancien accepte (grace period)
    - Si le nouveau a ete utilise → ancien revoque (superseded)
    - Si le nouveau a ete utilise ET on revoit l'ancien → reuse detection
      → revoke all sessions (potentiel vol de token)

    Args:
        token: Le refresh token JWT en clair
        grace_window: Si True, applique la grace window legacy

    Returns:
        True si le token est revoque ou expire, False sinon
    """
    token_hash = _hash_refresh_token(token)

    token_record = RefreshToken.query.filter_by(token_hash=token_hash).first()

    if not token_record:
        # Fallback gracieux : si le token n'est pas dans la DB (store_refresh_token
        # a pu echouer silencieusement lors du login/refresh), on ne le traite PAS
        # comme revoque. La validation JWT (signature + expiration) est suffisante.
        # Traiter "absent" comme "revoque" provoquait des deconnexions sur iOS
        # quand le stockage DB echouait de maniere transitoire.
        logger.warning(
            "Token non trouvé dans la DB (hash: %s) — fallback: accepted (JWT-only validation)",
            token_hash[:8],
        )
        return False

    now = datetime.now(UTC)

    if token_record.is_revoked:
        if (
            grace_window
            and token_record.revoked_reason == "Rotation automatique du token"
            and token_record.revoked_at is not None
            and (now - token_record.revoked_at).total_seconds()
            < ROTATION_GRACE_WINDOW_SECONDS
        ):
            logger.info(
                "Token revoked by legacy rotation but in grace window (%ds) — accepted (user_id=%d)",
                ROTATION_GRACE_WINDOW_SECONDS,
                token_record.user_id,
            )
            return False

        logger.debug(
            "Token révoqué (user_id=%d, reason=%s)",
            token_record.user_id,
            token_record.revoked_reason,
        )
        return True

    if token_record.expires_at < now:
        logger.debug(
            "Token expiré (user_id=%d, expires_at=%s)",
            token_record.user_id,
            token_record.expires_at,
        )
        return True

    # Rotation soft : verifier si ce token a ete rotate
    if token_record.rotated_to_hash:
        new_record = RefreshToken.query.filter_by(
            token_hash=token_record.rotated_to_hash
        ).first()

        if not new_record:
            # DB incoherente : le nouveau token n'existe pas.
            # Safe fail : accepter l'ancien (pas de faux positif reuse).
            logger.warning(
                "refresh_soft_rotated: new token not found in DB, accepting old "
                "(user_id=%d, old=%s, expected_new=%s)",
                token_record.user_id,
                token_hash[:8],
                token_record.rotated_to_hash[:8],
            )
            return False

        if new_record.last_used_at is not None:
            # Le nouveau token a deja ete utilise → reuse detection
            logger.warning(
                "refresh_reuse_detected: old token reused after new was used "
                "(user_id=%d, old=%s, new=%s, action=revoke_all)",
                token_record.user_id,
                token_hash[:8],
                token_record.rotated_to_hash[:8],
            )
            revoke_all_user_tokens(
                token_record.user_id,
                reason="Reuse detection — potential token theft",
            )
            return True

        # Le nouveau n'a pas encore ete utilise → ancien accepte
        logger.info(
            "refresh_soft_rotated: new token not yet used — old accepted (user_id=%d)",
            token_record.user_id,
        )
        return False

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
    """Met a jour la date de derniere utilisation d'un token.

    Supersede automatiquement tout ancien token qui a ete rotate vers celui-ci
    (rotation soft : l'ancien n'est revoque qu'apres la premiere utilisation du nouveau).

    Args:
        token: Le refresh token JWT en clair
    """
    token_hash = _hash_refresh_token(token)

    token_record = RefreshToken.query.filter_by(token_hash=token_hash).first()
    if token_record:
        token_record.last_used_at = datetime.now(UTC)

        # Superseder tout ancien token qui pointe vers celui-ci
        old_tokens = RefreshToken.query.filter_by(
            rotated_to_hash=token_hash, is_revoked=False
        ).all()
        for old in old_tokens:
            _supersede_old_token(old, commit=False)

        db.session.commit()
