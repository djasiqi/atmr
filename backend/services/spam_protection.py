# backend/services/spam_protection.py
"""Service anti-spam pour limiter le taux d'envoi de messages."""

import logging
import os
import time
from typing import Optional, cast

from ext import redis_client

logger = logging.getLogger(__name__)

# Configuration anti-spam
SPAM_RATE_LIMIT_SECONDS = float(
    os.getenv("SPAM_RATE_LIMIT_SECONDS", "1.0")
)  # 1 message par seconde
SPAM_REDIS_KEY_PREFIX = "user_msg_ts:"
SPAM_REDIS_TTL = int(os.getenv("SPAM_REDIS_TTL", "2"))  # TTL de 2 secondes


def can_send_message(user_id: int) -> tuple[bool, Optional[str]]:
    """
    Vérifie si un utilisateur peut envoyer un message (anti-spam).

    Args:
        user_id: ID de l'utilisateur

    Returns:
        Tuple (allowed, error_message):
        - allowed: True si l'utilisateur peut envoyer, False sinon
        - error_message: Message d'erreur si refusé (None si autorisé)
    """
    if not redis_client:
        # Si Redis n'est pas disponible, on autorise (fail-open)
        logger.warning("⚠️ Redis non disponible - anti-spam désactivé")
        return True, None

    try:
        key = f"{SPAM_REDIS_KEY_PREFIX}{user_id}"
        now = time.time()

        # Récupérer le dernier timestamp d'envoi
        last_time_str = redis_client.get(key)
        if last_time_str:
            try:
                # Gérer les types bytes ou str retournés par Redis
                # Cast pour indiquer que ce n'est pas un Awaitable
                value: bytes | str = cast(bytes | str, last_time_str)
                if isinstance(value, bytes):
                    last_time = float(value.decode("utf-8"))
                else:
                    last_time = float(value)
                elapsed = now - last_time

                if elapsed < SPAM_RATE_LIMIT_SECONDS:
                    # Trop rapide - spam détecté
                    wait_time = SPAM_RATE_LIMIT_SECONDS - elapsed
                    logger.warning(
                        (
                            "🚫 Anti-spam: Utilisateur %s a envoyé un message "
                            "il y a %.2fs (limite: %ss)"
                        ),
                        user_id,
                        elapsed,
                        SPAM_RATE_LIMIT_SECONDS,
                    )
                    return (
                        False,
                        f"Trop de messages. Attendez {wait_time:.1f} seconde(s).",
                    )

            except (ValueError, TypeError):
                # Valeur invalide dans Redis, on continue
                logger.warning("⚠️ Anti-spam: Valeur invalide dans Redis pour %s", key)

        # Mettre à jour le timestamp
        redis_client.set(key, str(now), ex=SPAM_REDIS_TTL)

        return True, None

    except Exception as e:
        logger.error("❌ Anti-spam: Erreur Redis - %s", e)
        # Fail-open: on autorise en cas d'erreur Redis
        return True, None
