"""✅ S3: Support des legacy keys JWT pour rotation sans interruption.

Permet de valider les tokens signés avec d'anciennes clés JWT pendant la période
de transition après une rotation de clé.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def get_jwt_secret_keys() -> list[str]:
    """✅ S3: Récupère toutes les clés JWT (actuelle + legacy) pour validation.

    Returns:
        Liste des clés JWT à essayer (actuelle en premier, puis legacy)
    """
    keys: list[str] = []

    # 1. Clé principale (actuelle)
    main_key = os.getenv("JWT_SECRET_KEY")
    if main_key:
        keys.append(main_key)

    # 2. Clés legacy (depuis variable d'environnement)
    legacy_keys_env = os.getenv("JWT_LEGACY_SECRET_KEYS", "")
    if legacy_keys_env:
        # Format: "key1,key2,key3" (séparées par virgule)
        for legacy_key in legacy_keys_env.split(","):
            legacy_key_clean = legacy_key.strip()
            if legacy_key_clean and legacy_key_clean not in keys:
                keys.append(legacy_key_clean)

    # 3. Clé legacy unique (pour compatibilité)
    legacy_key_single = os.getenv("JWT_LEGACY_SECRET_KEY")
    if legacy_key_single and legacy_key_single not in keys:
        keys.append(legacy_key_single)

    return keys


def try_decode_with_legacy_keys(
    encoded_token: str, algorithms: list[str] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """✅ S3: Essaie de décoder un token JWT avec toutes les clés disponibles.

    Essaie d'abord avec la clé principale, puis avec les legacy keys si nécessaire.

    Args:
        encoded_token: Token JWT encodé
        algorithms: Liste des algorithmes à essayer (défaut: ["HS256"])

    Returns:
        Tuple (payload, secret_key_used) ou (None, None) si échec
    """
    if algorithms is None:
        algorithms = ["HS256"]

    import jwt as pyjwt  # pyright: ignore[reportMissingImports]

    keys = get_jwt_secret_keys()
    if not keys:
        logger.warning("[JWT Legacy] Aucune clé JWT disponible")
        return None, None

    # Essayer avec chaque clé
    for key_idx, secret_key in enumerate(keys):
        try:
            payload = pyjwt.decode(
                encoded_token,
                secret_key,
                algorithms=algorithms,
                options={"verify_signature": True},
            )
            # Si décodage réussi avec une legacy key, logger pour audit
            if key_idx > 0:
                logger.debug(
                    "[JWT Legacy] ✅ Token décodé avec legacy key #%d (total legacy keys: %d)",
                    key_idx,
                    len(keys) - 1,
                )
            return payload, secret_key
        except pyjwt.InvalidSignatureError:
            # Signature invalide avec cette clé, essayer la suivante
            continue
        except pyjwt.ExpiredSignatureError:
            # Token expiré (peu importe la clé)
            logger.debug("[JWT Legacy] Token expiré")
            return None, None
        except pyjwt.InvalidTokenError as e:
            # Autre erreur (format invalide, etc.)
            if key_idx == 0:
                # Seulement logger pour la première clé (éviter spam)
                logger.debug("[JWT Legacy] Token invalide: %s", e)
            continue

    # Aucune clé n'a fonctionné
    logger.debug("[JWT Legacy] ❌ Impossible de décoder le token avec aucune clé")
    return None, None
