"""✅ D2: Chiffrement field-level (AES-256) pour données sensibles.

Objectif: Conformité RGPD/LPD pour données santé.

✅ 2.5: Support rotation secrets (multi-clés).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# ✅ D2: Configuration KMS
DEFAULT_KEY_LENGTH = 32  # AES-256
DEFAULT_IV_LENGTH = 16
KEY_DERIVATION_ITERATIONS = 100000


class EncryptionService:
    """✅ D2: Service de chiffrement pour données sensibles (nom, tel, etc.).

    ✅ 2.5: Support rotation progressive avec multi-clés.
    """

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        legacy_keys: Optional[list[bytes]] = None,
        key_rotation_interval: int = 90,  # jours
    ):
        """Initialise le service de chiffrement.

        Args:
            master_key: Clé maître active (générée si None)
            legacy_keys: Liste de clés legacy pour déchiffrement (rotation)
            key_rotation_interval: Intervalle de rotation des clés (jours)
        """
        super().__init__()  # Appel object.__init__() (héritage implicite en Python 3)
        self.master_key = master_key or self._generate_master_key()
        self.legacy_keys = legacy_keys or []
        self.key_rotation_interval = key_rotation_interval

        logger.info("[D2] EncryptionService initialisé avec %d clé(s) legacy", len(self.legacy_keys))

    def _generate_master_key(self) -> bytes:
        """Génère une clé maître aléatoire."""
        return os.urandom(DEFAULT_KEY_LENGTH)

    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Dérive une clé à partir d'un mot de passe et d'un sel."""
        kdf = PBKDF2HMAC(
            algorithm=algorithms.SHA256(),
            length=DEFAULT_KEY_LENGTH,
            salt=salt,
            iterations=KEY_DERIVATION_ITERATIONS,
            backend=default_backend(),
        )
        return kdf.derive(password)

    def encrypt_field(self, plaintext: str) -> str:
        """✅ D2: Chiffre un champ sensible avec AES-256.

        Args:
            plaintext: Texte en clair à chiffrer

        Returns:
            Chaîne chiffrée encodée en base64
        """
        if not plaintext:
            return ""

        try:
            # Générer IV aléatoire
            iv = os.urandom(DEFAULT_IV_LENGTH)

            # Créer cipher
            cipher = Cipher(algorithms.AES(self.master_key), modes.CBC(iv), backend=default_backend())

            encryptor = cipher.encryptor()

            # Padding PKCS7
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext.encode("utf-8"))
            padded_data += padder.finalize()

            # Chiffrer
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            # Concatenate IV + ciphertext et encoder en base64
            encrypted = iv + ciphertext
            return base64.b64encode(encrypted).decode("utf-8")

        except Exception as e:
            logger.error("[D2] Échec chiffrement: %s", e)
            raise

    def decrypt_field(self, ciphertext: str) -> str:
        """✅ D2: Déchiffre un champ sensible.

        ✅ 2.5: Essaie toutes les clés disponibles (active + legacy) pour rotation.

        Args:
            ciphertext: Texte chiffré encodé en base64

        Returns:
            Texte en clair
        """
        if not ciphertext:
            return ""

        # ✅ 2.5: Liste de toutes les clés à essayer (active en premier)
        all_keys = [self.master_key, *self.legacy_keys]

        # Décoder base64 une seule fois
        try:
            encrypted = base64.b64decode(ciphertext.encode("utf-8"))
        except Exception as e:
            logger.error("[D2] Échec décodage base64: %s", e)
            raise

        # Extraire IV
        if len(encrypted) < DEFAULT_IV_LENGTH:
            raise ValueError("Ciphertext trop court pour contenir IV")

        iv = encrypted[:DEFAULT_IV_LENGTH]
        ciphertext_bytes = encrypted[DEFAULT_IV_LENGTH:]

        # ✅ 2.5: Essayer toutes les clés jusqu'à réussir
        last_error = None
        for key_idx, key in enumerate(all_keys):
            try:
                # Créer cipher
                cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

                decryptor = cipher.decryptor()

                # Déchiffrer
                padded_data = decryptor.update(ciphertext_bytes) + decryptor.finalize()

                # Unpadding PKCS7
                unpadder = padding.PKCS7(128).unpadder()
                plaintext = unpadder.update(padded_data)
                plaintext += unpadder.finalize()

                # Si on utilise une clé legacy, logger pour audit
                if key_idx > 0:
                    logger.debug("[D2] Déchiffrement réussi avec clé legacy #%d", key_idx)

                return plaintext.decode("utf-8")

            except Exception as e:
                last_error = e
                continue  # Essayer la clé suivante

        # Si aucune clé n'a fonctionné
        logger.error("[D2] Échec déchiffrement avec toutes les clés (active + %d legacy)", len(self.legacy_keys))
        if last_error:
            raise last_error
        raise ValueError("Impossible de déchiffrer avec aucune des clés disponibles")


# ✅ 2.5: Méthodes pour gérer la rotation des clés
def add_legacy_key(service: EncryptionService, legacy_key: bytes) -> None:
    """Ajoute une clé legacy pour déchiffrement (rotation).

    Args:
        service: Instance EncryptionService
        legacy_key: Clé legacy à ajouter
    """
    if legacy_key not in service.legacy_keys and legacy_key != service.master_key:
        service.legacy_keys.append(legacy_key)
        logger.info("[2.5] Clé legacy ajoutée (total: %d)", len(service.legacy_keys))


def rotate_to_new_key(service: EncryptionService, new_key: bytes) -> bytes:
    """✅ 2.5: Effectue une rotation de clé (ancienne → nouvelle).

    L'ancienne clé active devient legacy, la nouvelle devient active.

    Args:
        service: Instance EncryptionService
        new_key: Nouvelle clé maître

    Returns:
        Ancienne clé (devenue legacy)
    """
    old_key = service.master_key

    # Ajouter l'ancienne clé aux legacy si pas déjà présente
    if old_key not in service.legacy_keys:
        service.legacy_keys.append(old_key)

    # Définir la nouvelle clé comme active
    service.master_key = new_key

    logger.info("[2.5] Rotation clé effectuée (legacy keys: %d)", len(service.legacy_keys))

    return old_key


# Singleton global
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """✅ D2: Récupère l'instance singleton du service de chiffrement.

    ✅ 2.5: Charge aussi les clés legacy depuis LEGACY_ENCRYPTION_KEYS.
    """
    global _encryption_service  # noqa: PLW0603

    if _encryption_service is None:
        # Charger la clé maître depuis variable d'environnement
        master_key_hex = os.getenv("MASTER_ENCRYPTION_KEY")

        master_key = None
        if master_key_hex:
            try:
                master_key = bytes.fromhex(master_key_hex)
            except ValueError:
                logger.warning("[D2] MASTER_ENCRYPTION_KEY invalide (format hex attendu)")
        else:
            logger.warning("[D2] MASTER_ENCRYPTION_KEY non définie, génération clé temporaire")

        # ✅ 2.5: Charger les clés legacy (séparées par virgule)
        legacy_keys: list[bytes] = []
        legacy_keys_env = os.getenv("LEGACY_ENCRYPTION_KEYS", "")
        if legacy_keys_env:
            for key_hex_raw in legacy_keys_env.split(","):
                key_hex_clean = key_hex_raw.strip()
                if key_hex_clean:
                    try:
                        legacy_key = bytes.fromhex(key_hex_clean)
                        if legacy_key != master_key:  # Éviter doublon
                            legacy_keys.append(legacy_key)
                    except ValueError:
                        logger.warning("[2.5] Clé legacy invalide ignorée: %s...", key_hex_clean[:10])

        _encryption_service = EncryptionService(master_key=master_key, legacy_keys=legacy_keys if legacy_keys else None)

    return _encryption_service


def reset_encryption_service() -> None:
    """Reset le service (pour tests)."""
    global _encryption_service  # noqa: PLW0603
    _encryption_service = None
