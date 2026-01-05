# backend/services/clamav_service.py
"""Service ClamAV pour scanner les fichiers uploadés (antivirus)."""

import logging
import os
import socket

logger = logging.getLogger(__name__)

# Configuration ClamAV
CLAMAV_HOST = os.getenv("CLAMAV_HOST", "127.0.0.1")
CLAMAV_PORT = int(os.getenv("CLAMAV_PORT", "3310"))
CLAMAV_ENABLED = os.getenv("CLAMAV_ENABLED", "false").lower() in ("true", "1", "yes")
CLAMAV_TIMEOUT = int(os.getenv("CLAMAV_TIMEOUT", "5"))  # 5 secondes par défaut


def scan_bytes(file_bytes: bytes) -> tuple[bool, str | None]:
    """
    Scanne un fichier avec ClamAV.

    Args:
        file_bytes: Contenu du fichier en bytes

    Returns:
        Tuple (is_safe, error_message):
        - is_safe: True si le fichier est sain, False si contaminé
        - error_message: Message d'erreur si scan échoué (None si OK)
    """
    # Si ClamAV est désactivé, on accepte le fichier
    if not CLAMAV_ENABLED:
        logger.debug("🛡️ ClamAV désactivé - fichier accepté sans scan")
        return True, None

    try:
        # Connexion à ClamAV
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(CLAMAV_TIMEOUT)
        s.connect((CLAMAV_HOST, CLAMAV_PORT))

        # Envoi de la commande INSTREAM
        s.send(b"zINSTREAM\0")

        # Envoi du fichier par chunks
        chunk_size = 1024
        for i in range(0, len(file_bytes), chunk_size):
            chunk = file_bytes[i : i + chunk_size]
            size = len(chunk).to_bytes(4, byteorder="big")
            s.send(size + chunk)

        # Signal de fin
        s.send(b"\0\0\0\0")

        # Réception du résultat
        result = s.recv(2048).decode("utf-8", errors="ignore")
        s.close()

        # Analyse du résultat
        if "FOUND" in result:
            # Virus détecté
            virus_name = (
                result.split("FOUND")[0].strip() if "FOUND" in result else "Unknown"
            )
            logger.warning("🦠 ClamAV: Virus détecté - %s", virus_name)
            return False, f"Fichier infecté détecté: {virus_name}"

        # Fichier sain ou résultat inattendu (fail-open)
        if "OK" in result:
            logger.debug("✅ ClamAV: Fichier sain")
        else:
            logger.warning("⚠️ ClamAV: Résultat inattendu - %s", result)
        return True, None

    except (TimeoutError, ConnectionRefusedError) as e:
        # Fail-open: on accepte le fichier en cas de timeout ou connexion refusée
        error_type = "Timeout" if isinstance(e, socket.timeout) else "Connexion refusée"
        logger.warning("⏱️ ClamAV: %s - fichier accepté (fail-open)", error_type)
        return True, None

    except Exception as e:
        # Fail-open: on accepte le fichier en cas d'erreur
        logger.error("❌ ClamAV: Erreur lors du scan - %s", e)
        return True, None


def is_clamav_available() -> bool:
    """Vérifie si ClamAV est disponible et configuré."""
    if not CLAMAV_ENABLED:
        return False

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((CLAMAV_HOST, CLAMAV_PORT))
        s.close()
        return True
    except Exception:
        return False
