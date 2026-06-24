"""✅ Utilitaires pour masquer les données sensibles (PII) dans les logs.

Conformité GDPR-like.

Ce module maintient la compatibilité rétroactive en exposant les fonctions
existantes qui utilisent maintenant le service centralisé PIIMaskingService.
"""

import logging
import os
from typing import Any

from services.security.pii import PIIFilter as _PIIFilter
from services.security.pii import PIIMaskingService

# Loggers kafka-python concernés par le bruit bénin "Task is already done!"
# (race interne du scheduler kafka.net.selector en série 3.x — voir docs/ops).
_KAFKA_NOISE_LOGGERS = ("kafka", "kafka.net.selector", "kafka.client", "kafka.conn")
_KAFKA_NOISE_FILTER_FLAG = "_kafka_noise_filter_installed"


class KafkaSelectorNoiseFilter(logging.Filter):
    """Supprime le bruit bénin kafka-python (race « Task is already done! »).

    Le RuntimeError est déjà catché par kafka-python lui-même ; ces lignes ne
    portent aucune information exploitable et polluent les logs/Sentry.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        is_benign_noise = record.name.startswith("kafka.") and (
            "task is already done" in record.getMessage().lower()
        )
        return not is_benign_noise


def configure_kafka_log_noise(level: str | None = None) -> None:
    """Réduit le bruit kafka-python côté logging stdout (P0-1 roadmap tracking).

    Deux effets complémentaires :
    - abaisse le niveau du logger ``kafka.net.selector`` à
      ``KAFKA_SELECTOR_LOG_LEVEL`` (défaut ``CRITICAL``) ;
    - installe un filtre supprimant les messages « Task is already done! »
      sur les loggers ``kafka.*``.

    Pilotable par la variable d'env ``KAFKA_SELECTOR_LOG_LEVEL`` (rollback :
    repasser à ``ERROR``/``WARNING`` pour réafficher le bruit). Idempotent :
    le filtre n'est pas ajouté en double.
    """
    raw_level = level or os.getenv("KAFKA_SELECTOR_LOG_LEVEL") or "CRITICAL"
    resolved = raw_level.upper()
    selector_level = getattr(logging, resolved, logging.CRITICAL)
    logging.getLogger("kafka.net.selector").setLevel(selector_level)

    for name in _KAFKA_NOISE_LOGGERS:
        log = logging.getLogger(name)
        if not getattr(log, _KAFKA_NOISE_FILTER_FLAG, False):
            log.addFilter(KafkaSelectorNoiseFilter())
            setattr(log, _KAFKA_NOISE_FILTER_FLAG, True)


# ✅ Compatibilité rétroactive : Exposer les fonctions via le service
def mask_email(email: str) -> str:
    """Masque email: john.doe@example.com → j***@e***.com.

    Cette fonction utilise maintenant PIIMaskingService en interne.
    """
    return PIIMaskingService.mask_email(email)


def mask_phone(phone: str) -> str:
    """Masque téléphone: +41 22 123 45 67 → +41 ** *** ** 67.

    Cette fonction utilise maintenant PIIMaskingService en interne.
    """
    return PIIMaskingService.mask_phone(phone)


def mask_iban(iban: str) -> str:
    """Masque IBAN: CH65 0900 0000 1234 5678 9 → CH** **** **** **** **89.

    Cette fonction utilise maintenant PIIMaskingService en interne.
    """
    return PIIMaskingService.mask_iban(iban)


def mask_gps_coords(lat: str, lon: str) -> str:
    """Réduit précision GPS de 6+ décimales à 4 décimales.

    Précision GPS:
    - 6 décimales: ~0.11m (identification individu)
    - 4 décimales: ~11m (conformité RGPD)
    Exemple: "46.519654, 6.632273" → "46.5197, 6.6323 [GPS_APPROX]".

    Cette fonction utilise maintenant PIIMaskingService en interne.
    """
    return PIIMaskingService.mask_gps_coords(lat, lon)


def sanitize_log_data(data: Any) -> Any:
    """Nettoie récursivement les données sensibles dans dict/str/list.

    Masque automatiquement:
    - Clés sensibles dans les dictionnaires (password, secret, token, etc.)
    - Patterns de tokens dans les chaînes (token: value, key: value, etc.)
    - PII (emails, téléphones, IBAN, cartes, GPS)

    Cette fonction utilise maintenant PIIMaskingService.mask_log_data() en interne.
    """
    return PIIMaskingService.mask_log_data(data)


# ✅ Compatibilité rétroactive : Exposer PIIFilter depuis le service
PIIFilter = _PIIFilter
