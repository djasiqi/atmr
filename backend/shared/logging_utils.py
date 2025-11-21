"""✅ Utilitaires pour masquer les données sensibles (PII) dans les logs.

Conformité GDPR-like.

Ce module maintient la compatibilité rétroactive en exposant les fonctions
existantes qui utilisent maintenant le service centralisé PIIMaskingService.
"""

from typing import Any

from services.pii_masking import PIIFilter as _PIIFilter
from services.pii_masking import PIIMaskingService


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
