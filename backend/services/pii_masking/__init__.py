"""✅ Service centralisé pour le masking PII (Personally Identifiable Information).

Ce service unifie toute la logique de masquage PII pour garantir la conformité RGPD
et la protection de la vie privée des utilisateurs.

Usage:
    from services.pii_masking import PIIMaskingService

    # Pour les logs (strict)
    masked_log = PIIMaskingService.mask_log_data(data)

    # Pour les réponses API (selon contexte)
    masked_response = PIIMaskingService.mask_api_response(data)
"""

import logging
import re
from typing import Any

# Patterns à masquer (renforcés pour RGPD/OWASP)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-\(\)]{6,20}\d")
# ✅ Pattern IBAN Suisse spécifique (CHxx xxxx xxxx xxxx xxxx x)
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
IBAN_CH_PATTERN = re.compile(r"\bCH\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{1}\b")
# ✅ Pattern carte bancaire (16 chiffres avec espaces/tirets optionnels)
CARD_PATTERN = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")
# ✅ Pattern téléphone suisse (079... 10 chiffres)
PHONE_CH_PATTERN = re.compile(r"\b0\d{9}\b")
# ✅ Pattern GPS haute précision (6+ décimales) - RGPD Art. 32
GPS_PATTERN = re.compile(r"\b(\d+\.\d{6,}),\s*(\d+\.\d{6,})\b")

# Constantes pour éviter les valeurs magiques
MIN_PHONE_LENGTH = 4
MIN_IBAN_LENGTH = 8

# Clés sensibles à masquer automatiquement dans les dictionnaires
SENSITIVE_KEYS = {
    "password",
    "secret",
    "token",
    "api_key",
    "access_key",
    "secret_key",
    "jwt",
    "authorization",
    "auth",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
    "privatekey",
    "credential",
    "credentials",
}

# Pattern pour détecter les tokens dans les chaînes (token: value, key: value, etc.)
TOKEN_PATTERN = re.compile(
    r"(?i)(token|key|secret|password|apikey|access_key|secret_key|authorization|auth)\s*[:=]\s*['\"]?([^'\"]\S+)",
    re.IGNORECASE,
)


class PIIMaskingService:
    """Service centralisé pour le masking PII.

    Fournit des méthodes pour masquer les données personnelles identifiables
    dans les logs et les réponses API, garantissant la conformité RGPD.
    """

    @staticmethod
    def mask_email(email: str) -> str:
        """Masque email: john.doe@example.com → j***@e***.com.

        Args:
            email: Adresse email à masquer

        Returns:
            Email masqué avec format préservé
        """
        if not email or "@" not in email:
            return email

        local, domain = email.rsplit("@", 1)
        domain_parts = domain.split(".")

        masked_local = local[0] + "***" if len(local) > 0 else "***"
        masked_domain_name = domain_parts[0][0] + "***" if len(domain_parts[0]) > 0 else "***"
        masked_domain = masked_domain_name + "." + ".".join(domain_parts[1:])

        return f"{masked_local}@{masked_domain}"

    @staticmethod
    def mask_phone(phone: str) -> str:
        """Masque téléphone: +41 22 123 45 67 → +41 ** *** ** 67.

        Args:
            phone: Numéro de téléphone à masquer

        Returns:
            Téléphone masqué avec préfixe et 2 derniers chiffres visibles
        """
        if not phone:
            return phone

        digits = re.sub(r"\D", "", phone)
        if len(digits) < MIN_PHONE_LENGTH:
            return "***"

        # Garder préfixe pays + 2 derniers chiffres
        return f"{phone[:3]} ** *** ** {digits[-2:]}"

    @staticmethod
    def mask_iban(iban: str) -> str:
        """Masque IBAN: CH65 0900 0000 1234 5678 9 → CH** **** **** **** **89.

        Args:
            iban: IBAN à masquer

        Returns:
            IBAN masqué avec code pays et 2 derniers caractères visibles
        """
        if not iban or len(iban) < MIN_IBAN_LENGTH:
            return iban

        return f"{iban[:2]}** **** **** **** **{iban[-2:]}"

    @staticmethod
    def mask_gps_coords(lat: str, lon: str) -> str:
        """Réduit précision GPS de 6+ décimales à 4 décimales.

        Précision GPS:
        - 6 décimales: ~0.11m (identification individu)
        - 4 décimales: ~11m (conformité RGPD)

        Exemple: "46.519654, 6.632273" → "46.5197, 6.6323 [GPS_APPROX]".

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Coordonnées GPS avec précision réduite
        """
        try:
            lat_float = float(lat)
            lon_float = float(lon)
            # Arrondir à 4 décimales (précision ~11m)
            return f"{lat_float:.4f}, {lon_float:.4f} [GPS_APPROX]"
        except (ValueError, TypeError):
            return f"{lat}, {lon} [GPS_REDACTED]"

    @staticmethod
    def mask_log_data(data: Any) -> Any:
        """Masque les données PII dans les logs (strict).

        Nettoie récursivement les données sensibles dans dict/str/list.
        Masque automatiquement:
        - Clés sensibles dans les dictionnaires (password, secret, token, etc.)
        - Patterns de tokens dans les chaînes (token: value, key: value, etc.)
        - PII (emails, téléphones, IBAN, cartes, GPS)

        Args:
            data: Données à masquer (dict, str, list, ou autre)

        Returns:
            Données masquées avec même structure
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                key_lower = str(key).lower()
                # Vérifier si la clé contient un mot-clé sensible
                if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
                    sanitized[key] = "[REDACTED]"
                else:
                    # Sanitizer récursivement la valeur
                    sanitized[key] = PIIMaskingService.mask_log_data(value)
            return sanitized

        if isinstance(data, (list, tuple)):
            return [PIIMaskingService.mask_log_data(item) for item in data]

        if isinstance(data, str):
            # ✅ IMPORTANT: Appliquer patterns spécifiques AVANT patterns génériques
            # pour éviter conflits (ex: 0791234567 masqué par PHONE_PATTERN avant PHONE_CH_PATTERN)

            # 1. Masquer les patterns de tokens dans les chaînes (token: value, key: value, etc.)
            sanitized = TOKEN_PATTERN.sub(r"\1: [REDACTED]", data)

            # 2. Patterns spécifiques (prioritaires)
            sanitized = IBAN_CH_PATTERN.sub("[IBAN_REDACTED]", sanitized)
            sanitized = CARD_PATTERN.sub("[CARD_REDACTED]", sanitized)
            sanitized = PHONE_CH_PATTERN.sub("[PHONE_REDACTED]", sanitized)
            # ✅ SECURITY CWE-778: Masquer coordonnées GPS précises (RGPD Art. 32)
            sanitized = GPS_PATTERN.sub(lambda m: PIIMaskingService.mask_gps_coords(m.group(1), m.group(2)), sanitized)

            # 3. Patterns génériques (fallback)
            sanitized = EMAIL_PATTERN.sub(lambda m: PIIMaskingService.mask_email(m.group(0)), sanitized)
            sanitized = PHONE_PATTERN.sub(lambda m: PIIMaskingService.mask_phone(m.group(0)), sanitized)
            return IBAN_PATTERN.sub(lambda m: PIIMaskingService.mask_iban(m.group(0)), sanitized)

        return data

    @staticmethod
    def mask_api_response(data: dict[str, Any]) -> dict[str, Any]:
        """Masque les données PII dans les réponses API (selon contexte).

        Version plus permissive que mask_log_data() pour les réponses API.
        Peut masquer moins de données selon le contexte d'utilisation.

        Args:
            data: Réponse API à masquer (dict)

        Returns:
            Réponse API masquée
        """
        # Pour l'instant, utilise la même logique que mask_log_data
        # Peut être adapté plus tard selon les besoins spécifiques de l'API
        return PIIMaskingService.mask_log_data(data)


class PIIFilter(logging.Filter):
    """Filtre logging pour masquer PII automatiquement.

    Utilise PIIMaskingService pour masquer les données sensibles dans les logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        """Filtre les enregistrements de log pour masquer les PII.

        Args:
            record: Enregistrement de log à filtrer

        Returns:
            True si l'enregistrement doit être loggé
        """
        # Masquer dans le message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = PIIMaskingService.mask_log_data(record.msg)

        # Masquer dans args
        if hasattr(record, "args") and record.args:
            record.args = tuple(PIIMaskingService.mask_log_data(arg) for arg in record.args)

        return True
