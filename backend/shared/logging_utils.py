"""
Utilitaires pour masquer les données sensibles (PII) dans les logs
Conformité GDPR-like
"""
import logging
import re
from typing import Any

# Patterns à masquer (renforcés pour RGPD/OWASP)
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\+?\d[\d\s\-\(\)]{6,20}\d')
# ✅ Pattern IBAN Suisse spécifique (CHxx xxxx xxxx xxxx xxxx x)
IBAN_PATTERN = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b')
IBAN_CH_PATTERN = re.compile(r'\bCH\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{1}\b')
# ✅ Pattern carte bancaire (16 chiffres avec espaces/tirets optionnels)
CARD_PATTERN = re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b')
# ✅ Pattern téléphone suisse (079... 10 chiffres)
PHONE_CH_PATTERN = re.compile(r'\b0\d{9}\b')

def mask_email(email: str) -> str:
    """
    Masque email: john.doe@example.com → j***@e***.com
    """
    if not email or '@' not in email:
        return email

    local, domain = email.rsplit('@', 1)
    domain_parts = domain.split('.')

    masked_local = local[0] + '***' if len(local) > 0 else '***'
    masked_domain_name = domain_parts[0][0] + '***' if len(domain_parts[0]) > 0 else '***'
    masked_domain = masked_domain_name + '.' + '.'.join(domain_parts[1:])

    return f"{masked_local}@{masked_domain}"

def mask_phone(phone: str) -> str:
    """
    Masque téléphone: +41 22 123 45 67 → +41 ** *** ** 67
    """
    if not phone:
        return phone

    digits = re.sub(r'\D', '', phone)
    if len(digits) < 4:
        return '***'

    # Garder préfixe pays + 2 derniers chiffres
    return f"{phone[:3]} ** *** ** {digits[-2:]}"

def mask_iban(iban: str) -> str:
    """
    Masque IBAN: CH65 0900 0000 1234 5678 9 → CH** **** **** **** **89
    """
    if not iban or len(iban) < 8:
        return iban

    return f"{iban[:2]}** **** **** **** **{iban[-2:]}"

def sanitize_log_data(data: Any) -> Any:
    """
    Nettoie récursivement les données sensibles dans dict/str
    """
    if isinstance(data, dict):
        return {k: sanitize_log_data(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        return [sanitize_log_data(item) for item in data]

    if isinstance(data, str):
        # Remplacer patterns sensibles
        sanitized = EMAIL_PATTERN.sub(lambda m: mask_email(m.group(0)), data)
        sanitized = PHONE_PATTERN.sub(lambda m: mask_phone(m.group(0)), sanitized)
        sanitized = IBAN_PATTERN.sub(lambda m: mask_iban(m.group(0)), sanitized)
        # ✅ SECURITY: Patterns supplémentaires pour RGPD
        sanitized = IBAN_CH_PATTERN.sub('[IBAN_REDACTED]', sanitized)
        sanitized = CARD_PATTERN.sub('[CARD_REDACTED]', sanitized)
        sanitized = PHONE_CH_PATTERN.sub('[PHONE_REDACTED]', sanitized)
        return sanitized

    return data

class PIIFilter(logging.Filter):
    """Filtre logging pour masquer PII automatiquement"""

    def filter(self, record: logging.LogRecord) -> bool:
        # Masquer dans le message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = sanitize_log_data(record.msg)

        # Masquer dans args
        if hasattr(record, 'args') and record.args:
            record.args = tuple(sanitize_log_data(arg) for arg in record.args)

        return True

