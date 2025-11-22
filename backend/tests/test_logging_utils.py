"""
Tests pour les utilitaires de masquage PII (logging_utils).
"""


from shared.logging_utils import PIIFilter, mask_email, mask_iban, mask_phone, sanitize_log_data


def test_mask_email():
    """Masquage d'email fonctionne."""
    masked = mask_email("john.doe@example.com")
    assert masked == "j***@e***.com"

    masked2 = mask_email("a@test.ch")
    assert "@" in masked2
    assert "***" in masked2


def test_mask_phone():
    """Masquage de téléphone fonctionne."""
    masked = mask_phone("+41 22 123 45 67")
    assert "**" in masked
    assert masked.endswith("67")


def test_mask_iban():
    """Masquage d'IBAN fonctionne."""
    masked = mask_iban("CH65 0900 0000 1234 5678 9")
    assert masked.startswith("CH**")
    assert "****" in masked


def test_sanitize_log_data_string():
    """Sanitize masque les PII dans les strings."""
    data = "Contact: john@example.com, phone: +41791234567"
    sanitized = sanitize_log_data(data)

    assert "john@example.com" not in sanitized
    assert "***" in sanitized


def test_sanitize_log_data_dict():
    """Sanitize masque PII dans les dicts récursivement."""
    data = {"name": "John", "email": "john@example.com", "nested": {"phone": "+41791234567"}}
    sanitized = sanitize_log_data(data)

    assert sanitized["name"] == "John"
    assert "***" in sanitized["email"]
    assert "**" in sanitized["nested"]["phone"]


def test_pii_filter():
    """PIIFilter filtre les logs."""
    import logging

    filter_obj = PIIFilter()

    # Créer un log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="User email: test@example.com",
        args=(),
        exc_info=None,
    )

    result = filter_obj.filter(record)

    assert result is True
    assert "***" in record.msg
