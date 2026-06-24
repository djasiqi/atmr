"""
Tests pour les utilitaires de masquage PII (logging_utils).
"""

import logging

import pytest

from shared.logging_utils import (
    KafkaSelectorNoiseFilter,
    PIIFilter,
    configure_kafka_log_noise,
    mask_email,
    mask_iban,
    mask_phone,
    sanitize_log_data,
)


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
    data = {
        "name": "John",
        "email": "john@example.com",
        "nested": {"phone": "+41791234567"},
    }
    sanitized = sanitize_log_data(data)

    assert sanitized["name"] == "John"
    assert "***" in sanitized["email"]
    assert "**" in sanitized["nested"]["phone"]


def _make_kafka_record(name: str, msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_kafka_noise_filter_drops_task_already_done():
    """Le filtre supprime le bruit bénin « Task is already done! »."""
    flt = KafkaSelectorNoiseFilter()
    noisy = _make_kafka_record("kafka.net.selector", "RuntimeError: Task is already done!")
    useful = _make_kafka_record("kafka.conn", "Broker connection lost")
    other = _make_kafka_record("app", "Task is already done!")

    assert flt.filter(noisy) is False
    assert flt.filter(useful) is True
    # Hors namespace kafka.* : non filtré
    assert flt.filter(other) is True


def test_configure_kafka_log_noise_sets_selector_level_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Défaut : le logger kafka.net.selector passe à CRITICAL + filtre installé."""
    monkeypatch.delenv("KAFKA_SELECTOR_LOG_LEVEL", raising=False)
    selector = logging.getLogger("kafka.net.selector")
    selector.setLevel(logging.NOTSET)

    configure_kafka_log_noise()

    assert selector.level == logging.CRITICAL
    assert any(isinstance(f, KafkaSelectorNoiseFilter) for f in selector.filters)


def test_kafka_selector_log_level_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KAFKA_SELECTOR_LOG_LEVEL surcharge le défaut (rollback du filtrage)."""
    monkeypatch.setenv("KAFKA_SELECTOR_LOG_LEVEL", "WARNING")
    selector = logging.getLogger("kafka.net.selector")
    selector.setLevel(logging.NOTSET)

    configure_kafka_log_noise()

    assert selector.level == logging.WARNING


def test_configure_kafka_log_noise_idempotent_filter() -> None:
    """Appels répétés n'ajoutent pas le filtre en double."""
    selector = logging.getLogger("kafka.net.selector")
    before = sum(isinstance(f, KafkaSelectorNoiseFilter) for f in selector.filters)
    configure_kafka_log_noise()
    configure_kafka_log_noise()
    after = sum(isinstance(f, KafkaSelectorNoiseFilter) for f in selector.filters)
    assert after == max(before, 1)


def test_pii_filter():
    """PIIFilter filtre les logs."""
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
