"""
Tests pour le masquage PII dans les logs (CWE-778).
"""
import logging
from io import StringIO

from shared.logging_utils import PIIFilter, mask_email, mask_gps_coords, mask_iban, mask_phone, sanitize_log_data


def test_mask_email():
    """Test masquage email"""
    assert "j***@e***.com" in mask_email("john.doe@example.com")
    assert "a***@c***.ch" in mask_email("alice@company.ch")


def test_mask_phone():
    """Test masquage téléphone"""
    result = mask_phone("+41 22 123 45 67")
    assert "+41" in result
    assert "67" in result


def test_mask_iban():
    """Test masquage IBAN"""
    result = mask_iban("CH6509000000123456789")
    assert result.startswith("CH**")


def test_mask_gps():
    """Test masquage coordonnées GPS (CWE-778)"""
    result = mask_gps_coords("46.519654", "6.632273")
    assert result == "46.5197, 6.6323 [GPS_APPROX]"

    result2 = mask_gps_coords("47.123456", "7.987654")
    assert result2 == "47.1235, 7.9877 [GPS_APPROX]"


def test_sanitize_log_data_email():
    """Test sanitization email"""
    input_str = "User john.doe@example.com logged in"
    result = sanitize_log_data(input_str)

    assert "john.doe@example.com" not in result
    assert "j***@e***.com" in result


def test_sanitize_log_data_gps():
    """Test sanitization GPS (RGPD Art. 32)"""
    input_str = "Driver location: 46.519654, 6.632273"
    result = sanitize_log_data(input_str)

    # Coordonnées précises NE DOIVENT PAS apparaître
    assert "46.519654" not in result
    assert "6.632273" not in result

    # Version approximative DOIT apparaître
    assert "46.5197, 6.6323 [GPS_APPROX]" in result


def test_sanitize_log_data_iban():
    """Test sanitization IBAN suisse"""
    input_str = "Payment to CH65 0900 0000 1234 5678 9 completed"
    result = sanitize_log_data(input_str)

    assert "CH65 0900" not in result
    assert "[IBAN_REDACTED]" in result


def test_sanitize_log_data_card():
    """Test sanitization carte bancaire"""
    input_str = "Card number 1234 5678 9012 3456 declined"
    result = sanitize_log_data(input_str)

    assert "1234 5678 9012 3456" not in result
    assert "[CARD_REDACTED]" in result


def test_sanitize_log_data_phone():
    """Test sanitization téléphone suisse"""
    input_str = "Contact at 0791234567 for details"
    result = sanitize_log_data(input_str)

    assert "0791234567" not in result
    assert "[PHONE_REDACTED]" in result


def test_pii_filter_integration():
    """Test du filtre PII intégré à logging"""
    # Créer logger de test
    logger = logging.getLogger("test_pii_integration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Nettoyer handlers existants

    # Handler qui capture les logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Ajouter filtre PII
    pii_filter = PIIFilter()
    logger.addFilter(pii_filter)

    # Log avec PII
    logger.info("Driver at 46.519654, 6.632273 contacted john.doe@example.com")

    # Vérifier le log masqué
    log_output = log_stream.getvalue()

    assert "46.519654" not in log_output, "GPS precise should be masked"
    assert "john.doe@example.com" not in log_output, "Email should be masked"
    assert "[GPS_APPROX]" in log_output, "GPS should be approximated"
    assert "j***@e***.com" in log_output, "Email should be masked"


def test_sanitize_nested_dict():
    """Test sanitization dans structures imbriquées"""
    input_data = {
        "user": "john.doe@example.com",
        "location": {
            "coords": "46.519654, 6.632273",
            "address": "Rue Test"
        },
        "phone": "0791234567"
    }

    result = sanitize_log_data(input_data)

    # Vérifier que les PII sont masqués à tous les niveaux
    assert "john.doe@example.com" not in str(result)
    assert "46.519654" not in str(result)
    assert "0791234567" not in str(result)


def test_multiple_pii_in_same_string():
    """Test plusieurs PII dans la même chaîne"""
    input_str = "User john.doe@example.com at 46.519654, 6.632273 called 0791234567"
    result = sanitize_log_data(input_str)

    # Tous les PII doivent être masqués
    assert "john.doe@example.com" not in result
    assert "46.519654" not in result
    assert "0791234567" not in result

    # Les versions masquées doivent être présentes
    assert "j***@e***.com" in result
    assert "[GPS_APPROX]" in result
    assert "[PHONE_REDACTED]" in result

