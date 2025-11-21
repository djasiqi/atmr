#!/usr/bin/env python3
"""
Tests de PII masking pour l'Étape 15.

Ces tests valident le système de masquage des données personnelles
identifiables (PII) pour garantir la conformité RGPD et la protection
de la vie privée des utilisateurs.
✅ FIX: Tests simplifiés pour utiliser les vraies fonctions de shared.logging_utils
au lieu de mocks de classes inexistantes.
"""

import logging
import sys
from io import StringIO
from pathlib import Path

import pytest

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from shared.logging_utils import (
    PIIFilter,
    mask_email,
    mask_gps_coords,
    mask_iban,
    mask_phone,
    sanitize_log_data,
)


class TestPIIMasking:
    """Tests du système de masquage PII avec les fonctions réelles."""

    def test_pii_masking_email(self):
        """Test de masquage des adresses email."""
        print("🧪 Test masquage des adresses email...")

        original_email = "user@example.com"
        masked_email = mask_email(original_email)

        assert masked_email is not None
        assert "@" in masked_email
        assert "***" in masked_email
        assert original_email != masked_email
        # Vérifier que le format masqué est correct
        assert "u***@" in masked_email or "u***@" in masked_email.lower()
        print("  ✅ Masquage email fonctionnel")

    def test_pii_masking_phone(self):
        """Test de masquage des numéros de téléphone."""
        print("🧪 Test masquage des numéros de téléphone...")

        original_phone = "+33123456789"
        masked_phone = mask_phone(original_phone)

        assert masked_phone is not None
        assert original_phone != masked_phone
        # Vérifier que le préfixe et les derniers chiffres sont présents
        assert "+33" in masked_phone or "+3" in masked_phone
        assert "***" in masked_phone
        print("  ✅ Masquage téléphone fonctionnel")

    def test_pii_masking_iban(self):
        """Test de masquage des IBAN."""
        print("🧪 Test masquage des IBAN...")

        original_iban = "CH6509000000123456789"
        masked_iban = mask_iban(original_iban)

        assert masked_iban is not None
        assert original_iban != masked_iban
        # Vérifier que le préfixe pays et les derniers chiffres sont présents
        assert "CH**" in masked_iban or "CH" in masked_iban
        assert "***" in masked_iban
        print("  ✅ Masquage IBAN fonctionnel")

    def test_pii_masking_gps(self):
        """Test de masquage des coordonnées GPS."""
        print("🧪 Test masquage des coordonnées GPS...")

        original_lat = "46.519654"
        original_lon = "6.632273"
        masked_gps = mask_gps_coords(original_lat, original_lon)

        assert masked_gps is not None
        assert "[GPS_APPROX]" in masked_gps
        # Vérifier que les coordonnées sont arrondies (4 décimales)
        assert "46.5197" in masked_gps
        assert "6.6323" in masked_gps
        assert original_lat not in masked_gps  # Coordonnées précises masquées
        assert original_lon not in masked_gps
        print("  ✅ Masquage GPS fonctionnel")

    def test_sanitize_log_data_email(self):
        """Test de sanitization des emails dans les données."""
        print("🧪 Test sanitization des emails...")

        test_data = {
            "user_email": "user@example.com",
            "user_name": "John Doe",
        }

        sanitized = sanitize_log_data(test_data)

        assert isinstance(sanitized, dict)
        # Vérifier que l'email est masqué
        assert "user@example.com" not in str(sanitized)
        assert "***" in str(sanitized["user_email"]) or "@" not in str(sanitized["user_email"])
        print("  ✅ Sanitization email fonctionnelle")

    def test_sanitize_log_data_phone(self):
        """Test de sanitization des téléphones dans les données."""
        print("🧪 Test sanitization des téléphones...")

        test_data = {
            "user_phone": "+33123456789",
            "user_name": "John Doe",
        }

        sanitized = sanitize_log_data(test_data)

        assert isinstance(sanitized, dict)
        # Vérifier que le téléphone est masqué
        assert "+33123456789" not in str(sanitized)
        assert "[PHONE_REDACTED]" in str(sanitized["user_phone"]) or "***" in str(sanitized["user_phone"])
        print("  ✅ Sanitization téléphone fonctionnelle")

    def test_sanitize_log_data_gps(self):
        """Test de sanitization des coordonnées GPS dans les données."""
        print("🧪 Test sanitization des coordonnées GPS...")

        test_data = {
            "location": "46.519654, 6.632273",
            "user_name": "John Doe",
        }

        sanitized = sanitize_log_data(test_data)

        assert isinstance(sanitized, dict)
        # Vérifier que les coordonnées précises sont masquées
        assert "46.519654" not in str(sanitized)
        assert "6.632273" not in str(sanitized)
        assert "[GPS_APPROX]" in str(sanitized["location"])
        print("  ✅ Sanitization GPS fonctionnelle")

    def test_sanitize_log_data_complete(self):
        """Test de sanitization complète des données."""
        print("🧪 Test sanitization complète des données...")

        test_data = {
            "user_email": "user@example.com",
            "user_phone": "+33123456789",
            "user_iban": "CH6509000000123456789",
            "location": "46.519654, 6.632273",
            "password": "secret123",  # Clé sensible
            "user_name": "John Doe",
        }

        sanitized = sanitize_log_data(test_data)

        assert isinstance(sanitized, dict)
        # Vérifier que toutes les PII sont masquées
        assert "user@example.com" not in str(sanitized)
        assert "+33123456789" not in str(sanitized)
        assert "CH6509000000123456789" not in str(sanitized)
        assert "46.519654" not in str(sanitized)
        assert "secret123" not in str(sanitized)
        # Vérifier que les clés sensibles sont masquées
        assert sanitized.get("password") == "[REDACTED]"
        print("  ✅ Sanitization complète fonctionnelle")

    def test_sanitize_log_data_string(self):
        """Test de sanitization d'une chaîne de caractères."""
        print("🧪 Test sanitization d'une chaîne...")

        test_string = "User user@example.com at 46.519654, 6.632273 called +33123456789"

        sanitized = sanitize_log_data(test_string)

        assert isinstance(sanitized, str)
        # Vérifier que toutes les PII sont masquées dans la chaîne
        assert "user@example.com" not in sanitized
        assert "46.519654" not in sanitized
        assert "6.632273" not in sanitized
        assert "+33123456789" not in sanitized
        # Vérifier que les versions masquées sont présentes
        assert "***" in sanitized or "[PHONE_REDACTED]" in sanitized
        assert "[GPS_APPROX]" in sanitized
        print("  ✅ Sanitization chaîne fonctionnelle")

    def test_sanitize_log_data_nested(self):
        """Test de sanitization dans des structures imbriquées."""
        print("🧪 Test sanitization structures imbriquées...")

        test_data = {
            "user": {
                "email": "user@example.com",
                "phone": "+33123456789",
                "location": {"coords": "46.519654, 6.632273", "address": "Rue Test"},
            },
            "bookings": [
                {"id": 1, "patient_email": "patient@example.com"},
                {"id": 2, "patient_phone": "+33111111111"},
            ],
        }

        sanitized = sanitize_log_data(test_data)

        assert isinstance(sanitized, dict)
        # Vérifier que les PII sont masquées à tous les niveaux
        assert "user@example.com" not in str(sanitized)
        assert "+33123456789" not in str(sanitized)
        assert "46.519654" not in str(sanitized)
        assert "patient@example.com" not in str(sanitized)
        assert "+33111111111" not in str(sanitized)
        print("  ✅ Sanitization structures imbriquées fonctionnelle")

    def test_pii_filter_integration(self):
        """Test d'intégration du filtre PII avec logging."""
        print("🧪 Test intégration filtre PII avec logging...")

        # Créer logger de test
        logger = logging.getLogger("test_pii_filter")
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
        logger.info("User user@example.com at 46.519654, 6.632273 called +33123456789")

        # Vérifier le log masqué
        log_output = log_stream.getvalue()

        assert "user@example.com" not in log_output
        assert "46.519654" not in log_output
        assert "6.632273" not in log_output
        assert "+33123456789" not in log_output
        # Vérifier que les versions masquées sont présentes
        assert "[GPS_APPROX]" in log_output or "***" in log_output
        print("  ✅ Intégration filtre PII avec logging fonctionnelle")


class TestPIIMaskingIntegration:
    """Tests d'intégration du masquage PII avec le système de dispatch."""

    def test_pii_masking_dispatch_integration(self):
        """Test d'intégration du masquage PII avec le dispatch."""
        print("🧪 Test intégration masquage PII avec dispatch...")

        # Simuler un booking avec PII
        original_booking = {
            "booking_id": 123,
            "patient_name": "John Doe",
            "patient_email": "patient@example.com",
            "patient_phone": "+33123456789",
            "pickup_location": "46.519654, 6.632273",
        }

        # Utiliser sanitize_log_data pour masquer les PII
        masked_booking = sanitize_log_data(original_booking)

        assert masked_booking is not None
        assert "patient@example.com" not in str(masked_booking)
        assert "+33123456789" not in str(masked_booking)
        assert "46.519654" not in str(masked_booking)
        # Vérifier que les versions masquées sont présentes
        assert "***" in str(masked_booking) or "[GPS_APPROX]" in str(masked_booking)
        print("  ✅ Intégration masquage PII avec dispatch fonctionnelle")

    def test_pii_masking_logging_integration(self):
        """Test d'intégration du masquage PII avec le système de logging."""
        print("🧪 Test intégration masquage PII avec logging...")

        # Créer logger avec filtre PII
        logger = logging.getLogger("test_pii_logging")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        pii_filter = PIIFilter()
        logger.addFilter(pii_filter)

        # Log avec données PII
        masked_data = {"user_email": "user@example.com", "user_phone": "+33123456789"}
        logger.info("Processing booking with PII: %s", masked_data)

        log_output = log_stream.getvalue()

        # Vérifier que les PII sont masquées
        assert "user@example.com" not in log_output
        assert "+33123456789" not in log_output
        # Vérifier que les versions masquées sont présentes
        assert "***" in log_output or "[PHONE_REDACTED]" in log_output
        print("  ✅ Intégration masquage PII avec logging fonctionnelle")


if __name__ == "__main__":
    # Exécution des tests
    print("🚀 TESTS DE MASQUAGE PII")
    print("=" * 50)

    test_instance = TestPIIMasking()

    # Tests de base
    test_instance.test_pii_masking_email()
    test_instance.test_pii_masking_phone()
    test_instance.test_pii_masking_iban()
    test_instance.test_pii_masking_gps()
    test_instance.test_sanitize_log_data_email()
    test_instance.test_sanitize_log_data_phone()
    test_instance.test_sanitize_log_data_gps()
    test_instance.test_sanitize_log_data_complete()
    test_instance.test_sanitize_log_data_string()
    test_instance.test_sanitize_log_data_nested()
    test_instance.test_pii_filter_integration()

    # Tests d'intégration
    integration_instance = TestPIIMaskingIntegration()
    integration_instance.test_pii_masking_dispatch_integration()
    integration_instance.test_pii_masking_logging_integration()

    print("=" * 50)
    print("✅ TOUS LES TESTS DE MASQUAGE PII RÉUSSIS")
