"""Tests pour les schemas de la Phase 2 (CompanyUpdate, DriverCreate, ClientUpdate, DriverProfileUpdate)."""

import pytest
from marshmallow import ValidationError

from schemas.client_schemas import ClientUpdateSchema
from schemas.company_schemas import CompanyUpdateSchema, DriverCreateSchema
from schemas.driver_schemas import DriverProfileUpdateSchema
from schemas.validation_utils import validate_request


class TestCompanyUpdateSchema:
    """Tests pour CompanyUpdateSchema."""

    def test_valid_update(self):
        """Test mise à jour valide."""
        data = {
            "name": "Ma Société",
            "contact_email": "contact@example.com",
            "iban": "CH9300762011623852957",
            "uid_ide": "CHE-123.456.789",
        }
        result = validate_request(CompanyUpdateSchema(), data, strict=False)
        assert result["name"] == "Ma Société"

    def test_invalid_iban(self):
        """Test erreur si IBAN invalide."""
        data = {"iban": "INVALID-IBAN"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(CompanyUpdateSchema(), data, strict=False)
        assert "iban" in exc_info.value.messages.get("errors", {})

    def test_invalid_uid_ide(self):
        """Test erreur si UID IDE invalide."""
        data = {"uid_ide": "INVALID-UID"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(CompanyUpdateSchema(), data, strict=False)
        assert "uid_ide" in exc_info.value.messages.get("errors", {})

    def test_invalid_email(self):
        """Test erreur si email invalide."""
        data = {"contact_email": "invalid-email"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(CompanyUpdateSchema(), data, strict=False)
        assert "contact_email" in exc_info.value.messages.get("errors", {})


class TestDriverCreateSchema:
    """Tests pour DriverCreateSchema."""

    def test_valid_driver_create(self):
        """Test création chauffeur valide."""
        data = {
            "username": "driver1",
            "first_name": "John",
            "last_name": "Doe",
            "email": "driver@example.com",
            "password": "password123",
            "vehicle_assigned": "Vehicle-001",
            "brand": "Mercedes",
            "license_plate": "GE-123-AB",
        }
        result = validate_request(DriverCreateSchema(), data)
        assert result["username"] == "driver1"
        assert result["email"] == "driver@example.com"

    def test_missing_required_fields(self):
        """Test erreur si champs requis manquants."""
        data = {"username": "driver1"}  # Manque email, password, etc.
        with pytest.raises(ValidationError) as exc_info:
            validate_request(DriverCreateSchema(), data)
        errors = exc_info.value.messages.get("errors", {})
        assert "email" in errors or "password" in errors

    def test_password_too_short(self):
        """Test erreur si mot de passe trop court."""
        data = {
            "username": "driver1",
            "first_name": "John",
            "last_name": "Doe",
            "email": "driver@example.com",
            "password": "short",  # < 8 caractères
            "vehicle_assigned": "V1",
            "brand": "M",
            "license_plate": "123",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(DriverCreateSchema(), data)
        assert "password" in exc_info.value.messages.get("errors", {})


class TestClientUpdateSchema:
    """Tests pour ClientUpdateSchema."""

    def test_valid_update(self):
        """Test mise à jour client valide."""
        data = {
            "first_name": "Jane",
            "last_name": "Doe",
            "phone": "+33612345678",
            "birth_date": "1990-01-01",
            "gender": "FEMME",
        }
        result = validate_request(ClientUpdateSchema(), data, strict=False)
        assert result["first_name"] == "Jane"
        assert result["gender"] == "FEMME"

    def test_invalid_phone(self):
        """Test erreur si téléphone invalide."""
        data = {"phone": "123"}  # Trop court
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientUpdateSchema(), data, strict=False)
        assert "phone" in exc_info.value.messages.get("errors", {})

    def test_invalid_gender(self):
        """Test erreur si gender invalide."""
        data = {"gender": "invalid"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientUpdateSchema(), data, strict=False)
        assert "gender" in exc_info.value.messages.get("errors", {})

    def test_invalid_birth_date_format(self):
        """Test erreur si format date invalide."""
        data = {"birth_date": "01/01/1990"}  # Format invalide
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientUpdateSchema(), data, strict=False)
        assert "birth_date" in exc_info.value.messages.get("errors", {})


class TestDriverProfileUpdateSchema:
    """Tests pour DriverProfileUpdateSchema."""

    def test_valid_update(self):
        """Test mise à jour profil chauffeur valide."""
        data = {
            "first_name": "Jean",
            "last_name": "Dupont",
            "status": "disponible",
            "contract_type": "CDI",
            "weekly_hours": 40,
            "license_valid_until": "2025-12-31",
        }
        result = validate_request(DriverProfileUpdateSchema(), data, strict=False)
        assert result["first_name"] == "Jean"
        assert result["status"] == "disponible"

    def test_invalid_status(self):
        """Test erreur si status invalide."""
        data = {"status": "invalid"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(DriverProfileUpdateSchema(), data, strict=False)
        assert "status" in exc_info.value.messages.get("errors", {})

    def test_weekly_hours_too_high(self):
        """Test erreur si weekly_hours > 168."""
        data = {"weekly_hours": 200}  # > 168
        with pytest.raises(ValidationError) as exc_info:
            validate_request(DriverProfileUpdateSchema(), data, strict=False)
        assert "weekly_hours" in exc_info.value.messages.get("errors", {})

    def test_invalid_date_format(self):
        """Test erreur si format date invalide."""
        data = {"license_valid_until": "31/12/2025"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(DriverProfileUpdateSchema(), data, strict=False)
        assert "license_valid_until" in exc_info.value.messages.get("errors", {})
