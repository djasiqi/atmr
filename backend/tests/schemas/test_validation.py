"""✅ Tests unitaires pour validation des schémas Marshmallow.

Teste les schémas créés pour s'assurer qu'ils valident correctement les inputs
et rejettent les tentatives d'injection.
"""

import pytest
from marshmallow import ValidationError

from schemas.alert_schemas import ClearAlertHistorySchema
from schemas.company_schemas import DriverVacationCreateSchema, VehicleUpdateSchema
from schemas.dispatch_schemas import DispatchRunRequestSchema
from schemas.query_schemas import PaginationQuerySchema


class TestDispatchRunRequestSchema:
    """Tests pour DispatchRunRequestSchema."""

    def test_valid_request(self):
        """Test avec une requête valide."""
        schema = DispatchRunRequestSchema()
        data = {"for_date": "2025-01-15", "async": True, "regular_first": True}
        result = schema.load(data)
        assert result["for_date"] == "2025-01-15"
        assert result["async_mode"] is True
        assert result["regular_first"] is True

    def test_invalid_date_format(self):
        """Test avec format de date invalide."""
        schema = DispatchRunRequestSchema()
        data = {"for_date": "2025/01/15", "async": True}
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "for_date" in str(exc_info.value.messages)

    def test_invalid_mode(self):
        """Test avec mode invalide."""
        schema = DispatchRunRequestSchema()
        data = {"for_date": "2025-01-15", "mode": "invalid_mode"}
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "mode" in str(exc_info.value.messages)

    def test_sql_injection_attempt(self):
        """Test avec tentative d'injection SQL dans for_date."""
        schema = DispatchRunRequestSchema()
        data = {"for_date": "2025-01-15'; DROP TABLE users; --", "async": True}
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "for_date" in str(exc_info.value.messages)

    def test_xss_attempt(self):
        """Test avec tentative XSS dans overrides."""
        schema = DispatchRunRequestSchema()
        data = {
            "for_date": "2025-01-15",
            "overrides": {"heuristic": {"<script>alert('xss')</script>": "test"}},
        }
        # Le schéma devrait accepter (overrides est un Dict), mais la sanitisation devrait être appliquée ailleurs
        result = schema.load(data, strict=False)
        assert "overrides" in result


class TestVehicleUpdateSchema:
    """Tests pour VehicleUpdateSchema."""

    def test_valid_update(self):
        """Test avec mise à jour valide."""
        schema = VehicleUpdateSchema()
        data = {"brand": "Toyota", "model": "Corolla", "license_plate": "ABC-123"}
        result = schema.load(data)
        assert result["brand"] == "Toyota"
        assert result["model"] == "Corolla"
        assert result["license_plate"] == "ABC-123"

    def test_invalid_license_plate_too_long(self):
        """Test avec plaque d'immatriculation trop longue."""
        schema = VehicleUpdateSchema()
        data = {"license_plate": "A" * 25}  # > 20 caractères
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "license_plate" in str(exc_info.value.messages)

    def test_invalid_year_range(self):
        """Test avec année hors limites."""
        schema = VehicleUpdateSchema()
        data = {"year": 1899}  # < 1900
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "year" in str(exc_info.value.messages)

    def test_invalid_seats_range(self):
        """Test avec nombre de places hors limites."""
        schema = VehicleUpdateSchema()
        data = {"seats": 100}  # > 50
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "seats" in str(exc_info.value.messages)


class TestDriverVacationCreateSchema:
    """Tests pour DriverVacationCreateSchema."""

    def test_valid_vacation(self):
        """Test avec congé valide."""
        schema = DriverVacationCreateSchema()
        data = {"start_date": "2025-06-01", "end_date": "2025-06-15", "vacation_type": "VACANCES"}
        result = schema.load(data)
        assert result["start_date"] == "2025-06-01"
        assert result["end_date"] == "2025-06-15"
        assert result["vacation_type"] == "VACANCES"

    def test_invalid_date_format(self):
        """Test avec format de date invalide."""
        schema = DriverVacationCreateSchema()
        data = {"start_date": "2025/06/01", "end_date": "2025-06-15"}
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "start_date" in str(exc_info.value.messages)

    def test_invalid_vacation_type(self):
        """Test avec type de congé invalide."""
        schema = DriverVacationCreateSchema()
        data = {"start_date": "2025-06-01", "end_date": "2025-06-15", "vacation_type": "INVALID"}
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "vacation_type" in str(exc_info.value.messages)

    def test_missing_required_fields(self):
        """Test avec champs requis manquants."""
        schema = DriverVacationCreateSchema()
        data = {"start_date": "2025-06-01"}  # end_date manquant
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "end_date" in str(exc_info.value.messages)


class TestClearAlertHistorySchema:
    """Tests pour ClearAlertHistorySchema."""

    def test_valid_with_booking_id(self):
        """Test avec booking_id valide."""
        schema = ClearAlertHistorySchema()
        data = {"booking_id": "123"}
        result = schema.load(data)
        assert result["booking_id"] == "123"

    def test_valid_without_booking_id(self):
        """Test sans booking_id (optionnel)."""
        schema = ClearAlertHistorySchema()
        data = {}
        result = schema.load(data)
        assert result.get("booking_id") is None

    def test_booking_id_too_long(self):
        """Test avec booking_id trop long."""
        schema = ClearAlertHistorySchema()
        data = {"booking_id": "A" * 101}  # > 100 caractères
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "booking_id" in str(exc_info.value.messages)


class TestPaginationQuerySchema:
    """Tests pour PaginationQuerySchema."""

    def test_valid_pagination(self):
        """Test avec pagination valide."""
        schema = PaginationQuerySchema()
        data = {"page": 2, "per_page": 25}
        result = schema.load(data)
        assert result["page"] == 2
        assert result["per_page"] == 25

    def test_default_values(self):
        """Test avec valeurs par défaut."""
        schema = PaginationQuerySchema()
        data = {}
        result = schema.load(data)
        assert result["page"] == 1
        assert result["per_page"] == 50

    def test_invalid_page_negative(self):
        """Test avec page négative."""
        schema = PaginationQuerySchema()
        data = {"page": -1, "per_page": 50}
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "page" in str(exc_info.value.messages)

    def test_invalid_per_page_too_large(self):
        """Test avec per_page trop grand."""
        schema = PaginationQuerySchema()
        data = {"page": 1, "per_page": 600}  # > 500
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "per_page" in str(exc_info.value.messages)

    def test_invalid_per_page_zero(self):
        """Test avec per_page à zéro."""
        schema = PaginationQuerySchema()
        data = {"page": 1, "per_page": 0}
        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "per_page" in str(exc_info.value.messages)
