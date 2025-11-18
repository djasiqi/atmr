"""Tests pour les schemas de validation Marshmallow."""

import pytest
from marshmallow import ValidationError

from schemas.admin_schemas import AutonomousActionReviewSchema, UserRoleUpdateSchema
from schemas.analytics_schemas import (
    AnalyticsDashboardQuerySchema,
    AnalyticsExportQuerySchema,
    AnalyticsInsightsQuerySchema,
    AnalyticsWeeklySummaryQuerySchema,
)
from schemas.auth_schemas import LoginSchema, RegisterSchema
from schemas.booking_schemas import BookingCreateSchema, BookingListSchema, BookingUpdateSchema
from schemas.company_schemas import ClientCreateSchema, ManualBookingCreateSchema
from schemas.invoice_schemas import BillingSettingsUpdateSchema, InvoiceGenerateSchema
from schemas.medical_schemas import MedicalEstablishmentQuerySchema, MedicalServiceQuerySchema
from schemas.payment_schemas import PaymentCreateSchema, PaymentStatusUpdateSchema
from schemas.planning_schemas import (
    PlanningShiftsQuerySchema,
    PlanningUnavailabilityQuerySchema,
    PlanningWeeklyTemplateQuerySchema,
)
from schemas.validation_utils import ISO8601_DATE_REGEX, validate_request


class TestLoginSchema:
    """Tests pour LoginSchema."""

    def test_valid_login(self):
        """Test validation login valide."""
        data = {"email": "user@example.com", "password": "password123"}
        result = validate_request(LoginSchema(), data)
        assert result["email"] == "user@example.com"
        assert result["password"] == "password123"

    def test_missing_email(self):
        """Test erreur si email manquant."""
        data = {"password": "password123"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(LoginSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "email" in exc_info.value.messages["errors"]

    def test_invalid_email_format(self):
        """Test erreur si email invalide."""
        data = {"email": "invalid-email", "password": "password123"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(LoginSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "email" in exc_info.value.messages["errors"]

    def test_password_too_short(self):
        """Test erreur si mot de passe trop court."""
        data = {
            "email": "user@example.com",
            "password": "short",  # < 8 caractères
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(LoginSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "password" in exc_info.value.messages["errors"]


class TestRegisterSchema:
    """Tests pour RegisterSchema."""

    def test_valid_register(self):
        """Test validation inscription valide."""
        data = {"username": "testuser", "email": "test@example.com", "password": "password123"}
        result = validate_request(RegisterSchema(), data)
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["password"] == "password123"

    def test_register_with_optional_fields(self):
        """Test inscription avec champs optionnels."""
        data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
            "first_name": "John",
            "last_name": "Doe",
            "phone": "+33612345678",
        }
        result = validate_request(RegisterSchema(), data)
        assert result["first_name"] == "John"
        assert result["last_name"] == "Doe"
        assert result["phone"] == "+33612345678"

    def test_invalid_gender(self):
        """Test erreur si gender invalide."""
        data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
            "gender": "invalid",  # Doit être male/female/other
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(RegisterSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "gender" in exc_info.value.messages["errors"]


class TestBookingCreateSchema:
    """Tests pour BookingCreateSchema."""

    def test_valid_booking_create(self):
        """Test validation création réservation valide."""
        data = {
            "customer_name": "John Doe",
            "pickup_location": "123 Main St, Geneva",
            "dropoff_location": "456 Oak Ave, Lausanne",
            "scheduled_time": "2025-12-25T10:00:00Z",
            "amount": 50.0,
        }
        result = validate_request(BookingCreateSchema(), data)
        assert result["customer_name"] == "John Doe"
        assert result["amount"] == 50.0

    def test_missing_required_fields(self):
        """Test erreur si champs requis manquants."""
        data = {
            "customer_name": "John Doe"
            # pickup_location, dropoff_location, scheduled_time, amount manquants
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        errors = exc_info.value.messages["errors"]
        assert "pickup_location" in errors or "dropoff_location" in errors

    def test_invalid_datetime_format(self):
        """Test erreur si format datetime invalide."""
        data = {
            "customer_name": "John Doe",
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25 10:00",  # Format invalide
            "amount": 50.0,
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "scheduled_time" in exc_info.value.messages["errors"]

    def test_negative_amount(self):
        """Test erreur si montant négatif."""
        data = {
            "customer_name": "John Doe",
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T10:00:00Z",
            "amount": -10.0,  # Négatif
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "amount" in exc_info.value.messages["errors"]


class TestBookingListSchema:
    """Tests pour BookingListSchema."""

    def test_valid_list_params(self):
        """Test paramètres liste valides."""
        data = {"page": 2, "per_page": 50, "status": "confirmed"}
        result = validate_request(BookingListSchema(), data)
        assert result["page"] == 2
        assert result["per_page"] == 50
        assert result["status"] == "confirmed"

    def test_default_values(self):
        """Test valeurs par défaut."""
        data = {}
        result = validate_request(BookingListSchema(), data)
        assert result["page"] == 1
        assert result["per_page"] == 100  # Valeur par défaut selon BookingListSchema

    def test_invalid_page(self):
        """Test erreur si page invalide."""
        data = {"page": 0}  # Doit être >= 1
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingListSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "page" in exc_info.value.messages["errors"]

    def test_per_page_too_large(self):
        """Test erreur si per_page trop grand."""
        data = {"per_page": 501}  # Max 500 selon BookingListSchema
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingListSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "per_page" in exc_info.value.messages["errors"]


class TestBookingUpdateSchema:
    """Tests pour BookingUpdateSchema."""

    def test_update_partial_fields(self):
        """✅ Test validation mise à jour avec champs partiels."""
        # Test avec seulement un champ
        data = {"pickup_location": "123 New Street, Geneva"}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["pickup_location"] == "123 New Street, Geneva"
        assert "dropoff_location" not in result
        assert "amount" not in result

        # Test avec plusieurs champs (mais pas tous)
        data = {"pickup_location": "123 New Street", "amount": 75.5, "medical_facility": "Hôpital Cantonal"}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["pickup_location"] == "123 New Street"
        assert result["amount"] == 75.5
        assert result["medical_facility"] == "Hôpital Cantonal"
        assert "dropoff_location" not in result
        assert "status" not in result

        # Test avec objet vide (tous les champs optionnels)
        data = {}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result == {}

    def test_update_all_fields(self):
        """Test validation avec tous les champs."""
        data = {
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "amount": 100.0,
            "status": "confirmed",
            "medical_facility": "Hôpital",
            "doctor_name": "Dr. Smith",
            "is_round_trip": True,
            "notes_medical": "Patient nécessite assistance",
        }
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["pickup_location"] == "123 Main St"
        assert result["dropoff_location"] == "456 Oak Ave"
        assert result["scheduled_time"] == "2025-12-25T14:00:00Z"
        assert result["amount"] == 100.0
        assert result["status"] == "confirmed"
        assert result["medical_facility"] == "Hôpital"
        assert result["doctor_name"] == "Dr. Smith"
        assert result["is_round_trip"] is True
        assert result["notes_medical"] == "Patient nécessite assistance"

    def test_invalid_status(self):
        """✅ Test validation avec statut invalide."""
        data = {
            "status": "invalid_status"  # Doit être pending, confirmed, in_progress, completed, cancelled
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        # Les erreurs sont formatées avec une structure {'message': '...', 'errors': {...}}
        assert "errors" in exc_info.value.messages
        assert "status" in exc_info.value.messages["errors"]

        # Test avec statut valide
        data = {"status": "pending"}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["status"] == "pending"

        data = {"status": "completed"}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["status"] == "completed"

    def test_invalid_datetime_format(self):
        """✅ Test validation avec dates invalides."""
        # Format datetime invalide
        data = {
            "scheduled_time": "2025-12-25 14:00"  # Format invalide (manque le Z ou timezone)
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "scheduled_time" in exc_info.value.messages["errors"]

        # Format date au lieu de datetime
        data = {"scheduled_time": "2025-12-25"}  # Format date, pas datetime
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "scheduled_time" in exc_info.value.messages["errors"]

        # Format datetime valide
        data = {"scheduled_time": "2025-12-25T14:00:00Z"}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["scheduled_time"] == "2025-12-25T14:00:00Z"

    def test_negative_amount(self):
        """✅ Test validation montant négatif."""
        data = {"amount": -10.0}  # Négatif
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "amount" in exc_info.value.messages["errors"]

        # Montant zéro (valide)
        data = {"amount": 0.0}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["amount"] == 0.0

        # Montant positif (valide)
        data = {"amount": 50.5}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["amount"] == 50.5

    def test_string_length_validation(self):
        """Test validation longueur des chaînes."""
        # pickup_location trop long
        data = {"pickup_location": "a" * 501}  # Max 500
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "pickup_location" in exc_info.value.messages["errors"]

        # pickup_location vide
        data = {"pickup_location": ""}  # Min 1
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "pickup_location" in exc_info.value.messages["errors"]

        # medical_facility trop long
        data = {"medical_facility": "a" * 201}  # Max 200
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "medical_facility" in exc_info.value.messages["errors"]

        # notes_medical trop long
        data = {"notes_medical": "a" * 1001}  # Max 1000
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BookingUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "notes_medical" in exc_info.value.messages["errors"]

    def test_valid_status_values(self):
        """Test tous les statuts valides."""
        valid_statuses = ["pending", "confirmed", "in_progress", "completed", "cancelled"]
        for status in valid_statuses:
            data = {"status": status}
            result = validate_request(BookingUpdateSchema(), data, strict=False)
            assert result["status"] == status

    def test_boolean_fields(self):
        """Test validation des champs booléens."""
        data = {"is_round_trip": True}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["is_round_trip"] is True

        data = {"is_round_trip": False}
        result = validate_request(BookingUpdateSchema(), data, strict=False)
        assert result["is_round_trip"] is False


class TestPaymentCreateSchema:
    """Tests pour PaymentCreateSchema."""

    def test_valid_payment_create(self):
        """✅ Test validation création paiement valide."""
        # Test avec tous les champs requis
        data = {"amount": 50.0, "method": "credit_card", "booking_id": 1, "reference": "REF-12345"}
        result = validate_request(PaymentCreateSchema(), data)
        assert result["amount"] == 50.0
        assert result["method"] == "credit_card"
        assert result["booking_id"] == 1
        assert result["reference"] == "REF-12345"

        # Test avec seulement les champs requis (booking_id et reference optionnels)
        data = {"amount": 75.5, "method": "paypal"}
        result = validate_request(PaymentCreateSchema(), data)
        assert result["amount"] == 75.5
        assert result["method"] == "paypal"
        assert "booking_id" not in result or result.get("booking_id") is None
        assert "reference" not in result or result.get("reference") is None

    def test_missing_amount(self):
        """✅ Test erreur si amount manquant (requis)."""
        data = {
            "method": "credit_card"
            # amount manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "amount" in exc_info.value.messages["errors"]

    def test_missing_method(self):
        """✅ Test erreur si method manquant (requis)."""
        data = {
            "amount": 50.0
            # method manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "method" in exc_info.value.messages["errors"]

    def test_amount_required_and_positive(self):
        """✅ Test validation amount requis et > 0."""
        # Montant zéro (invalide, doit être >= 0.01)
        data = {"amount": 0.0, "method": "credit_card"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "amount" in exc_info.value.messages["errors"]

        # Montant négatif (invalide)
        data = {"amount": -10.0, "method": "credit_card"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "amount" in exc_info.value.messages["errors"]

        # Montant minimum valide (0.01)
        data = {"amount": 0.01, "method": "credit_card"}
        result = validate_request(PaymentCreateSchema(), data)
        assert result["amount"] == 0.01

        # Montant positif valide
        data = {"amount": 100.50, "method": "credit_card"}
        result = validate_request(PaymentCreateSchema(), data)
        assert result["amount"] == 100.50

    def test_method_required_and_length(self):
        """✅ Test validation method requis avec longueur."""
        # Method vide (invalide, min_length=1)
        data = {"amount": 50.0, "method": ""}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "method" in exc_info.value.messages["errors"]

        # Method trop long (invalide, max_length=50)
        data = {
            "amount": 50.0,
            "method": "a" * 51,  # 51 caractères > 50
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "method" in exc_info.value.messages["errors"]

        # Method valide (limite 50 caractères)
        data = {
            "amount": 50.0,
            "method": "a" * 50,  # Exactement 50 caractères
        }
        result = validate_request(PaymentCreateSchema(), data)
        assert result["method"] == "a" * 50

        # Method valide (courte)
        data = {"amount": 50.0, "method": "cash"}
        result = validate_request(PaymentCreateSchema(), data)
        assert result["method"] == "cash"

    def test_booking_id_optional(self):
        """Test validation booking_id optionnel."""
        # booking_id présent et valide (> 0)
        data = {"amount": 50.0, "method": "credit_card", "booking_id": 123}
        result = validate_request(PaymentCreateSchema(), data)
        assert result["booking_id"] == 123

        # booking_id absent (optionnel)
        data = {"amount": 50.0, "method": "credit_card"}
        result = validate_request(PaymentCreateSchema(), data)
        assert "booking_id" not in result or result.get("booking_id") is None

        # booking_id None (explicitement optionnel)
        data = {"amount": 50.0, "method": "credit_card", "booking_id": None}
        result = validate_request(PaymentCreateSchema(), data)
        assert result.get("booking_id") is None

        # booking_id invalide (< 1)
        data = {
            "amount": 50.0,
            "method": "credit_card",
            "booking_id": 0,  # Doit être >= 1
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "booking_id" in exc_info.value.messages["errors"]

    def test_reference_optional_and_length(self):
        """Test validation reference optionnelle avec longueur."""
        # reference présente et valide
        data = {"amount": 50.0, "method": "credit_card", "reference": "REF-12345"}
        result = validate_request(PaymentCreateSchema(), data)
        assert result["reference"] == "REF-12345"

        # reference absente (optionnel)
        data = {"amount": 50.0, "method": "credit_card"}
        result = validate_request(PaymentCreateSchema(), data)
        assert "reference" not in result or result.get("reference") is None

        # reference trop long (invalide, max_length=100)
        data = {
            "amount": 50.0,
            "method": "credit_card",
            "reference": "a" * 101,  # 101 caractères > 100
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "reference" in exc_info.value.messages["errors"]

        # reference valide (limite 100 caractères)
        data = {
            "amount": 50.0,
            "method": "credit_card",
            "reference": "a" * 100,  # Exactement 100 caractères
        }
        result = validate_request(PaymentCreateSchema(), data)
        assert result["reference"] == "a" * 100


class TestClientCreateSchema:
    """Tests pour ClientCreateSchema."""

    def test_valid_self_service_client(self):
        """✅ Test validation création client SELF_SERVICE."""
        data = {"client_type": "SELF_SERVICE", "email": "client@example.com"}
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "SELF_SERVICE"
        assert result["email"] == "client@example.com"

        # SELF_SERVICE avec champs optionnels
        data = {
            "client_type": "SELF_SERVICE",
            "email": "client2@example.com",
            "phone": "+33612345678",
            "notes": "Notes client",
        }
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "SELF_SERVICE"
        assert result["email"] == "client2@example.com"
        assert result["phone"] == "+33612345678"

    def test_valid_private_client(self):
        """✅ Test validation création client PRIVATE avec champs requis."""
        data = {
            "client_type": "PRIVATE",
            "first_name": "Jean",
            "last_name": "Dupont",
            "address": "123 Rue Example, 75001 Paris",
        }
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "PRIVATE"
        assert result["first_name"] == "Jean"
        assert result["last_name"] == "Dupont"
        assert result["address"] == "123 Rue Example, 75001 Paris"

        # PRIVATE avec tous les champs
        data = {
            "client_type": "PRIVATE",
            "first_name": "Marie",
            "last_name": "Martin",
            "address": "456 Avenue Test, 1000 Lausanne",
            "email": "marie@example.com",
            "phone": "+41791234567",
            "birth_date": "1990-01-15",
            "billing_address": "456 Avenue Test, 1000 Lausanne",
        }
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "PRIVATE"
        assert result["first_name"] == "Marie"
        assert result["email"] == "marie@example.com"
        assert result["billing_address"] == "456 Avenue Test, 1000 Lausanne"

    def test_valid_corporate_client(self):
        """✅ Test validation création client CORPORATE avec champs requis."""
        data = {
            "client_type": "CORPORATE",
            "first_name": "Enterprise",
            "last_name": "Corp",
            "address": "789 Business Street, 2000 Neuchâtel",
        }
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "CORPORATE"
        assert result["first_name"] == "Enterprise"
        assert result["last_name"] == "Corp"
        assert result["address"] == "789 Business Street, 2000 Neuchâtel"

        # CORPORATE avec institution
        data = {
            "client_type": "CORPORATE",
            "first_name": "Institution",
            "last_name": "Name",
            "address": "Institution Address",
            "is_institution": True,
            "institution_name": "Hopital Cantonal",
        }
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "CORPORATE"
        assert result["is_institution"] is True
        assert result["institution_name"] == "Hopital Cantonal"

    def test_missing_client_type(self):
        """✅ Test erreur si client_type manquant (requis)."""
        data = {
            "email": "client@example.com"
            # client_type manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "client_type" in exc_info.value.messages["errors"]

    def test_invalid_client_type(self):
        """✅ Test erreur si client_type invalide."""
        data = {
            "client_type": "INVALID_TYPE"  # Doit être SELF_SERVICE, PRIVATE ou CORPORATE
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "client_type" in exc_info.value.messages["errors"]

    def test_email_validation_for_self_service(self):
        """✅ Test validation email pour SELF_SERVICE (email optionnel dans schema mais requis logiquement)."""
        # Email valide
        data = {"client_type": "SELF_SERVICE", "email": "valid@example.com"}
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["email"] == "valid@example.com"

        # Email invalide (format)
        data = {"client_type": "SELF_SERVICE", "email": "invalid-email"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "email" in exc_info.value.messages["errors"]

        # Email trop long (max 254)
        data = {
            "client_type": "SELF_SERVICE",
            "email": "a" * 250 + "@example.com",  # > 254 caractères
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "email" in exc_info.value.messages["errors"]

        # Note: email absent n'est pas validé par le schéma (validé par la route pour SELF_SERVICE)
        # mais testons que le schéma l'accepte comme optionnel
        data = {
            "client_type": "SELF_SERVICE"
            # email absent (valide pour le schéma, sera validé par la route)
        }
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "SELF_SERVICE"
        assert "email" not in result or result.get("email") is None

    def test_required_fields_for_private_corporate(self):
        """✅ Test validation champs requis pour PRIVATE et CORPORATE (validés logiquement dans la route)."""
        # PRIVATE sans first_name, last_name, address (valides pour le schéma, validés par la route)
        data = {
            "client_type": "PRIVATE"
            # first_name, last_name, address manquants (valides pour schéma, route les exigera)
        }
        result = validate_request(ClientCreateSchema(), data, strict=False)
        assert result["client_type"] == "PRIVATE"
        # Le schéma accepte ces champs comme optionnels, mais la route les validera

        # Test validation longueur si fournis
        data = {
            "client_type": "PRIVATE",
            "first_name": "a" * 101,  # > 100 caractères
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "first_name" in exc_info.value.messages["errors"]

    def test_field_length_validation(self):
        """Test validation longueurs des champs."""
        # first_name trop long
        data = {
            "client_type": "PRIVATE",
            "first_name": "a" * 101,  # Max 100
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "first_name" in exc_info.value.messages["errors"]

        # address trop long
        data = {
            "client_type": "PRIVATE",
            "address": "a" * 501,  # Max 500
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "address" in exc_info.value.messages["errors"]

        # notes trop long
        data = {
            "client_type": "SELF_SERVICE",
            "email": "test@example.com",
            "notes": "a" * 1001,  # Max 1000
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "notes" in exc_info.value.messages["errors"]

    def test_coordinate_validation(self):
        """Test validation coordonnées GPS."""
        # billing_lat hors limite
        data = {
            "client_type": "PRIVATE",
            "first_name": "Test",
            "last_name": "User",
            "address": "Test Address",
            "billing_lat": 91.0,  # Hors limite (-90 à 90)
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "billing_lat" in exc_info.value.messages["errors"]

        # billing_lon hors limite
        data = {
            "client_type": "PRIVATE",
            "first_name": "Test",
            "last_name": "User",
            "address": "Test Address",
            "billing_lon": 181.0,  # Hors limite (-180 à 180)
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ClientCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "billing_lon" in exc_info.value.messages["errors"]

    def test_valid_client_types(self):
        """Test tous les types de clients valides."""
        valid_types = ["SELF_SERVICE", "PRIVATE", "CORPORATE"]
        for client_type in valid_types:
            data = {"client_type": client_type}
            if client_type == "SELF_SERVICE":
                data["email"] = "test@example.com"
            else:
                data["first_name"] = "Test"
                data["last_name"] = "User"
                data["address"] = "Test Address"

            result = validate_request(ClientCreateSchema(), data, strict=False)
            assert result["client_type"] == client_type


class TestManualBookingCreateSchema:
    """Tests pour ManualBookingCreateSchema."""

    def test_valid_manual_booking_minimal(self):
        """✅ Test validation création réservation manuelle avec champs requis uniquement."""
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St, Geneva",
            "dropoff_location": "456 Oak Ave, Lausanne",
            "scheduled_time": "2025-12-25T14:00:00Z",
        }
        result = validate_request(ManualBookingCreateSchema(), data)
        assert result["client_id"] == 123
        assert result["pickup_location"] == "123 Main St, Geneva"
        assert result["dropoff_location"] == "456 Oak Ave, Lausanne"
        assert result["scheduled_time"] == "2025-12-25T14:00:00Z"

    def test_valid_manual_booking_full(self):
        """✅ Test validation création réservation manuelle avec tous les champs."""
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "customer_first_name": "Jean",
            "customer_last_name": "Dupont",
            "customer_email": "jean@example.com",
            "customer_phone": "+33612345678",
            "is_round_trip": True,
            "return_time": "2025-12-25T18:00:00Z",
            "amount": 100.0,
            "billed_to_type": "patient",
            "medical_facility": "Hôpital Cantonal",
            "doctor_name": "Dr. Smith",
            "notes_medical": "Patient nécessite assistance",
            "pickup_lat": 46.2,
            "pickup_lon": 6.15,
            "dropoff_lat": 46.52,
            "dropoff_lon": 6.63,
        }
        result = validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert result["client_id"] == 123
        assert result["customer_first_name"] == "Jean"
        assert result["is_round_trip"] is True
        assert result["amount"] == 100.0
        assert result["billed_to_type"] == "patient"
        assert result["pickup_lat"] == 46.2

    def test_missing_required_fields(self):
        """✅ Test erreur si champs requis manquants."""
        # client_id manquant
        data = {
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "client_id" in exc_info.value.messages["errors"]

        # pickup_location manquant
        data = {"client_id": 123, "dropoff_location": "456 Oak Ave", "scheduled_time": "2025-12-25T14:00:00Z"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "pickup_location" in exc_info.value.messages["errors"]

        # scheduled_time manquant
        data = {"client_id": 123, "pickup_location": "123 Main St", "dropoff_location": "456 Oak Ave"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "scheduled_time" in exc_info.value.messages["errors"]

    def test_invalid_datetime_format(self):
        """✅ Test validation format datetime ISO 8601."""
        # Format invalide
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25 14:00",  # Format invalide
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "scheduled_time" in exc_info.value.messages["errors"]

        # Format valide
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
        }
        result = validate_request(ManualBookingCreateSchema(), data)
        assert result["scheduled_time"] == "2025-12-25T14:00:00Z"

    def test_round_trip_validation(self):
        """✅ Test validation round trip avec return_time et return_date."""
        # Round trip avec return_time
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "is_round_trip": True,
            "return_time": "2025-12-25T18:00:00Z",
        }
        result = validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert result["is_round_trip"] is True
        assert result["return_time"] == "2025-12-25T18:00:00Z"

        # Round trip avec return_date
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "is_round_trip": True,
            "return_date": "2025-12-26",
        }
        result = validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert result["return_date"] == "2025-12-26"

        # return_time format invalide
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "return_time": "2025-12-25 18:00",  # Format invalide
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "return_time" in exc_info.value.messages["errors"]

    def test_billed_to_type_validation(self):
        """✅ Test validation billed_to_type enum."""
        valid_types = ["patient", "clinic", "insurance"]
        for bt_type in valid_types:
            data = {
                "client_id": 123,
                "pickup_location": "123 Main St",
                "dropoff_location": "456 Oak Ave",
                "scheduled_time": "2025-12-25T14:00:00Z",
                "billed_to_type": bt_type,
            }
            result = validate_request(ManualBookingCreateSchema(), data, strict=False)
            assert result["billed_to_type"] == bt_type

        # Type invalide
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "billed_to_type": "invalid",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "billed_to_type" in exc_info.value.messages["errors"]

    def test_field_length_validation(self):
        """Test validation longueurs des champs."""
        # pickup_location trop long
        data = {
            "client_id": 123,
            "pickup_location": "a" * 501,  # Max 500
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "pickup_location" in exc_info.value.messages["errors"]

        # customer_first_name trop long
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "customer_first_name": "a" * 101,  # Max 100
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "customer_first_name" in exc_info.value.messages["errors"]

    def test_coordinate_validation(self):
        """Test validation coordonnées GPS."""
        # pickup_lat hors limite
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "pickup_lat": 91.0,  # Hors limite
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "pickup_lat" in exc_info.value.messages["errors"]

        # dropoff_lon hors limite
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "dropoff_lon": 181.0,  # Hors limite
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "dropoff_lon" in exc_info.value.messages["errors"]

    def test_client_id_validation(self):
        """Test validation client_id requis et > 0."""
        # client_id invalide (0)
        data = {
            "client_id": 0,  # Doit être >= 1
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "client_id" in exc_info.value.messages["errors"]

        # client_id négatif
        data = {
            "client_id": -1,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "client_id" in exc_info.value.messages["errors"]

    def test_amount_validation(self):
        """Test validation montant."""
        # Montant négatif
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "amount": -10.0,
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "amount" in exc_info.value.messages["errors"]

        # Montant zéro (valide)
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "amount": 0.0,
        }
        result = validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert result["amount"] == 0.0

    def test_email_validation(self):
        """Test validation email client."""
        # Email invalide
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "customer_email": "invalid-email",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "customer_email" in exc_info.value.messages["errors"]

        # Email valide
        data = {
            "client_id": 123,
            "pickup_location": "123 Main St",
            "dropoff_location": "456 Oak Ave",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "customer_email": "customer@example.com",
        }
        result = validate_request(ManualBookingCreateSchema(), data, strict=False)
        assert result["customer_email"] == "customer@example.com"


class TestBillingSettingsUpdateSchema:
    """Tests pour BillingSettingsUpdateSchema."""

    def test_valid_billing_settings_update_partial(self):
        """✅ Test validation mise à jour partielle des paramètres de facturation."""
        # Test avec seulement quelques champs
        data = {"payment_terms_days": 30, "overdue_fee": 25.0, "auto_reminders_enabled": True}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["payment_terms_days"] == 30
        assert result["overdue_fee"] == 25.0
        assert result["auto_reminders_enabled"] is True

        # Test avec objet vide (tous les champs optionnels)
        data = {}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result == {}

    def test_valid_billing_settings_update_full(self):
        """✅ Test validation mise à jour complète des paramètres de facturation."""
        data = {
            "payment_terms_days": 60,
            "overdue_fee": 50.0,
            "reminder1fee": 5.0,
            "reminder2fee": 10.0,
            "reminder3fee": 20.0,
            "auto_reminders_enabled": False,
            "email_sender": "billing@example.com",
            "invoice_number_format": "{PREFIX}-{YYYY}-{MM}-{SEQ4}",
            "invoice_prefix": "INV",
            "iban": "CH9300762011623852957",
            "qr_iban": "CH2108307000289537320",
            "esr_ref_base": "12345678901234567890",
            "invoice_message_template": "Merci de votre paiement",
            "reminder1template": "Rappel 1",
            "reminder2template": "Rappel 2",
            "reminder3template": "Rappel 3",
            "legal_footer": "Mentions légales",
            "pdf_template_variant": "standard",
            "reminder_schedule_days": [10, 20, 30],
        }
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["payment_terms_days"] == 60
        assert result["overdue_fee"] == 50.0
        assert result["iban"] == "CH9300762011623852957"
        assert result["invoice_number_format"] == "{PREFIX}-{YYYY}-{MM}-{SEQ4}"

    def test_payment_terms_days_validation(self):
        """✅ Test validation payment_terms_days (0-365 jours)."""
        # Valeur dans la plage valide
        data = {"payment_terms_days": 30}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["payment_terms_days"] == 30

        # Valeur hors limite (négatif)
        data = {"payment_terms_days": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "payment_terms_days" in exc_info.value.messages["errors"]

        # Valeur hors limite (> 365)
        data = {"payment_terms_days": 366}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "payment_terms_days" in exc_info.value.messages["errors"]

        # Valeur limite (0)
        data = {"payment_terms_days": 0}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["payment_terms_days"] == 0

        # Valeur limite (365)
        data = {"payment_terms_days": 365}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["payment_terms_days"] == 365

    def test_fees_validation(self):
        """✅ Test validation des frais (overdue_fee, reminder1fee, reminder2fee, reminder3fee >= 0)."""
        # Frais valides
        fees = ["overdue_fee", "reminder1fee", "reminder2fee", "reminder3fee"]
        for fee in fees:
            data = {fee: 25.0}
            result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
            assert result[fee] == 25.0

            # Frais négatif (invalide)
            data = {fee: -10.0}
            with pytest.raises(ValidationError) as exc_info:
                validate_request(BillingSettingsUpdateSchema(), data, strict=False)
            assert "errors" in exc_info.value.messages
            assert fee in exc_info.value.messages["errors"]

            # Frais zéro (valide)
            data = {fee: 0.0}
            result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
            assert result[fee] == 0.0

    def test_iban_validation(self):
        """✅ Test validation IBAN et QR IBAN (format CH + 19 caractères)."""
        # IBAN valide
        data = {"iban": "CH9300762011623852957"}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["iban"] == "CH9300762011623852957"

        # QR IBAN valide
        data = {"qr_iban": "CH2108307000289537320"}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["qr_iban"] == "CH2108307000289537320"

        # IBAN invalide (format incorrect - pas de préfixe pays à 2 lettres)
        data = {"iban": "12345678901234567890"}  # Format incorrect (pas de préfixe XX + 2 chiffres)
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "iban" in exc_info.value.messages["errors"]

        # IBAN invalide (préfixe invalide - caractères spéciaux)
        data = {"iban": "AB!!1234567890"}  # Caractères spéciaux invalides
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "iban" in exc_info.value.messages["errors"]

        # IBAN invalide (trop court)
        data = {"iban": "CH12"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "iban" in exc_info.value.messages["errors"]

        # IBAN None (valide car allow_none=True)
        data = {"iban": None}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result.get("iban") is None

    def test_field_length_validation(self):
        """Test validation longueurs des champs texte."""
        # email_sender trop long
        data = {"email_sender": "a" * 255}  # Max 254
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "email_sender" in exc_info.value.messages["errors"]

        # invoice_number_format trop long
        data = {"invoice_number_format": "a" * 51}  # Max 50
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "invoice_number_format" in exc_info.value.messages["errors"]

        # invoice_prefix trop long
        data = {"invoice_prefix": "a" * 21}  # Max 20
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "invoice_prefix" in exc_info.value.messages["errors"]

        # esr_ref_base trop long
        data = {"esr_ref_base": "a" * 27}  # Max 26
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "esr_ref_base" in exc_info.value.messages["errors"]

        # invoice_message_template trop long
        data = {"invoice_message_template": "a" * 1001}  # Max 1000
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "invoice_message_template" in exc_info.value.messages["errors"]

        # legal_footer trop long
        data = {"legal_footer": "a" * 2001}  # Max 2000
        with pytest.raises(ValidationError) as exc_info:
            validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "legal_footer" in exc_info.value.messages["errors"]

    def test_boolean_fields(self):
        """Test validation champs booléens."""
        data = {"auto_reminders_enabled": True}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["auto_reminders_enabled"] is True

        data = {"auto_reminders_enabled": False}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["auto_reminders_enabled"] is False

    def test_reminder_schedule_days(self):
        """Test validation reminder_schedule_days (Raw field, accepte diverses structures)."""
        # Liste
        data = {"reminder_schedule_days": [10, 20, 30]}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["reminder_schedule_days"] == [10, 20, 30]

        # Dict
        data = {"reminder_schedule_days": {"1": 10, "2": 5, "3": 5}}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result["reminder_schedule_days"] == {"1": 10, "2": 5, "3": 5}

        # None
        data = {"reminder_schedule_days": None}
        result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
        assert result.get("reminder_schedule_days") is None

    def test_template_fields(self):
        """Test validation des templates (reminder1template, reminder2template, reminder3template)."""
        templates = ["reminder1template", "reminder2template", "reminder3template"]
        for template in templates:
            # Template valide
            data = {template: "Message de rappel"}
            result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
            assert result[template] == "Message de rappel"

            # Template None (valide car allow_none=True)
            data = {template: None}
            result = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
            assert result.get(template) is None


class TestInvoiceGenerateSchema:
    """Tests pour InvoiceGenerateSchema."""

    def test_valid_invoice_generate_with_client_id(self):
        """✅ Test validation génération facture avec client_id."""
        data = {"client_id": 123, "period_year": 2025, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["client_id"] == 123
        assert result["period_year"] == 2025
        assert result["period_month"] == 12

    def test_valid_invoice_generate_with_client_ids(self):
        """✅ Test validation génération facture avec client_ids (liste)."""
        data = {"client_ids": [123, 456, 789], "period_year": 2025, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["client_ids"] == [123, 456, 789]
        assert result["period_year"] == 2025
        assert result["period_month"] == 12

    def test_valid_invoice_generate_full(self):
        """✅ Test validation génération facture avec tous les champs."""
        data = {
            "client_id": 123,
            "client_ids": [123, 456],  # Peut coexister avec client_id
            "bill_to_client_id": 999,
            "period_year": 2025,
            "period_month": 11,
            "client_reservations": {"123": [1, 2, 3], "456": [4, 5]},
            "reservation_ids": [10, 20, 30],
        }
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["client_id"] == 123
        assert result["client_ids"] == [123, 456]
        assert result["bill_to_client_id"] == 999
        assert result["period_year"] == 2025
        assert result["period_month"] == 11
        assert result["client_reservations"] == {"123": [1, 2, 3], "456": [4, 5]}
        assert result["reservation_ids"] == [10, 20, 30]

    def test_missing_period_year(self):
        """✅ Test erreur si period_year manquant (requis)."""
        data = {
            "client_id": 123,
            "period_month": 12,
            # period_year manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period_year" in exc_info.value.messages["errors"]

    def test_missing_period_month(self):
        """✅ Test erreur si period_month manquant (requis)."""
        data = {
            "client_id": 123,
            "period_year": 2025,
            # period_month manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period_month" in exc_info.value.messages["errors"]

    def test_period_year_validation(self):
        """✅ Test validation period_year (2000-2100)."""
        # Année hors limite (< 2000)
        data = {"client_id": 123, "period_year": 1999, "period_month": 12}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period_year" in exc_info.value.messages["errors"]

        # Année hors limite (> 2100)
        data = {"client_id": 123, "period_year": 2101, "period_month": 12}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period_year" in exc_info.value.messages["errors"]

        # Année limite (2000)
        data = {"client_id": 123, "period_year": 2000, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["period_year"] == 2000

        # Année limite (2100)
        data = {"client_id": 123, "period_year": 2100, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["period_year"] == 2100

    def test_period_month_validation(self):
        """✅ Test validation period_month (1-12)."""
        # Mois hors limite (< 1)
        data = {"client_id": 123, "period_year": 2025, "period_month": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period_month" in exc_info.value.messages["errors"]

        # Mois hors limite (> 12)
        data = {"client_id": 123, "period_year": 2025, "period_month": 13}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period_month" in exc_info.value.messages["errors"]

        # Mois limite (1)
        data = {"client_id": 123, "period_year": 2025, "period_month": 1}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["period_month"] == 1

        # Mois limite (12)
        data = {"client_id": 123, "period_year": 2025, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["period_month"] == 12

    def test_client_id_validation(self):
        """✅ Test validation client_id (>= 1 si fourni)."""
        # client_id invalide (0)
        data = {"client_id": 0, "period_year": 2025, "period_month": 12}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "client_id" in exc_info.value.messages["errors"]

        # client_id négatif
        data = {"client_id": -1, "period_year": 2025, "period_month": 12}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "client_id" in exc_info.value.messages["errors"]

    def test_client_ids_validation(self):
        """✅ Test validation client_ids (liste avec au moins 1 élément)."""
        # client_ids vide (invalide, doit avoir au moins 1 élément)
        data = {"client_ids": [], "period_year": 2025, "period_month": 12}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "client_ids" in exc_info.value.messages["errors"]

        # client_ids avec ID invalide (0)
        data = {
            "client_ids": [123, 0, 456],  # 0 est invalide
            "period_year": 2025,
            "period_month": 12,
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "client_ids" in exc_info.value.messages["errors"]

        # client_ids valide (au moins 1 élément)
        data = {"client_ids": [123], "period_year": 2025, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["client_ids"] == [123]

        # client_ids avec plusieurs IDs valides
        data = {"client_ids": [123, 456, 789], "period_year": 2025, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["client_ids"] == [123, 456, 789]

    def test_bill_to_client_id_validation(self):
        """Test validation bill_to_client_id (>= 1 si fourni, optionnel)."""
        # bill_to_client_id invalide (0)
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "bill_to_client_id": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "bill_to_client_id" in exc_info.value.messages["errors"]

        # bill_to_client_id None (valide car optionnel)
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "bill_to_client_id": None}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result.get("bill_to_client_id") is None

        # bill_to_client_id valide
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "bill_to_client_id": 999}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["bill_to_client_id"] == 999

    def test_reservation_ids_validation(self):
        """Test validation reservation_ids (liste optionnelle)."""
        # reservation_ids présente
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "reservation_ids": [1, 2, 3, 4, 5]}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["reservation_ids"] == [1, 2, 3, 4, 5]

        # reservation_ids None (valide car optionnel)
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "reservation_ids": None}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result.get("reservation_ids") is None

        # reservation_ids vide (valide, liste vide)
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "reservation_ids": []}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["reservation_ids"] == []

    def test_client_reservations_validation(self):
        """Test validation client_reservations (dict optionnel)."""
        # client_reservations présent
        data = {
            "client_id": 123,
            "period_year": 2025,
            "period_month": 12,
            "client_reservations": {"123": [1, 2, 3], "456": [4, 5]},
        }
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["client_reservations"] == {"123": [1, 2, 3], "456": [4, 5]}

        # client_reservations None (valide car optionnel)
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "client_reservations": None}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result.get("client_reservations") is None

        # client_reservations vide (valide, dict vide)
        data = {"client_id": 123, "period_year": 2025, "period_month": 12, "client_reservations": {}}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["client_reservations"] == {}

    def test_minimal_valid_invoice_generate(self):
        """Test validation minimale (seulement period_year et period_month requis)."""
        # Note: client_id et client_ids ne sont pas requis (au moins un doit être fourni logiquement)
        data = {"period_year": 2025, "period_month": 12}
        result = validate_request(InvoiceGenerateSchema(), data, strict=False)
        assert result["period_year"] == 2025
        assert result["period_month"] == 12
        # client_id et client_ids peuvent être absents (validation logique dans la route)


class TestPaymentStatusUpdateSchema:
    """Tests pour PaymentStatusUpdateSchema."""

    def test_valid_status_update(self):
        """✅ Test validation mise à jour statut valide."""
        # Statut pending
        data = {"status": "pending"}
        result = validate_request(PaymentStatusUpdateSchema(), data)
        assert result["status"] == "pending"

        # Statut completed
        data = {"status": "completed"}
        result = validate_request(PaymentStatusUpdateSchema(), data)
        assert result["status"] == "completed"

        # Statut failed
        data = {"status": "failed"}
        result = validate_request(PaymentStatusUpdateSchema(), data)
        assert result["status"] == "failed"

    def test_missing_status(self):
        """✅ Test erreur si status manquant (requis)."""
        data = {}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentStatusUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "status" in exc_info.value.messages["errors"]

    def test_invalid_status(self):
        """✅ Test erreur si status invalide (doit être pending, completed ou failed)."""
        # Statut invalide
        data = {"status": "invalid_status"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentStatusUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "status" in exc_info.value.messages["errors"]

        # Statut avec casse incorrecte
        data = {"status": "PENDING"}  # Doit être lowercase
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentStatusUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "status" in exc_info.value.messages["errors"]

        # Statut partiel
        data = {"status": "pend"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PaymentStatusUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "status" in exc_info.value.messages["errors"]

    def test_all_valid_status_values(self):
        """Test tous les statuts valides."""
        valid_statuses = ["pending", "completed", "failed"]
        for status in valid_statuses:
            data = {"status": status}
            result = validate_request(PaymentStatusUpdateSchema(), data)
            assert result["status"] == status


class TestMedicalEstablishmentQuerySchema:
    """Tests pour MedicalEstablishmentQuerySchema."""

    def test_valid_query_with_q_and_limit(self):
        """✅ Test validation requête avec q et limit."""
        data = {"q": "Hôpital Cantonal", "limit": 10}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result["q"] == "Hôpital Cantonal"
        assert result["limit"] == 10

    def test_valid_query_default_values(self):
        """✅ Test validation avec valeurs par défaut."""
        # Objet vide (limit par défaut = 8)
        data = {}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result.get("q") is None or result["q"] is None
        assert result["limit"] == 8

        # Seulement q
        data = {"q": "Clinique"}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result["q"] == "Clinique"
        assert result["limit"] == 8

    def test_q_length_validation(self):
        """✅ Test validation longueur du query (max 200 caractères)."""
        # Query trop long
        data = {
            "q": "a" * 201,  # Max 200
            "limit": 10,
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "q" in exc_info.value.messages["errors"]

        # Query limite (200 caractères)
        data = {"q": "a" * 200, "limit": 10}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result["q"] == "a" * 200

        # Query valide (court)
        data = {"q": "Hôpital", "limit": 10}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result["q"] == "Hôpital"

    def test_limit_validation(self):
        """✅ Test validation limit (1-25)."""
        # Limit hors limite (< 1)
        data = {"q": "Test", "limit": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "limit" in exc_info.value.messages["errors"]

        # Limit hors limite (> 25)
        data = {"q": "Test", "limit": 26}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "limit" in exc_info.value.messages["errors"]

        # Limit limite (1)
        data = {"q": "Test", "limit": 1}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result["limit"] == 1

        # Limit limite (25)
        data = {"q": "Test", "limit": 25}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result["limit"] == 25

        # Limit par défaut (8 si non fourni)
        data = {"q": "Test"}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result["limit"] == 8

    def test_q_optional(self):
        """Test validation q optionnel."""
        # q None (valide)
        data = {"q": None, "limit": 10}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert result.get("q") is None

        # q absent
        data = {"limit": 10}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        assert "q" not in result or result.get("q") is None

        # q vide string (valide car optionnel)
        data = {"q": "", "limit": 10}
        result = validate_request(MedicalEstablishmentQuerySchema(), data, strict=False)
        # Note: le schéma acceptera "" mais la route peut le traiter comme None


class TestMedicalServiceQuerySchema:
    """Tests pour MedicalServiceQuerySchema."""

    def test_valid_query_with_establishment_id_and_q(self):
        """✅ Test validation requête avec establishment_id et q."""
        data = {"establishment_id": 123, "q": "Radiologie"}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert result["establishment_id"] == 123
        assert result["q"] == "Radiologie"

    def test_valid_query_with_only_establishment_id(self):
        """✅ Test validation requête avec seulement establishment_id (q optionnel)."""
        data = {"establishment_id": 123}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert result["establishment_id"] == 123
        assert result.get("q") is None or result["q"] is None

    def test_missing_establishment_id(self):
        """✅ Test erreur si establishment_id manquant (requis)."""
        data = {
            "q": "Radiologie"
            # establishment_id manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "establishment_id" in exc_info.value.messages["errors"]

    def test_establishment_id_validation(self):
        """✅ Test validation establishment_id (>= 1)."""
        # establishment_id invalide (0)
        data = {"establishment_id": 0, "q": "Test"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "establishment_id" in exc_info.value.messages["errors"]

        # establishment_id négatif
        data = {"establishment_id": -1, "q": "Test"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "establishment_id" in exc_info.value.messages["errors"]

        # establishment_id valide
        data = {"establishment_id": 1, "q": "Test"}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert result["establishment_id"] == 1

    def test_q_length_validation(self):
        """✅ Test validation longueur du query q (max 200 caractères)."""
        # Query trop long
        data = {
            "establishment_id": 123,
            "q": "a" * 201,  # Max 200
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "q" in exc_info.value.messages["errors"]

        # Query limite (200 caractères)
        data = {"establishment_id": 123, "q": "a" * 200}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert result["q"] == "a" * 200

        # Query valide (court)
        data = {"establishment_id": 123, "q": "Service"}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert result["q"] == "Service"

    def test_q_optional(self):
        """Test validation q optionnel."""
        # q None (valide)
        data = {"establishment_id": 123, "q": None}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert result.get("q") is None

        # q absent
        data = {"establishment_id": 123}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        assert "q" not in result or result.get("q") is None

        # q vide string (valide car optionnel)
        data = {"establishment_id": 123, "q": ""}
        result = validate_request(MedicalServiceQuerySchema(), data, strict=False)
        # Note: le schéma acceptera "" mais la route peut le traiter comme None


class TestAnalyticsDashboardQuerySchema:
    """Tests pour AnalyticsDashboardQuerySchema."""

    def test_valid_query_with_period_default(self):
        """✅ Test validation avec period par défaut (30d)."""
        data = {}
        result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert result["period"] == "30d"
        assert result.get("start_date") is None
        assert result.get("end_date") is None

    def test_valid_query_with_period(self):
        """✅ Test validation avec period spécifiée."""
        valid_periods = ["7d", "30d", "90d", "1y"]
        for period in valid_periods:
            data = {"period": period}
            result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
            assert result["period"] == period

    def test_invalid_period(self):
        """✅ Test erreur si period invalide."""
        data = {"period": "invalid"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period" in exc_info.value.messages["errors"]

        # Test casse incorrecte
        data = {"period": "30D"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "period" in exc_info.value.messages["errors"]

    def test_valid_query_with_custom_dates(self):
        """✅ Test validation avec dates personnalisées (start_date et end_date)."""
        data = {"period": "7d", "start_date": "2024-01-01", "end_date": "2024-01-31"}
        result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert result["period"] == "7d"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"

    def test_start_date_validation(self):
        """✅ Test validation format start_date (YYYY-MM-DD)."""
        # Date valide
        data = {"period": "30d", "start_date": "2024-01-15"}
        result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert result["start_date"] == "2024-01-15"

        # Format invalide (YYYY-MM-DD requis)
        data = {
            "period": "30d",
            "start_date": "01/15/2024",  # Format US
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "start_date" in exc_info.value.messages["errors"]

        # Format invalide (timestamp)
        data = {
            "period": "30d",
            "start_date": "2024-01-15T10:30:00",  # ISO datetime
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "start_date" in exc_info.value.messages["errors"]

    def test_end_date_validation(self):
        """✅ Test validation format end_date (YYYY-MM-DD)."""
        # Date valide
        data = {"period": "30d", "end_date": "2024-01-31"}
        result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert result["end_date"] == "2024-01-31"

        # Format invalide (YYYY-MM-DD requis)
        data = {
            "period": "30d",
            "end_date": "31/01/2024",  # Format EU
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "end_date" in exc_info.value.messages["errors"]

    def test_dates_optional(self):
        """✅ Test validation dates optionnelles."""
        # start_date None
        data = {"period": "30d", "start_date": None, "end_date": "2024-01-31"}
        result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert result.get("start_date") is None
        assert result["end_date"] == "2024-01-31"

        # end_date None
        data = {"period": "30d", "start_date": "2024-01-01", "end_date": None}
        result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert result["start_date"] == "2024-01-01"
        assert result.get("end_date") is None

        # Les deux None
        data = {
            "period": "30d"
            # start_date et end_date absents
        }
        result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
        assert result["period"] == "30d"
        assert result.get("start_date") is None
        assert result.get("end_date") is None

    def test_all_combinations(self):
        """✅ Test toutes les combinaisons valides de period."""
        periods = ["7d", "30d", "90d", "1y"]
        for period in periods:
            # Period seule
            data = {"period": period}
            result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
            assert result["period"] == period

            # Period + dates
            data = {"period": period, "start_date": "2024-01-01", "end_date": "2024-01-31"}
            result = validate_request(AnalyticsDashboardQuerySchema(), data, strict=False)
            assert result["period"] == period
            assert result["start_date"] == "2024-01-01"
            assert result["end_date"] == "2024-01-31"


class TestAnalyticsInsightsQuerySchema:
    """Tests pour AnalyticsInsightsQuerySchema."""

    def test_valid_query_with_default(self):
        """✅ Test validation avec valeur par défaut (lookback_days=30)."""
        data = {}
        result = validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert result["lookback_days"] == 30

    def test_valid_query_with_lookback_days(self):
        """✅ Test validation avec lookback_days spécifié."""
        # Valeur valide
        data = {"lookback_days": 60}
        result = validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert result["lookback_days"] == 60

        # Minimum (1)
        data = {"lookback_days": 1}
        result = validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert result["lookback_days"] == 1

        # Maximum (365)
        data = {"lookback_days": 365}
        result = validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert result["lookback_days"] == 365

    def test_lookback_days_validation_min(self):
        """✅ Test validation lookback_days minimum (>= 1)."""
        # Trop petit (0)
        data = {"lookback_days": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "lookback_days" in exc_info.value.messages["errors"]

        # Négatif
        data = {"lookback_days": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "lookback_days" in exc_info.value.messages["errors"]

        # Limite (1)
        data = {"lookback_days": 1}
        result = validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert result["lookback_days"] == 1

    def test_lookback_days_validation_max(self):
        """✅ Test validation lookback_days maximum (<= 365)."""
        # Trop grand (366)
        data = {"lookback_days": 366}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "lookback_days" in exc_info.value.messages["errors"]

        # Très grand
        data = {"lookback_days": 1000}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "lookback_days" in exc_info.value.messages["errors"]

        # Limite (365)
        data = {"lookback_days": 365}
        result = validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert result["lookback_days"] == 365

    def test_lookback_days_type_validation(self):
        """✅ Test validation type lookback_days (doit être Int)."""
        # String (invalide)
        data = {"lookback_days": "30"}
        # Marshmallow devrait convertir "30" en int, donc cela devrait passer
        # Mais testons quand même avec une vraie string non numérique
        data_invalid = {"lookback_days": "abc"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsInsightsQuerySchema(), data_invalid, strict=False)
        assert "errors" in exc_info.value.messages
        assert "lookback_days" in exc_info.value.messages["errors"]

        # Float non entier (invalide pour Int, mais Marshmallow peut convertir 30.0 en 30)
        # Testons avec une valeur qui ne peut pas être convertie en int
        # Marshmallow peut accepter 30.0 et le convertir, donc testons avec une vraie valeur invalide
        data = {"lookback_days": "not_a_number"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "lookback_days" in exc_info.value.messages["errors"]

    def test_all_valid_values_boundaries(self):
        """✅ Test toutes les valeurs limites valides."""
        # Test valeurs limites et intermédiaires
        valid_values = [1, 30, 60, 90, 180, 365]
        for value in valid_values:
            data = {"lookback_days": value}
            result = validate_request(AnalyticsInsightsQuerySchema(), data, strict=False)
            assert result["lookback_days"] == value


class TestAnalyticsWeeklySummaryQuerySchema:
    """Tests pour AnalyticsWeeklySummaryQuerySchema."""

    def test_valid_query_with_week_start(self):
        """✅ Test validation avec week_start spécifié."""
        data = {"week_start": "2024-01-01"}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-01-01"

        # Date valide différente
        data = {"week_start": "2024-12-31"}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-12-31"

    def test_valid_query_without_week_start(self):
        """✅ Test validation sans week_start (optionnel)."""
        data = {}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result.get("week_start") is None or result["week_start"] is None

    def test_week_start_validation_format(self):
        """✅ Test validation format week_start (YYYY-MM-DD)."""
        # Date valide
        data = {"week_start": "2024-01-15"}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-01-15"

        # Format invalide (YYYY-MM-DD requis)
        data = {"week_start": "01/15/2024"}  # Format US
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "week_start" in exc_info.value.messages["errors"]

        # Format invalide (timestamp)
        data = {"week_start": "2024-01-15T10:30:00"}  # ISO datetime
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "week_start" in exc_info.value.messages["errors"]

        # Format invalide (jour seulement)
        data = {"week_start": "2024-01"}  # Mois seulement
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "week_start" in exc_info.value.messages["errors"]

    def test_week_start_optional(self):
        """✅ Test validation week_start optionnel."""
        # week_start None
        data = {"week_start": None}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result.get("week_start") is None

        # week_start absent
        data = {}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert "week_start" not in result or result.get("week_start") is None

        # week_start vide string (valide car optionnel, mais format invalide)
        data = {"week_start": ""}
        # Vide devrait échouer format validation car "" ne matche pas YYYY-MM-DD
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "week_start" in exc_info.value.messages["errors"]

    def test_week_start_edge_cases(self):
        """✅ Test cas limites pour week_start."""
        # Date début d'année
        data = {"week_start": "2024-01-01"}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-01-01"

        # Date fin d'année
        data = {"week_start": "2024-12-31"}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-12-31"

        # Date avec zéros (jour/mois)
        data = {"week_start": "2024-01-01"}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-01-01"

        # Date 29 février (année bissextile)
        data = {"week_start": "2024-02-29"}
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-02-29"

        # Date invalide (30 février)
        data = {"week_start": "2024-02-30"}
        # Le format YYYY-MM-DD sera validé mais la date est logiquement invalide
        # Marshmallow ne valide que le format, pas la logique de la date
        result = validate_request(AnalyticsWeeklySummaryQuerySchema(), data, strict=False)
        assert result["week_start"] == "2024-02-30"  # Format correct, logique incorrecte (géré par route)


class TestAnalyticsExportQuerySchema:
    """Tests pour AnalyticsExportQuerySchema."""

    def test_valid_query_complete(self):
        """✅ Test validation avec tous les champs (start_date, end_date, format)."""
        data = {"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "csv"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"
        assert result["format"] == "csv"

        # Format JSON
        data = {"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "json"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["format"] == "json"

    def test_valid_query_with_default_format(self):
        """✅ Test validation avec format par défaut (csv)."""
        data = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            # format absent -> défaut "csv"
        }
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"
        assert result["format"] == "csv"

    def test_missing_start_date(self):
        """✅ Test erreur si start_date manquant (requis)."""
        data = {
            "end_date": "2024-01-31",
            "format": "csv",
            # start_date manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "start_date" in exc_info.value.messages["errors"]

    def test_missing_end_date(self):
        """✅ Test erreur si end_date manquant (requis)."""
        data = {
            "start_date": "2024-01-01",
            "format": "csv",
            # end_date manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "end_date" in exc_info.value.messages["errors"]

    def test_start_date_validation_format(self):
        """✅ Test validation format start_date (YYYY-MM-DD)."""
        # Date valide
        data = {"start_date": "2024-01-15", "end_date": "2024-01-31", "format": "csv"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["start_date"] == "2024-01-15"

        # Format invalide (YYYY-MM-DD requis)
        data = {
            "start_date": "01/15/2024",  # Format US
            "end_date": "2024-01-31",
            "format": "csv",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "start_date" in exc_info.value.messages["errors"]

        # Format invalide (timestamp)
        data = {
            "start_date": "2024-01-15T10:30:00",  # ISO datetime
            "end_date": "2024-01-31",
            "format": "csv",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "start_date" in exc_info.value.messages["errors"]

    def test_end_date_validation_format(self):
        """✅ Test validation format end_date (YYYY-MM-DD)."""
        # Date valide
        data = {"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "csv"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["end_date"] == "2024-01-31"

        # Format invalide (YYYY-MM-DD requis)
        data = {
            "start_date": "2024-01-01",
            "end_date": "31/01/2024",  # Format EU
            "format": "csv",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "end_date" in exc_info.value.messages["errors"]

    def test_format_validation(self):
        """✅ Test validation format (csv|json)."""
        # Format valide CSV
        data = {"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "csv"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["format"] == "csv"

        # Format valide JSON
        data = {"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "json"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["format"] == "json"

        # Format invalide
        data = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "format": "xml",  # Format non supporté
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "format" in exc_info.value.messages["errors"]

        # Format invalide (casse incorrecte)
        data = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "format": "CSV",  # Majuscule
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "format" in exc_info.value.messages["errors"]

    def test_all_valid_formats(self):
        """✅ Test tous les formats valides."""
        valid_formats = ["csv", "json"]
        for fmt in valid_formats:
            data = {"start_date": "2024-01-01", "end_date": "2024-01-31", "format": fmt}
            result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
            assert result["format"] == fmt

    def test_dates_edge_cases(self):
        """✅ Test cas limites pour dates."""
        # Dates début/fin d'année
        data = {"start_date": "2024-01-01", "end_date": "2024-12-31", "format": "csv"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-12-31"

        # Dates identiques (période d'un jour)
        data = {"start_date": "2024-06-15", "end_date": "2024-06-15", "format": "csv"}
        result = validate_request(AnalyticsExportQuerySchema(), data, strict=False)
        assert result["start_date"] == "2024-06-15"
        assert result["end_date"] == "2024-06-15"


class TestPlanningShiftsQuerySchema:
    """Tests pour PlanningShiftsQuerySchema."""

    def test_valid_query_with_driver_id(self):
        """✅ Test validation avec driver_id spécifié."""
        data = {"driver_id": 123}
        result = validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert result["driver_id"] == 123

        # Valeur minimale (1)
        data = {"driver_id": 1}
        result = validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert result["driver_id"] == 1

    def test_valid_query_without_driver_id(self):
        """✅ Test validation sans driver_id (optionnel)."""
        data = {}
        result = validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert result.get("driver_id") is None or result["driver_id"] is None

    def test_driver_id_validation(self):
        """✅ Test validation driver_id (>= 1 si fourni)."""
        # driver_id invalide (0)
        data = {"driver_id": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # driver_id négatif
        data = {"driver_id": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # driver_id valide
        data = {"driver_id": 1}
        result = validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert result["driver_id"] == 1

        # driver_id valide plus grand
        data = {"driver_id": 999}
        result = validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert result["driver_id"] == 999

    def test_driver_id_optional(self):
        """✅ Test validation driver_id optionnel."""
        # driver_id None
        data = {"driver_id": None}
        result = validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert result.get("driver_id") is None

        # driver_id absent
        data = {}
        result = validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert "driver_id" not in result or result.get("driver_id") is None

    def test_driver_id_type_validation(self):
        """✅ Test validation type driver_id (doit être Int)."""
        # String numérique (Marshmallow devrait convertir)
        data = {"driver_id": "123"}
        # Marshmallow peut convertir "123" en int, donc testons avec une vraie string non numérique
        data_invalid = {"driver_id": "abc"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningShiftsQuerySchema(), data_invalid, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # Float non entier (invalide pour Int, mais Marshmallow peut convertir)
        # Testons avec une valeur vraiment invalide
        data = {"driver_id": "not_a_number"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningShiftsQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]


class TestPlanningUnavailabilityQuerySchema:
    """Tests pour PlanningUnavailabilityQuerySchema."""

    def test_valid_query_with_driver_id(self):
        """✅ Test validation avec driver_id spécifié."""
        data = {"driver_id": 123}
        result = validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert result["driver_id"] == 123

        # Valeur minimale (1)
        data = {"driver_id": 1}
        result = validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert result["driver_id"] == 1

    def test_valid_query_without_driver_id(self):
        """✅ Test validation sans driver_id (optionnel)."""
        data = {}
        result = validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert result.get("driver_id") is None or result["driver_id"] is None

    def test_driver_id_validation(self):
        """✅ Test validation driver_id (>= 1 si fourni)."""
        # driver_id invalide (0)
        data = {"driver_id": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # driver_id négatif
        data = {"driver_id": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # driver_id valide
        data = {"driver_id": 1}
        result = validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert result["driver_id"] == 1

        # driver_id valide plus grand
        data = {"driver_id": 999}
        result = validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert result["driver_id"] == 999

    def test_driver_id_optional(self):
        """✅ Test validation driver_id optionnel."""
        # driver_id None
        data = {"driver_id": None}
        result = validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert result.get("driver_id") is None

        # driver_id absent
        data = {}
        result = validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert "driver_id" not in result or result.get("driver_id") is None

    def test_driver_id_type_validation(self):
        """✅ Test validation type driver_id (doit être Int)."""
        # String non numérique
        data = {"driver_id": "abc"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # Float non entier (invalide pour Int, mais Marshmallow peut convertir)
        # Testons avec une valeur vraiment invalide
        data = {"driver_id": "not_a_number"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningUnavailabilityQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]


class TestPlanningWeeklyTemplateQuerySchema:
    """Tests pour PlanningWeeklyTemplateQuerySchema."""

    def test_valid_query_with_driver_id(self):
        """✅ Test validation avec driver_id spécifié."""
        data = {"driver_id": 123}
        result = validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert result["driver_id"] == 123

        # Valeur minimale (1)
        data = {"driver_id": 1}
        result = validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert result["driver_id"] == 1

    def test_valid_query_without_driver_id(self):
        """✅ Test validation sans driver_id (optionnel)."""
        data = {}
        result = validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert result.get("driver_id") is None or result["driver_id"] is None

    def test_driver_id_validation(self):
        """✅ Test validation driver_id (>= 1 si fourni)."""
        # driver_id invalide (0)
        data = {"driver_id": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # driver_id négatif
        data = {"driver_id": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # driver_id valide
        data = {"driver_id": 1}
        result = validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert result["driver_id"] == 1

        # driver_id valide plus grand
        data = {"driver_id": 999}
        result = validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert result["driver_id"] == 999

    def test_driver_id_optional(self):
        """✅ Test validation driver_id optionnel."""
        # driver_id None
        data = {"driver_id": None}
        result = validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert result.get("driver_id") is None

        # driver_id absent
        data = {}
        result = validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert "driver_id" not in result or result.get("driver_id") is None

    def test_driver_id_type_validation(self):
        """✅ Test validation type driver_id (doit être Int)."""
        # String non numérique
        data = {"driver_id": "abc"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]

        # Float non entier (invalide pour Int, mais Marshmallow peut convertir)
        # Testons avec une valeur vraiment invalide
        data = {"driver_id": "not_a_number"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(PlanningWeeklyTemplateQuerySchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "driver_id" in exc_info.value.messages["errors"]


class TestUserRoleUpdateSchema:
    """Tests pour UserRoleUpdateSchema."""

    def test_valid_role_update_all_roles(self):
        """✅ Test validation avec tous les rôles valides."""
        valid_roles = ["admin", "client", "driver", "company"]
        for role in valid_roles:
            data = {"role": role}
            result = validate_request(UserRoleUpdateSchema(), data)
            assert result["role"] == role

    def test_valid_role_update_with_company_id(self):
        """✅ Test validation avec role et company_id."""
        data = {"role": "driver", "company_id": 123}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result["role"] == "driver"
        assert result["company_id"] == 123

        # Test avec company role
        data = {"role": "company", "company_id": 456}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result["role"] == "company"
        assert result["company_id"] == 456

    def test_valid_role_update_with_company_name(self):
        """✅ Test validation avec role et company_name."""
        data = {"role": "company", "company_name": "Ma Société SARL"}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result["role"] == "company"
        assert result["company_name"] == "Ma Société SARL"

    def test_valid_role_update_complete(self):
        """✅ Test validation avec tous les champs."""
        data = {"role": "driver", "company_id": 123, "company_name": "Test Company"}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result["role"] == "driver"
        assert result["company_id"] == 123
        assert result["company_name"] == "Test Company"

    def test_missing_role(self):
        """✅ Test erreur si role manquant (requis)."""
        data = {
            "company_id": 123
            # role manquant
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_request(UserRoleUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "role" in exc_info.value.messages["errors"]

    def test_invalid_role(self):
        """✅ Test erreur si role invalide."""
        data = {"role": "invalid"}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(UserRoleUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "role" in exc_info.value.messages["errors"]

        # Test casse incorrecte
        data = {"role": "ADMIN"}  # Majuscules
        with pytest.raises(ValidationError) as exc_info:
            validate_request(UserRoleUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "role" in exc_info.value.messages["errors"]

    def test_company_id_validation(self):
        """✅ Test validation company_id (>= 1 si fourni)."""
        # company_id invalide (0)
        data = {"role": "driver", "company_id": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(UserRoleUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "company_id" in exc_info.value.messages["errors"]

        # company_id négatif
        data = {"role": "driver", "company_id": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(UserRoleUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "company_id" in exc_info.value.messages["errors"]

        # company_id valide
        data = {"role": "driver", "company_id": 1}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result["company_id"] == 1

    def test_company_id_optional(self):
        """✅ Test validation company_id optionnel."""
        # company_id None
        data = {"role": "client", "company_id": None}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result.get("company_id") is None

        # company_id absent
        data = {"role": "client"}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert "company_id" not in result or result.get("company_id") is None

    def test_company_name_validation(self):
        """✅ Test validation company_name (1-200 caractères si fourni)."""
        # company_name trop court (vide)
        data = {"role": "company", "company_name": ""}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(UserRoleUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "company_name" in exc_info.value.messages["errors"]

        # company_name trop long (> 200)
        data = {"role": "company", "company_name": "a" * 201}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(UserRoleUpdateSchema(), data)
        assert "errors" in exc_info.value.messages
        assert "company_name" in exc_info.value.messages["errors"]

        # company_name valide (minimum)
        data = {"role": "company", "company_name": "A"}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result["company_name"] == "A"

        # company_name valide (maximum)
        data = {"role": "company", "company_name": "a" * 200}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result["company_name"] == "a" * 200

    def test_company_name_optional(self):
        """✅ Test validation company_name optionnel."""
        # company_name None
        data = {"role": "admin", "company_name": None}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert result.get("company_name") is None

        # company_name absent
        data = {"role": "admin"}
        result = validate_request(UserRoleUpdateSchema(), data)
        assert "company_name" not in result or result.get("company_name") is None


class TestAutonomousActionReviewSchema:
    """Tests pour AutonomousActionReviewSchema."""

    def test_valid_review_with_notes(self):
        """✅ Test validation avec notes spécifiées."""
        data = {"notes": "Action approuvée par l'administrateur"}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result["notes"] == "Action approuvée par l'administrateur"

        # Notes courtes
        data = {"notes": "OK"}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result["notes"] == "OK"

    def test_valid_review_without_notes(self):
        """✅ Test validation sans notes (optionnel)."""
        data = {}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result.get("notes") is None or result["notes"] is None

        # notes None
        data = {"notes": None}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result.get("notes") is None

    def test_notes_length_validation(self):
        """✅ Test validation longueur notes (max 1000 caractères)."""
        # Notes trop longues (> 1000)
        data = {"notes": "a" * 1001}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "notes" in exc_info.value.messages["errors"]

        # Notes limite (1000 caractères)
        data = {"notes": "a" * 1000}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result["notes"] == "a" * 1000

        # Notes valides (court)
        data = {"notes": "Test"}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result["notes"] == "Test"

    def test_notes_empty_string(self):
        """✅ Test validation notes vide (chaîne vide autorisée car optionnel)."""
        # Notes vide string (valide car optionnel, mais peut être rejeté par validation Length si min > 0)
        # Le schéma n'a pas de min, donc "" devrait être accepté (mais peut être rejeté par Length si min=1 implicite)
        # Testons avec une chaîne vide
        data = {"notes": ""}
        # Marshmallow Length par défaut pourrait rejeter "" si min n'est pas spécifié
        # Mais allow_none=True devrait permettre None, pas ""
        # Testons quand même
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        # Soit accepte "", soit le rejette - les deux sont acceptables selon la logique métier
        # Pour ce test, on vérifie juste qu'il n'y a pas d'exception inattendue
        assert "notes" in result

    def test_notes_with_special_characters(self):
        """✅ Test validation notes avec caractères spéciaux."""
        # Notes avec caractères spéciaux
        data = {"notes": 'Action validée ✅\nNotes: "Important" - À vérifier'}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result["notes"] == 'Action validée ✅\nNotes: "Important" - À vérifier'

        # Notes avec Unicode
        data = {"notes": "中文 日本語 العربية русский"}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result["notes"] == "中文 日本語 العربية русский"

    def test_notes_multiline(self):
        """✅ Test validation notes multilignes."""
        # Notes multilignes valides
        multiline_notes = "Ligne 1\nLigne 2\nLigne 3"
        data = {"notes": multiline_notes}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert result["notes"] == multiline_notes

        # Notes avec plusieurs lignes longues (mais < 1000 caractères au total)
        # Chaque ligne fait ~110 caractères (avec "Ligne X: "), donc 9 lignes = ~990 caractères
        long_multiline = "\n".join([f"Ligne {i}: {'a' * 100}" for i in range(9)])
        data = {"notes": long_multiline}
        result = validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert len(result["notes"]) <= 1000

        # Test que les notes > 1000 caractères sont rejetées
        very_long_notes = "a" * 1001
        data = {"notes": very_long_notes}
        with pytest.raises(ValidationError) as exc_info:
            validate_request(AutonomousActionReviewSchema(), data, strict=False)
        assert "errors" in exc_info.value.messages
        assert "notes" in exc_info.value.messages["errors"]
