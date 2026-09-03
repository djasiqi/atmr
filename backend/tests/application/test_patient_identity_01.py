"""PATIENT-IDENTITY-01 — civilité + DOB obligatoires, mineur avec confirmation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from marshmallow import ValidationError

from application.institutions.patient_identity_rules import (
    adult_dob_cutoff,
    is_minor,
    requires_minor_dob_confirmation,
    validate_patient_dob,
)
from schemas.institution_schemas import (
    InstitutionPatientCreateSchema,
    InstitutionPatientUpdateSchema,
)


def _subtract_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


class TestValidatePatientDob:
    def test_future_rejected(self):
        with pytest.raises(ValidationError, match="futur"):
            validate_patient_dob("2027-01-01", today=date(2026, 9, 3))

    def test_today_accepted_as_minor(self):
        today = date(2026, 9, 3)
        assert validate_patient_dob("2026-09-03", today=today) == today
        assert is_minor(today, today=today) is True

    def test_exactly_18_not_minor(self):
        today = date(2026, 9, 3)
        dob = adult_dob_cutoff(today)
        assert validate_patient_dob(dob.isoformat(), today=today) == dob
        assert is_minor(dob, today=today) is False

    def test_17_is_minor_but_valid(self):
        today = date(2026, 9, 3)
        minor = adult_dob_cutoff(today) + timedelta(days=1)
        assert validate_patient_dob(minor.isoformat(), today=today) == minor
        assert is_minor(minor, today=today) is True

    def test_invalid_calendar_date(self):
        with pytest.raises(ValidationError, match="invalide"):
            validate_patient_dob("2026-02-31", today=date(2026, 9, 3))

    def test_leap_day_age_before_birthday(self):
        """29.02.2008 → au 28.02.2026 encore 17 ans."""
        dob = date(2008, 2, 29)
        assert is_minor(dob, today=date(2026, 2, 28)) is True

    def test_leap_day_age_on_march_first(self):
        """29.02.2008 → au 01.03.2026 a 18 ans."""
        dob = date(2008, 2, 29)
        assert is_minor(dob, today=date(2026, 3, 1)) is False
        assert validate_patient_dob("2008-02-29", today=date(2026, 3, 1)) == dob

    def test_update_minor_to_other_minor_needs_confirm(self):
        today = date(2026, 9, 3)
        m1 = adult_dob_cutoff(today) + timedelta(days=1)
        m2 = adult_dob_cutoff(today) + timedelta(days=30)
        assert (
            requires_minor_dob_confirmation(
                new_dob=m2,
                previous_dob=m1,
                today=today,
            )
            is True
        )


class TestRequiresMinorConfirmation:
    def test_create_adult_no_confirm(self):
        assert (
            requires_minor_dob_confirmation(
                new_dob=date(1985, 3, 15),
                previous_dob=None,
            )
            is False
        )

    def test_create_minor_needs_confirm(self):
        today = date(2026, 9, 3)
        minor = adult_dob_cutoff(today) + timedelta(days=1)
        assert (
            requires_minor_dob_confirmation(
                new_dob=minor,
                previous_dob=None,
                today=today,
            )
            is True
        )

    def test_update_unchanged_minor_no_reconfirm(self):
        today = date(2026, 9, 3)
        minor = adult_dob_cutoff(today) + timedelta(days=1)
        assert (
            requires_minor_dob_confirmation(
                new_dob=minor,
                previous_dob=minor,
                today=today,
            )
            is False
        )

    def test_update_adult_to_minor_needs_confirm(self):
        today = date(2026, 9, 3)
        minor = adult_dob_cutoff(today) + timedelta(days=1)
        assert (
            requires_minor_dob_confirmation(
                new_dob=minor,
                previous_dob=date(1980, 1, 1),
                today=today,
            )
            is True
        )


class TestInstitutionPatientIdentitySchemas:
    def test_create_requires_gender_and_dob(self):
        schema = InstitutionPatientCreateSchema()
        errors = schema.validate({"first_name": "A", "last_name": "B"})
        assert "gender" in errors or "dob" in errors

    def test_create_future_dob_rejected(self):
        schema = InstitutionPatientCreateSchema()
        errors = schema.validate(
            {
                "first_name": "A",
                "last_name": "B",
                "gender": "HOMME",
                "dob": "2027-01-01",
            }
        )
        assert "dob" in errors

    def test_create_minor_dob_schema_ok(self):
        """Schema accepte le mineur — la confirmation est dans la route."""
        schema = InstitutionPatientCreateSchema()
        today = date.today()
        minor = adult_dob_cutoff(today) + timedelta(days=1)
        data = schema.load(
            {
                "first_name": "A",
                "last_name": "B",
                "gender": "FEMME",
                "dob": minor.isoformat(),
                "minor_dob_confirmed": True,
                "address": "12 rue du Lac",
                "postal_code": "1200",
                "city": "Genève",
            }
        )
        assert data["dob"] == minor.isoformat()
        assert data["minor_dob_confirmed"] is True

    def test_create_adult_ok(self):
        schema = InstitutionPatientCreateSchema()
        data = schema.load(
            {
                "first_name": "A",
                "last_name": "B",
                "gender": "HOMME",
                "dob": "1985-03-15",
                "address": "12 rue du Lac",
                "postal_code": "1200",
                "city": "Genève",
            }
        )
        assert data["dob"] == "1985-03-15"
        assert data["city"] == "Genève"

    def test_create_missing_address_rejected(self):
        schema = InstitutionPatientCreateSchema()
        errors = schema.validate(
            {
                "first_name": "A",
                "last_name": "B",
                "gender": "HOMME",
                "dob": "1985-03-15",
                "postal_code": "1200",
                "city": "Genève",
            }
        )
        assert "address" in errors

    def test_create_blank_city_rejected(self):
        schema = InstitutionPatientCreateSchema()
        errors = schema.validate(
            {
                "first_name": "A",
                "last_name": "B",
                "gender": "HOMME",
                "dob": "1985-03-15",
                "address": "12 rue",
                "postal_code": "1200",
                "city": "   ",
            }
        )
        assert "city" in errors

    def test_update_partial_without_dob_ok(self):
        schema = InstitutionPatientUpdateSchema()
        data = schema.load({"phone": "+41791234567"})
        assert data["phone"] == "+41791234567"

    def test_update_null_dob_rejected(self):
        schema = InstitutionPatientUpdateSchema()
        errors = schema.validate({"dob": None})
        assert "dob" in errors

    def test_update_empty_gender_rejected(self):
        schema = InstitutionPatientUpdateSchema()
        errors = schema.validate({"gender": ""})
        assert "gender" in errors

    def test_update_future_dob_rejected(self):
        schema = InstitutionPatientUpdateSchema()
        future = date.today() + timedelta(days=30)
        errors = schema.validate({"dob": future.isoformat()})
        assert "dob" in errors
