"""Destination : service OU médecin uniquement si destination_type=medical."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from marshmallow import ValidationError

from application.institutions.destination_details_rules import (
    MEDICAL_DESTINATION_OR_ERROR,
    has_service_or_doctor,
    is_medical_destination_type,
    validate_medical_destination,
    validate_medical_destination_details,
)
from schemas.institution_schemas import (
    TransportRequestCreateSchema,
    TransportRequestUpdateSchema,
)


def _future_iso() -> str:
    return (
        (datetime.now(UTC) + timedelta(days=2))
        .replace(hour=10, minute=0, second=0, microsecond=0)
        .isoformat()
    )


def _future_date() -> str:
    return (datetime.now(UTC) + timedelta(days=2)).date().isoformat()


class TestHasServiceOrDoctor:
    def test_service_only(self):
        assert has_service_or_doctor("Radiologie", "") is True

    def test_doctor_only(self):
        assert has_service_or_doctor("  ", "Dr Martin") is True

    def test_both(self):
        assert has_service_or_doctor("Cardio", "Dr Martin") is True

    def test_both_empty(self):
        assert has_service_or_doctor("", None) is False
        assert has_service_or_doctor("   ", "  ") is False


class TestIsMedicalDestinationType:
    def test_only_explicit_medical(self):
        assert is_medical_destination_type("medical") is True
        assert is_medical_destination_type("other") is False
        assert is_medical_destination_type("") is False
        assert is_medical_destination_type(None) is False
        assert is_medical_destination_type("domicile") is False


class TestValidateMedicalDestination:
    def test_rejects_both_empty_on_medical(self):
        with pytest.raises(ValidationError, match="service ou le médecin"):
            validate_medical_destination("", "", destination_type="medical")

    def test_accepts_service_only(self):
        validate_medical_destination("Radiologie", "", destination_type="medical")

    def test_accepts_doctor_only(self):
        validate_medical_destination("", "Dr Martin", destination_type="medical")

    def test_accepts_both(self):
        validate_medical_destination(
            "Cardiologie", "Dr Martin", destination_type="medical"
        )

    def test_other_allows_both_empty(self):
        validate_medical_destination("", "", destination_type="other")

    def test_omitted_type_is_not_medical(self):
        validate_medical_destination("", "")

    def test_domicile_skips(self):
        validate_medical_destination("", "", destination_type="domicile")

    def test_material_delivery_skips(self):
        validate_medical_destination(
            "",
            "",
            destination_type="medical",
            mission_type="material_delivery",
        )


class TestValidateMedicalDestinationDetails:
    def test_restaurant_other_allows_both_empty(self):
        validate_medical_destination_details(
            {
                "destination_type": "other",
                "dropoff_type": "other",
                "dropoff_location": "Restaurant Les Armures, Genève",
            }
        )

    def test_hotel_other_allows_both_empty(self):
        validate_medical_destination_details(
            {
                "destination_type": "other",
                "dropoff_location": "Hôtel Beau-Rivage, Genève",
            }
        )

    def test_generic_other_allows_both_empty(self):
        validate_medical_destination_details(
            {
                "destination_type": "other",
                "dropoff_location": "Gare Cornavin",
            }
        )

    def test_medical_rejects_both_empty(self):
        with pytest.raises(ValidationError, match="service ou le médecin"):
            validate_medical_destination_details(
                {
                    "destination_type": "medical",
                    "dropoff_location": "HUG",
                    "dropoff_service": "",
                    "dropoff_doctor": "",
                }
            )

    def test_medical_accepts_service_only(self):
        validate_medical_destination_details(
            {
                "destination_type": "medical",
                "dropoff_location": "HUG",
                "dropoff_service": "Radiologie",
            }
        )

    def test_medical_accepts_doctor_only(self):
        validate_medical_destination_details(
            {
                "destination_type": "medical",
                "dropoff_location": "HUG",
                "dropoff_doctor": "Dr Martin",
            }
        )

    def test_legacy_omitted_type_is_not_medical(self):
        validate_medical_destination_details({"dropoff_location": "HUG"})

    def test_legacy_empty_dropoff_type_is_not_medical(self):
        validate_medical_destination_details(
            {
                "dropoff_type": "",
                "dropoff_location": "HUG",
            }
        )

    def test_dropoff_type_other_without_destination_type_allows_empty(self):
        validate_medical_destination_details(
            {
                "dropoff_type": "other",
                "dropoff_location": "Restaurant Les Armures",
            }
        )

    def test_domicile_skips_rule(self):
        validate_medical_destination_details(
            {
                "destination_type": "domicile",
                "dropoff_type": "domicile",
                "dropoff_location": "Rue du Patient 1",
            }
        )

    def test_material_delivery_skips_rule(self):
        validate_medical_destination_details(
            {
                "mission_type": "material_delivery",
                "destination_type": "medical",
                "dropoff_location": "Clinique",
            }
        )

    def test_multi_stop_rejects_only_incomplete_medical(self):
        with pytest.raises(ValidationError, match="service ou le médecin"):
            validate_medical_destination_details(
                {
                    "multi_stop": True,
                    "intermediate_stops": [
                        {
                            "dropoff_location": "Restaurant Les Armures",
                            "destination_type": "other",
                        },
                        {
                            "dropoff_location": "HUG",
                            "destination_type": "medical",
                        },
                    ],
                }
            )

    def test_multi_stop_restaurant_plus_medical_service_passes(self):
        validate_medical_destination_details(
            {
                "multi_stop": True,
                "intermediate_stops": [
                    {
                        "dropoff_location": "Restaurant Les Armures",
                        "destination_type": "other",
                    },
                    {
                        "dropoff_location": "HUG",
                        "destination_type": "medical",
                        "dropoff_service": "Radiologie",
                    },
                ],
            }
        )

    def test_partial_update_notes_only_skips(self):
        validate_medical_destination_details(
            {"notes": "Correction horaire"},
            partial=True,
        )


class TestCreateSchemaMedicalDestination:
    def test_other_without_service_or_doctor_passes(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "pickup_location": "Clinique",
                "dropoff_location": "Restaurant Les Armures, Genève",
                "dropoff_type": "other",
                "destination_type": "other",
            }
        )
        assert loaded["destination_type"] == "other"

    def test_create_accepts_destination_type_on_first_intermediate_stop(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "pickup_location": "Clinique",
                "multi_stop": True,
                "return_to_institution": True,
                "intermediate_stops": [
                    {
                        "dropoff_location": "Restaurant Les Armures",
                        "destination_type": "other",
                    }
                ],
            }
        )
        assert loaded["intermediate_stops"][0]["destination_type"] == "other"

    def test_medical_rejects_both_empty(self):
        schema = TransportRequestCreateSchema()
        with pytest.raises(ValidationError) as exc:
            schema.load(
                {
                    "mission_date": _future_date(),
                    "pickup_location": "Clinique",
                    "dropoff_location": "HUG",
                    "destination_type": "medical",
                }
            )
        assert MEDICAL_DESTINATION_OR_ERROR in str(exc.value)

    def test_accepts_service_only_on_medical(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "scheduled_time": _future_iso(),
                "pickup_location": "Clinique",
                "dropoff_location": "HUG",
                "destination_type": "medical",
                "dropoff_service": "Radiologie",
            }
        )
        assert loaded["dropoff_service"] == "Radiologie"

    def test_accepts_doctor_only_on_medical(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "pickup_location": "Clinique",
                "dropoff_location": "HUG",
                "destination_type": "medical",
                "dropoff_doctor": "Dr Martin",
            }
        )
        assert loaded["dropoff_doctor"] == "Dr Martin"

    def test_promotes_routing_destination_type(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "pickup_location": "Clinique",
                "dropoff_location": "HUG",
                "billing_details": {
                    "routing": {
                        "destination_type": "medical",
                        "dropoff_service": "Cardiologie",
                    }
                },
            }
        )
        assert loaded["destination_type"] == "medical"
        assert loaded["dropoff_service"] == "Cardiologie"

    def test_omitted_destination_type_allows_empty_service(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "pickup_location": "A",
                "dropoff_location": "Restaurant Les Armures",
            }
        )
        assert loaded["dropoff_location"] == "Restaurant Les Armures"

    def test_home_final_allows_empty_service_and_doctor(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "pickup_location": "Clinique",
                "dropoff_location": "Domicile patient",
                "dropoff_type": "domicile",
                "destination_type": "domicile",
            }
        )
        assert loaded["dropoff_type"] == "domicile"

    def test_material_delivery_allows_empty_service_and_doctor(self):
        schema = TransportRequestCreateSchema()
        loaded = schema.load(
            {
                "mission_date": _future_date(),
                "pickup_location": "Pharmacie",
                "dropoff_location": "Clinique",
                "mission_type": "material_delivery",
                "delivery_description": "Colis",
            }
        )
        assert loaded["mission_type"] == "material_delivery"

    def test_multi_stop_rejects_incomplete_medical_only(self):
        schema = TransportRequestCreateSchema()
        with pytest.raises(ValidationError) as exc:
            schema.load(
                {
                    "mission_date": _future_date(),
                    "pickup_location": "Clinique",
                    "dropoff_location": "Restaurant",
                    "multi_stop": True,
                    "return_to_institution": False,
                    "intermediate_stops": [
                        {
                            "dropoff_location": "Restaurant",
                            "destination_type": "other",
                        },
                        {
                            "dropoff_location": "HUG",
                            "destination_type": "medical",
                        },
                    ],
                }
            )
        assert MEDICAL_DESTINATION_OR_ERROR in str(exc.value)


class TestUpdateSchemaMedicalDestination:
    def test_rejects_incomplete_medical_intermediate_stop(self):
        schema = TransportRequestUpdateSchema()
        with pytest.raises(ValidationError) as exc:
            schema.load(
                {
                    "multi_stop": True,
                    "intermediate_stops": [
                        {
                            "dropoff_location": "HUG",
                            "destination_type": "medical",
                        }
                    ],
                }
            )
        assert MEDICAL_DESTINATION_OR_ERROR in str(exc.value)

    def test_other_intermediate_stop_allows_empty(self):
        schema = TransportRequestUpdateSchema()
        loaded = schema.load(
            {
                "multi_stop": True,
                "intermediate_stops": [
                    {
                        "dropoff_location": "Restaurant Les Armures",
                        "destination_type": "other",
                    }
                ],
            }
        )
        assert loaded["intermediate_stops"][0]["destination_type"] == "other"

    def test_accepts_notes_only(self):
        schema = TransportRequestUpdateSchema()
        loaded = schema.load({"notes": "RAS"})
        assert loaded["notes"] == "RAS"
