"""Contrat unique mission_type=material_delivery + delivery_description."""

import pytest

from shared.utils.material_delivery import (
    is_material_delivery,
    require_delivery_description_on_write,
)


def test_create_delivery_with_description_ok():
    assert (
        require_delivery_description_on_write(
            mission_type="material_delivery",
            delivery_description="Livraison d'effets personnels au domicile du patient.",
            mission_type_in_payload=True,
            description_in_payload=True,
        )
        == "Livraison d'effets personnels au domicile du patient."
    )


@pytest.mark.parametrize("value", ["", "   ", None])
def test_create_delivery_blank_description_rejected(value):
    with pytest.raises(ValueError, match="description"):
        require_delivery_description_on_write(
            mission_type="material_delivery",
            delivery_description=value,
            mission_type_in_payload=True,
            description_in_payload=True,
        )


def test_patient_transport_without_description_ok():
    assert (
        require_delivery_description_on_write(
            mission_type="patient_transport",
            delivery_description=None,
            mission_type_in_payload=True,
            description_in_payload=False,
        )
        is None
    )


def test_switch_to_delivery_requires_description():
    with pytest.raises(ValueError, match="description"):
        require_delivery_description_on_write(
            mission_type="material_delivery",
            delivery_description=None,
            mission_type_in_payload=True,
            description_in_payload=False,
            existing_mission_type="patient_transport",
            existing_description=None,
        )


def test_legacy_empty_patch_other_fields_ok():
    assert (
        require_delivery_description_on_write(
            mission_type=None,
            delivery_description=None,
            mission_type_in_payload=False,
            description_in_payload=False,
            existing_mission_type="material_delivery",
            existing_description=None,
        )
        is None
    )


def test_patient_transport_with_accidental_description_is_not_delivery():
    """Une description ne doit jamais transformer le type de mission."""
    assert is_material_delivery("patient_transport") is False
    assert (
        require_delivery_description_on_write(
            mission_type="patient_transport",
            delivery_description="Livraison de documents",
            mission_type_in_payload=True,
            description_in_payload=True,
        )
        is None
    )


def test_explicit_clear_existing_description_rejected():
    with pytest.raises(ValueError, match="description"):
        require_delivery_description_on_write(
            mission_type="material_delivery",
            delivery_description="   ",
            mission_type_in_payload=False,
            description_in_payload=True,
            existing_mission_type="material_delivery",
            existing_description="Fauteuil roulant",
        )
