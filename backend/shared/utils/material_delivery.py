"""Contrat unique : mission_type=material_delivery + delivery_description."""

from __future__ import annotations

from typing import Any

MATERIAL_DELIVERY = "material_delivery"
PATIENT_TRANSPORT = "patient_transport"


def normalize_mission_type(value: Any) -> str:
    key = str(value or "").strip().lower()
    return key or PATIENT_TRANSPORT


def is_material_delivery(value: Any) -> bool:
    return normalize_mission_type(value) == MATERIAL_DELIVERY


def normalize_delivery_description(value: Any) -> str:
    raw = str(value or "").strip()
    return " ".join(raw.split()) if raw else ""


def require_delivery_description_on_write(
    *,
    mission_type: Any,
    delivery_description: Any,
    mission_type_in_payload: bool,
    description_in_payload: bool,
    existing_mission_type: Any = None,
    existing_description: Any = None,
) -> str | None:
    """Valide la description selon le type effectif.

    - Nouvelle livraison / passage en livraison : description obligatoire.
    - PATCH sans toucher à la description : une ancienne livraison vide reste
      modifiable (compatibilité).
    - Suppression explicite d'une description existante : refusée.
    """
    effective_type = (
        normalize_mission_type(mission_type)
        if mission_type_in_payload
        else normalize_mission_type(existing_mission_type)
    )
    if not is_material_delivery(effective_type):
        return None

    incoming = (
        normalize_delivery_description(delivery_description)
        if description_in_payload
        else None
    )
    existing = normalize_delivery_description(existing_description)

    if description_in_payload:
        if not incoming:
            raise ValueError(
                "Veuillez saisir une description pour la livraison."
            )
        return incoming

    if mission_type_in_payload and not existing:
        raise ValueError(
            "Veuillez saisir une description pour la livraison."
        )
    return existing or None
