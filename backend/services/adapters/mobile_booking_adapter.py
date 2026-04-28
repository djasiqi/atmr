"""Adaptateur mobile → payload canonique ManualBookingCreateSchema.

Transforme le payload mobile (RideCreatePayload) vers le format attendu
par ManualBookingCreateSchema. Couche de transformation uniquement,
sans logique métier.

assign_driver_id et priority ne font PAS partie du contrat canonique
et ne doivent pas être inclus dans le payload retourné.
"""

from __future__ import annotations

import os
from typing import Any

# Longueur d'une date ISO YYYY-MM-DD
_ISO_DATE_LEN = 10


def map_mobile_ride_payload_to_manual_booking_payload(
    payload: dict[str, object],
    *,
    enforce_structured_address: bool = False,
) -> dict[str, object]:
    """Convertit le payload mobile vers le format canonique ManualBookingCreateSchema.

    Conversions:
        - pickup_address → pickup_location
        - dropoff_address → dropoff_location
        - is_return → is_round_trip
        - return_time: si format date seul (YYYY-MM-DD) → return_date sans return_time;
          si datetime (YYYY-MM-DDTHH:mm:ss) → return_date + return_time

    notes et notes_medical restent distincts (jamais fusionnés).

    Args:
        payload: Payload brut du mobile (RideCreatePayload)

    Returns:
        Payload conforme à ManualBookingCreateSchema
    """
    result: dict[str, object] = {}

    def _to_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    enforce_mode = enforce_structured_address or (
        os.getenv("COMPANY_MOBILE_STRUCTURED_RIDE_PAYLOAD_ENABLED", "0") == "1"
    )

    def _extract_address(
        value: object,
        *,
        address_key: str,
    ) -> tuple[str | None, str | None, float | None, float | None]:
        if isinstance(value, str):
            if enforce_mode:
                raise ValueError(f"{address_key}.label est requis")
            normalized = value.strip()
            return (normalized or None, None, None, None)
        if not isinstance(value, dict):
            if enforce_mode:
                raise ValueError(f"{address_key} doit être un objet structuré")
            return (None, None, None, None)
        label_raw = value.get("label") or value.get("address") or value.get("description")
        place_id_raw = value.get("place_id") or value.get("placeId")
        lat_raw = value.get("lat") or value.get("latitude")
        lon_raw = value.get("lon") or value.get("lng") or value.get("longitude")
        label = str(label_raw).strip() if isinstance(label_raw, str) else None
        place_id = str(place_id_raw).strip() if isinstance(place_id_raw, str) else None
        lat = _to_float(lat_raw)
        lon = _to_float(lon_raw)
        if enforce_mode and not label:
            raise ValueError(f"{address_key}.label est requis en mode structuré")
        return (label or None, place_id or None, lat, lon)

    # Adresses (conversion des noms + extraction coordonnées)
    if "pickup_address" in payload:
        pickup_label, pickup_place_id, pickup_lat, pickup_lon = _extract_address(
            payload["pickup_address"],
            address_key="pickup_address",
        )
        if pickup_label:
            result["pickup_location"] = pickup_label
        if pickup_lat is not None:
            result["pickup_lat"] = pickup_lat
        if pickup_lon is not None:
            result["pickup_lon"] = pickup_lon
        if pickup_place_id:
            result["pickup_place_id"] = pickup_place_id
        if enforce_mode and not pickup_label:
            raise ValueError("pickup_address.label est requis")

    if "dropoff_address" in payload:
        dropoff_label, dropoff_place_id, dropoff_lat, dropoff_lon = _extract_address(
            payload["dropoff_address"],
            address_key="dropoff_address",
        )
        if dropoff_label:
            result["dropoff_location"] = dropoff_label
        if dropoff_lat is not None:
            result["dropoff_lat"] = dropoff_lat
        if dropoff_lon is not None:
            result["dropoff_lon"] = dropoff_lon
        if dropoff_place_id:
            result["dropoff_place_id"] = dropoff_place_id
        if enforce_mode and not dropoff_label:
            raise ValueError("dropoff_address.label est requis")

    # Aller-retour
    if "is_return" in payload:
        result["is_round_trip"] = bool(payload["is_return"])

    # return_date / return_time (aligné sur le web)
    # Web envoie return_date (YYYY-MM-DD) + return_time optionnel (heure à définir si absent)
    # Mobile peut envoyer return_date directement (comme le web) ou return_time à convertir
    return_date_raw = payload.get("return_date")
    return_time_raw = payload.get("return_time")

    if return_date_raw:
        # Mobile envoie return_date directement (format web) → priorité
        result["return_date"] = str(return_date_raw).strip()
        if return_time_raw:
            return_time_str = str(return_time_raw).strip()
            if "T" in return_time_str:
                parts = return_time_str.split("T")
                if len(parts) > 1:
                    time_part = parts[1]
                    _HOUR_MIN_PARTS = 2
                    if len(time_part.split(":")) == _HOUR_MIN_PARTS:
                        time_part = f"{time_part}:00"
                    result["return_time"] = f"{parts[0]}T{time_part}"
    elif return_time_raw:
        # Fallback : return_time seul (conversion legacy)
        return_time_str = str(return_time_raw).strip()
        if "T" in return_time_str:
            parts = return_time_str.split("T")
            result["return_date"] = parts[0]
            if len(parts) > 1:
                time_part = parts[1]
                _HOUR_MIN_PARTS = 2
                if len(time_part.split(":")) == _HOUR_MIN_PARTS:
                    time_part = f"{time_part}:00"
                result["return_time"] = f"{parts[0]}T{time_part}"
        else:
            result["return_date"] = return_time_str
    elif result.get("is_round_trip") and payload.get("scheduled_time"):
        # Dernier recours : dériver return_date de scheduled_time (même jour)
        scheduled = str(payload["scheduled_time"]).strip()
        if "T" in scheduled:
            result["return_date"] = scheduled.split("T")[0]
        else:
            result["return_date"] = (
                scheduled[:_ISO_DATE_LEN] if len(scheduled) >= _ISO_DATE_LEN else scheduled
            )

    # Champs pass-through (noms identiques)
    pass_through = [
        "client_id",
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "scheduled_time",
        "notes_medical",
        "medical_facility",
        "hospital_service",
        "doctor_name",
        "pickup_access_notes",
        "dropoff_access_notes",
        "wheelchair_client_has",
        "wheelchair_need",
        "amount",
        "is_recurring",
        "recurrence_type",
        "recurrence_days",
        "recurrence_end_date",
        "occurrences",
    ]
    for key in pass_through:
        if key in payload and payload[key] is not None:
            result[key] = payload[key]

    # notes: le contrat canonique a notes_medical. Si le mobile envoie "notes"
    # (notes internes), le schéma ManualBookingCreateSchema n'a pas de champ
    # "notes" dédié. On ne fusionne jamais avec notes_medical.
    # Si notes existe côté modèle à l'avenir, l'ajouter ici.
    # Pour l'instant, notes_medical est déjà mappé via pass_through.

    return result
