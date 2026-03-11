"""Adaptateur mobile → payload canonique ManualBookingCreateSchema.

Transforme le payload mobile (RideCreatePayload) vers le format attendu
par ManualBookingCreateSchema. Couche de transformation uniquement,
sans logique métier.

assign_driver_id et priority ne font PAS partie du contrat canonique
et ne doivent pas être inclus dans le payload retourné.
"""

from __future__ import annotations


def map_mobile_ride_payload_to_manual_booking_payload(
    payload: dict,
) -> dict:
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
    result: dict = {}

    # Adresses (conversion des noms)
    if "pickup_address" in payload:
        result["pickup_location"] = payload["pickup_address"]
    if "dropoff_address" in payload:
        result["dropoff_location"] = payload["dropoff_address"]

    # Aller-retour
    if "is_return" in payload:
        result["is_round_trip"] = bool(payload["is_return"])

    # return_time → return_date / return_time
    return_time_raw = payload.get("return_time")
    if return_time_raw:
        return_time_str = str(return_time_raw).strip()
        if "T" in return_time_str:
            # Datetime: YYYY-MM-DDTHH:mm:ss
            parts = return_time_str.split("T")
            result["return_date"] = parts[0]
            if len(parts) > 1:
                time_part = parts[1]
                if len(time_part.split(":")) == 2:
                    time_part = f"{time_part}:00"
                result["return_time"] = f"{parts[0]}T{time_part}"
        else:
            # Date seule: YYYY-MM-DD (heure à confirmer)
            result["return_date"] = return_time_str
            # Pas de return_time → heure à confirmer

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
