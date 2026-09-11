"""Règles destination — service OU médecin uniquement si type médical.

Le type de trajet (``dropoff_type`` : institution / domicile / other)
n'est pas le type de destination. Seul ``destination_type == medical``
exige un service ou un médecin.

``other``, ``domicile``, ``institution``, ainsi qu'un type omis ou vide
(legacy), n'imposent aucune obligation. Un restaurant, une gare ou un
hôtel ne sont pas médicaux.
"""

from __future__ import annotations

from marshmallow import ValidationError

from models.enums import LocationType, MissionType

MEDICAL_DESTINATION_OR_ERROR = (
    "Veuillez renseigner au moins le service ou le médecin."
)
DESTINATION_TYPE_MEDICAL = "medical"
VALID_DESTINATION_TYPES = (
    DESTINATION_TYPE_MEDICAL,
    LocationType.OTHER.value,
    LocationType.DOMICILE.value,
    LocationType.INSTITUTION.value,
)


def _text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def has_service_or_doctor(service: object | None, doctor: object | None) -> bool:
    """True si le service ou le médecin (ou les deux) est renseigné."""
    return bool(_text(service) or _text(doctor))


def is_patient_transport(mission_type: object | None) -> bool:
    """True si mission patient (défaut création = patient_transport)."""
    if mission_type is None:
        return True
    return str(mission_type) == MissionType.PATIENT_TRANSPORT.value


def is_medical_destination_type(destination_type: object | None) -> bool:
    """True uniquement si le type est explicitement ``medical``.

    ``other`` n'est pas médical. Type omis / vide : legacy, hors règle
    (évite les faux positifs restaurant / gare / hôtel).
    """
    return _text(destination_type) == DESTINATION_TYPE_MEDICAL


def _stop_has_location(stop: dict) -> bool:
    return bool(_text(stop.get("dropoff_location")))


def _is_return_stop(stop: dict) -> bool:
    return bool(stop.get("is_return_stop"))


def _medical_intermediate_stops(stops: list) -> list[tuple[int, dict]]:
    """Étapes explicitement médicales (hors retour)."""
    medical: list[tuple[int, dict]] = []
    for idx, stop in enumerate(stops or []):
        if not isinstance(stop, dict):
            continue
        if not _stop_has_location(stop) or _is_return_stop(stop):
            continue
        if is_medical_destination_type(stop.get("destination_type")):
            medical.append((idx, stop))
    return medical


def _raise_missing(field_name: str) -> None:
    raise ValidationError(MEDICAL_DESTINATION_OR_ERROR, field_name=field_name)


def validate_medical_destination(
    service: object | None,
    doctor: object | None,
    *,
    destination_type: object | None = None,
    mission_type: object | None = None,
    field_name: str = "dropoff_service",
) -> None:
    """Invariant unique : destination médicale ⇒ service OU médecin.

    Utilisé par création/édition de demande et par le PATCH booking
    opérationnel (``hospital_service`` / ``doctor_name``).
    """
    if not is_patient_transport(mission_type):
        return
    if not is_medical_destination_type(destination_type):
        return
    if not has_service_or_doctor(service, doctor):
        _raise_missing(field_name)


def validate_medical_destination_details(
    data: dict, *, partial: bool = False
) -> None:
    """Refuse une destination médicale sans service ni médecin.

    ``partial=True`` (PUT) : n'applique la règle que si le payload touche
    aux destinations (destination_type, dropoff_* ou intermediate_stops).
    """
    if not isinstance(data, dict):
        return
    if not is_patient_transport(data.get("mission_type")):
        return

    stops = data.get("intermediate_stops")
    has_stops = isinstance(stops, list)
    is_multi = bool(data.get("multi_stop")) or (partial and has_stops)

    if is_multi:
        if partial and not has_stops:
            pass
        else:
            for _idx, stop in _medical_intermediate_stops(stops or []):
                validate_medical_destination(
                    stop.get("dropoff_service"),
                    stop.get("dropoff_doctor"),
                    destination_type=stop.get("destination_type"),
                    mission_type=data.get("mission_type"),
                    field_name="intermediate_stops",
                )
            return

    if partial:
        destination_keys = {
            "destination_type",
            "dropoff_type",
            "dropoff_location",
            "dropoff_establishment",
            "dropoff_service",
            "dropoff_doctor",
        }
        if not destination_keys.intersection(data.keys()):
            return

    validate_medical_destination(
        data.get("dropoff_service"),
        data.get("dropoff_doctor"),
        destination_type=data.get("destination_type"),
        mission_type=data.get("mission_type"),
        field_name="dropoff_service",
    )
