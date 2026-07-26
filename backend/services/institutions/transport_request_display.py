"""Blocs d'affichage canoniques TransportRequest — TransportRequestDisplayModel v1.

Contrat : docs/architecture/canonical-display-model.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from services.companies.booking_display import (
    DISPLAY_CATEGORY_INSTITUTION_PATIENT,
    DISPLAY_MODEL_VERSION,
    build_identity_labels,
)
from services.institutions.mission_schedule import (
    get_effective_dispatch_time,
    get_mission_date,
    is_operational_time,
)
from shared.time_utils import split_date_time_local

if TYPE_CHECKING:
    from models.transport_request import TransportRequest

DISPLAY_MODEL_TRANSPORT_REQUEST = "transport_request"

_UNDEFINED_TIME = "À définir"


def _fmt_time_local(scheduled_time: Any) -> str:
    if scheduled_time is None:
        return ""
    _, time_local = split_date_time_local(scheduled_time)
    return time_local or ""


def _leg_display_time(leg: Any) -> str:
    st = getattr(leg, "scheduled_time", None)
    confirmed = bool(getattr(leg, "time_confirmed", False))
    if st is None:
        return _UNDEFINED_TIME
    time_str = _fmt_time_local(st)
    if not time_str:
        return _UNDEFINED_TIME
    if confirmed:
        return time_str
    return f"{time_str} (non confirmé)"


def _departure_display(transport_request: TransportRequest) -> dict[str, Any]:
    dep = getattr(transport_request, "scheduled_time", None)
    confirmed = bool(getattr(transport_request, "pickup_time_confirmed", False))
    time_defined = is_operational_time(scheduled_time=dep, time_confirmed=confirmed)
    if not dep:
        return {"time_defined": False, "display_time": _UNDEFINED_TIME}
    time_str = _fmt_time_local(dep)
    if not time_str:
        return {"time_defined": False, "display_time": _UNDEFINED_TIME}
    if time_defined:
        return {"time_defined": True, "display_time": time_str}
    return {"time_defined": False, "display_time": f"{time_str} (non confirmé)"}


def _return_display(transport_request: TransportRequest) -> dict[str, Any]:
    if not bool(getattr(transport_request, "is_round_trip", False)):
        return {"time_defined": False, "display_time": ""}
    confirmed = bool(getattr(transport_request, "return_time_confirmed", False))
    rt = getattr(transport_request, "return_time", None)
    if rt is None and not getattr(transport_request, "return_date", None):
        return {"time_defined": False, "display_time": ""}
    if rt is None:
        return {"time_defined": False, "display_time": _UNDEFINED_TIME}
    time_str = _fmt_time_local(rt)
    if confirmed and time_str:
        return {"time_defined": True, "display_time": time_str}
    if time_str:
        return {"time_defined": False, "display_time": f"{time_str} (non confirmé)"}
    return {"time_defined": False, "display_time": _UNDEFINED_TIME}


def _build_schedule_summary(transport_request: TransportRequest) -> str:
    parts: list[str] = []
    dep = _departure_display(transport_request)
    if dep.get("display_time") and dep["display_time"] != _UNDEFINED_TIME:
        parts.append(f"{dep['display_time']} Départ")

    legs = sorted(
        getattr(transport_request, "legs", None) or [],
        key=lambda leg: getattr(leg, "sequence_index", 0),
    )
    return_to_inst = bool(getattr(transport_request, "return_to_institution", False))

    for index, leg in enumerate(legs):
        is_return = return_to_inst and index == len(legs) - 1
        label = (
            "Retour"
            if is_return
            else (
                getattr(leg, "dropoff_establishment", None)
                or getattr(leg, "dropoff_service", None)
                or f"Dest. {index + 1}"
            )
        )
        time_label = _leg_display_time(leg)
        if is_return and time_label == _UNDEFINED_TIME:
            parts.append("Retour à définir")
        else:
            parts.append(f"{time_label} {label}")

    if not legs and getattr(transport_request, "dropoff_location", None):
        st = getattr(transport_request, "scheduled_time", None)
        st_type = getattr(transport_request, "scheduled_time_type", None)
        if st_type == "arrival" and st:
            parts.append(f"{_fmt_time_local(st) or _UNDEFINED_TIME} RDV")

    ret = _return_display(transport_request)
    if ret.get("display_time") == _UNDEFINED_TIME and (
        parts or bool(getattr(transport_request, "is_round_trip", False))
    ):
        parts.append("Retour à définir")

    return " · ".join(parts)


def _build_tr_identity(transport_request: TransportRequest) -> dict[str, Any]:
    patient = getattr(transport_request, "patient", None)
    if patient is not None:
        passenger = f"{getattr(patient, 'last_name', '')} {getattr(patient, 'first_name', '')}".strip()
    else:
        passenger = (
            getattr(transport_request, "external_reference", None)
            or f"#{transport_request.id}"
        )
    inst = getattr(transport_request, "institution", None)
    inst_id = getattr(transport_request, "institution_id", None)
    inst_name = getattr(inst, "name", None) if inst else None
    source = {
        "type": "institution",
        "id": int(inst_id) if inst_id is not None else None,
        "name": inst_name or "Institution",
    }
    display_category = DISPLAY_CATEGORY_INSTITUTION_PATIENT
    primary_label, secondary_label = build_identity_labels(
        passenger=passenger,
        source=source,
        display_category=display_category,
    )
    requester = None
    contact = getattr(transport_request, "contact_on_site", None) or {}
    if isinstance(contact, dict) and contact.get("requester_name"):
        requester = {"name": str(contact["requester_name"]).strip()}

    return {
        "passenger": {"name": passenger},
        "display_category": display_category,
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "source": source,
        "requester": requester,
    }


def _build_tr_trip_flags(transport_request: TransportRequest) -> dict[str, Any]:
    legs = list(getattr(transport_request, "legs", None) or [])
    multi = bool(getattr(transport_request, "multi_stop", False)) or len(legs) > 1
    return {
        "round_trip": bool(getattr(transport_request, "is_round_trip", False)),
        "multi_stop": multi,
        "return_to_institution": bool(
            getattr(transport_request, "return_to_institution", False)
        ),
    }


def _build_tr_search_index(identity: dict[str, Any]) -> list[str]:
    tokens: list[str] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text not in tokens:
            tokens.append(text)

    _add(identity.get("primary_label"))
    _add(identity.get("secondary_label"))
    _add((identity.get("passenger") or {}).get("name"))
    source = identity.get("source") or {}
    _add(source.get("name"))
    requester = identity.get("requester")
    if isinstance(requester, dict):
        _add(requester.get("name"))
    return tokens


def build_transport_request_display_blocks(
    transport_request: TransportRequest,
) -> dict[str, Any]:
    """TransportRequestDisplayModel v1."""
    identity = _build_tr_identity(transport_request)
    mission_day = get_mission_date(transport_request)
    next_confirmed = get_effective_dispatch_time(transport_request)
    next_iso = None
    if next_confirmed is not None:
        from shared.time_utils import mission_scheduled_to_api_iso

        next_iso = mission_scheduled_to_api_iso(next_confirmed)

    legs_out: list[dict[str, Any]] = []
    for leg in sorted(
        getattr(transport_request, "legs", None) or [],
        key=lambda item: getattr(item, "sequence_index", 0),
    ):
        st = getattr(leg, "scheduled_time", None)
        scheduled_iso = None
        if st is not None:
            from shared.time_utils import mission_scheduled_to_api_iso

            scheduled_iso = mission_scheduled_to_api_iso(st)
        legs_out.append(
            {
                "sequence_index": getattr(leg, "sequence_index", 0),
                "label": (
                    getattr(leg, "dropoff_establishment", None)
                    or getattr(leg, "dropoff_service", None)
                    or "Étape"
                ),
                "scheduled_time": scheduled_iso,
                "time_confirmed": bool(getattr(leg, "time_confirmed", False)),
                "display_time": _leg_display_time(leg),
            }
        )

    return {
        "display_model": DISPLAY_MODEL_TRANSPORT_REQUEST,
        "display_model_version": DISPLAY_MODEL_VERSION,
        "identity": identity,
        "scheduling": {
            "mission_date": mission_day.isoformat() if mission_day else None,
            "next_confirmed_time": next_iso,
            "summary": _build_schedule_summary(transport_request),
            "departure": _departure_display(transport_request),
            "return": _return_display(transport_request),
        },
        "trip_flags": _build_tr_trip_flags(transport_request),
        "legs": legs_out,
        "search_index": _build_tr_search_index(identity),
    }
