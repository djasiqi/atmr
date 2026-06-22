"""Construction des blocs d'affichage canoniques pour les réservations entreprise.

Contrat : docs/architecture/canonical-display-model.md (BookingDisplayModel v1).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from models.enums import BookingCreatedVia, ClientType
from shared.time_utils import split_date_time_local

DISPLAY_MODEL_BOOKING = "booking"
DISPLAY_MODEL_VERSION = 1

DISPLAY_CATEGORY_INSTITUTION_PATIENT = "institution_patient"
DISPLAY_CATEGORY_PARTNER_CLIENT = "partner_client"
DISPLAY_CATEGORY_LIRIE_CLIENT = "lirie_client"
DISPLAY_CATEGORY_LIRIE_GUEST = "lirie_guest"
DISPLAY_CATEGORY_COMPANY_CLIENT = "company_client"

SOURCE_TYPE_INSTITUTION = "institution"
SOURCE_TYPE_PARTNER = "partner_company"
SOURCE_TYPE_COMPANY_CLIENT = "company_client"
SOURCE_TYPE_COMPANY_ACCOUNT = "company_account"
SOURCE_TYPE_LIRIE_CLIENT = "lirie_client"
SOURCE_TYPE_LIRIE_GUEST = "lirie_guest"
SOURCE_TYPE_LEGACY = "legacy"

COMPANY_CLIENT_DISPLAY_NAME = "Portefeuille propre"
LIRIE_PLATFORM_NAME = "LIRIE"
GUEST_DISPLAY_NAME = "Invité LIRIE"

_CHANGE_REQUEST_PENDING_STATUSES = frozenset(
    {"pending", "escalation_required", "expired"}
)


def derive_entity_code(
    source_type: str,
    entity_id: int | None,
    display_name: str | None,
) -> str | None:
    """Code court pour filtres (Phase A : acronyme ou identifiant stable)."""
    if source_type == SOURCE_TYPE_LIRIE_CLIENT:
        return "LIRIE"
    if source_type == SOURCE_TYPE_LIRIE_GUEST:
        return "GUEST"
    if entity_id is not None and source_type in {
        SOURCE_TYPE_PARTNER,
        SOURCE_TYPE_COMPANY_CLIENT,
        SOURCE_TYPE_COMPANY_ACCOUNT,
    }:
        return f"C{entity_id}"
    name = (display_name or "").strip()
    if not name:
        return None
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", name)
    if not words:
        return None
    if len(words) == 1:
        token = words[0].upper()
        return token[:8] if len(token) > 8 else token
    return "".join(w[0].upper() for w in words[:6])


def _created_via_value(booking: Any) -> str:
    raw = getattr(booking, "created_via", None)
    if raw is None:
        return BookingCreatedVia.LEGACY.value
    if isinstance(raw, BookingCreatedVia):
        return raw.value
    return str(raw).lower()


def _passenger_name(booking: Any) -> str:
    fn = getattr(booking, "customer_full_name", None)
    if callable(fn):
        name = fn()
    else:
        name = getattr(booking, "customer_name", None)
    text = str(name or "").strip()
    return text or "Non spécifié"


def _get_institution_timeline(booking: Any) -> dict[str, Any] | None:
    fn = getattr(booking, "_get_institution_timeline", None)
    if callable(fn):
        result = fn()
        if isinstance(result, dict):
            return result
    return None


def _get_active_transfer(booking: Any) -> dict[str, Any] | None:
    fn = getattr(booking, "_get_active_transfer_info", None)
    if callable(fn):
        result = fn()
        if isinstance(result, dict):
            return result
    return None


def _has_institution_origin(booking: Any) -> bool:
    timeline = _get_institution_timeline(booking)
    if timeline and timeline.get("institution_name"):
        return True
    reqs = getattr(booking, "source_request", None)
    if reqs:
        return True
    cli = getattr(booking, "client", None)
    if cli and getattr(cli, "linked_institution_id", None):
        return True
    return False


def _resolve_institution_ref(booking: Any) -> dict[str, Any] | None:
    timeline = _get_institution_timeline(booking)
    cli = getattr(booking, "client", None)
    inst_id: int | None = None
    inst_name: str | None = None

    if cli is not None:
        linked = getattr(cli, "linked_institution_id", None)
        if linked is not None:
            inst_id = int(linked)
        if getattr(cli, "institution_name", None):
            inst_name = str(cli.institution_name).strip() or inst_name

    if timeline:
        if timeline.get("institution_name"):
            inst_name = str(timeline["institution_name"]).strip()
        reqs = getattr(booking, "source_request", None)
        if reqs and inst_id is None:
            req = reqs[0] if isinstance(reqs, list) else reqs
            inst = getattr(req, "institution", None)
            if inst is not None and getattr(inst, "id", None) is not None:
                inst_id = int(inst.id)
            if inst is not None and getattr(inst, "name", None):
                inst_name = str(inst.name).strip()

    if not inst_name and not inst_id:
        return None

    name = inst_name or "Institution"
    return {
        "type": SOURCE_TYPE_INSTITUTION,
        "id": inst_id,
        "code": derive_entity_code(SOURCE_TYPE_INSTITUTION, inst_id, name),
        "name": name,
    }


def _is_company_account_client(cli: Any, passenger: str) -> bool:
    if cli is None:
        return False
    if getattr(cli, "linked_institution_id", None):
        return False
    if bool(getattr(cli, "is_institution", False)):
        return True
    inst_name = (getattr(cli, "institution_name", None) or "").strip()
    if inst_name and inst_name.lower() != passenger.lower():
        return True
    return False


def _resolve_crm_source(booking: Any, passenger: str) -> dict[str, Any]:
    cli = getattr(booking, "client", None)
    client_id = int(cli.id) if cli is not None and getattr(cli, "id", None) else None

    if _is_company_account_client(cli, passenger):
        name = (getattr(cli, "institution_name", None) or "").strip() or "Compte entreprise"
        return {
            "type": SOURCE_TYPE_COMPANY_ACCOUNT,
            "id": client_id,
            "code": derive_entity_code(SOURCE_TYPE_COMPANY_ACCOUNT, client_id, name),
            "name": name,
        }

    owner = getattr(booking, "company", None)
    owner_name = getattr(owner, "name", None) if owner else None
    display = COMPANY_CLIENT_DISPLAY_NAME
    if owner_name:
        display = f"{COMPANY_CLIENT_DISPLAY_NAME} · {owner_name}"
    return {
        "type": SOURCE_TYPE_COMPANY_CLIENT,
        "id": client_id,
        "code": derive_entity_code(SOURCE_TYPE_COMPANY_CLIENT, client_id, display),
        "name": display,
    }


def _resolve_upstream(booking: Any, source: dict[str, Any]) -> dict[str, Any] | None:
    institution_ref = _resolve_institution_ref(booking)
    if not institution_ref:
        return None
    if source.get("type") == SOURCE_TYPE_INSTITUTION:
        return None
    if source.get("type") == SOURCE_TYPE_PARTNER:
        return institution_ref
    return None


def resolve_booking_source(
    booking: Any,
    viewer_company_id: int | None,
) -> dict[str, Any]:
    """Origine commerciale viewer-relative."""
    created_via = _created_via_value(booking)
    passenger = _passenger_name(booking)

    if created_via == BookingCreatedVia.PUBLIC_GUEST.value:
        return {
            "type": SOURCE_TYPE_LIRIE_GUEST,
            "id": None,
            "code": "GUEST",
            "name": GUEST_DISPLAY_NAME,
        }

    transfer = _get_active_transfer(booking)
    executing_id = getattr(booking, "executing_company_id", None)
    owner_id = getattr(booking, "company_id", None)

    if (
        transfer
        and viewer_company_id is not None
        and executing_id is not None
        and int(viewer_company_id) == int(executing_id)
        and owner_id is not None
        and int(owner_id) != int(viewer_company_id)
    ):
        owner_name = (
            transfer.get("owner_company_name")
            or getattr(getattr(booking, "company", None), "name", None)
            or "Partenaire"
        )
        oid = int(transfer.get("owner_company_id") or owner_id)
        return {
            "type": SOURCE_TYPE_PARTNER,
            "id": oid,
            "code": derive_entity_code(SOURCE_TYPE_PARTNER, oid, owner_name),
            "name": owner_name,
        }

    if _has_institution_origin(booking):
        ref = _resolve_institution_ref(booking)
        if ref:
            return ref

    cli = getattr(booking, "client", None)
    client_type = getattr(cli, "client_type", None)
    client_type_val = (
        client_type.value if hasattr(client_type, "value") else str(client_type or "")
    ).upper()

    if (
        created_via == BookingCreatedVia.CLIENT_APP.value
        or client_type_val == ClientType.PORTAL.value
    ):
        return {
            "type": SOURCE_TYPE_LIRIE_CLIENT,
            "id": int(cli.id) if cli is not None and getattr(cli, "id", None) else None,
            "code": "LIRIE",
            "name": LIRIE_PLATFORM_NAME,
        }

    booking_type = str(getattr(booking, "booking_type", "") or "").lower()
    if (
        created_via == BookingCreatedVia.DISPATCHER.value
        or booking_type == "manual"
        or client_type_val == ClientType.TRANSPORT.value
    ):
        return _resolve_crm_source(booking, passenger)

    if created_via == BookingCreatedVia.INSTITUTION_PORTAL.value:
        ref = _resolve_institution_ref(booking)
        if ref:
            return ref

    return {
        "type": SOURCE_TYPE_LEGACY,
        "id": getattr(booking, "client_id", None),
        "code": None,
        "name": passenger,
    }


def _passenger_birth_date(booking: Any) -> str | None:
    fn = getattr(booking, "_get_institution_passenger_brief", None)
    if callable(fn):
        brief = fn()
        if isinstance(brief, dict):
            raw = brief.get("birth_date")
            if raw:
                return str(raw)
    return None


def _passenger_gender(booking: Any) -> str | None:
    fn = getattr(booking, "_get_institution_passenger_brief", None)
    if callable(fn):
        brief = fn()
        if isinstance(brief, dict):
            raw = brief.get("gender")
            if raw:
                return str(raw)
    cli = getattr(booking, "client", None)
    cli_user = getattr(cli, "user", None) if cli is not None else None
    gender_raw = getattr(cli_user, "gender", None) if cli_user is not None else None
    if gender_raw is None:
        return None
    if hasattr(gender_raw, "value"):
        return str(gender_raw.value)
    text = str(gender_raw).strip()
    return text or None


def build_booking_identity(
    booking: Any,
    viewer_company_id: int | None = None,
) -> dict[str, Any]:
    passenger = _passenger_name(booking)
    source = resolve_booking_source(booking, viewer_company_id)

    timeline = _get_institution_timeline(booking)
    requester: dict[str, Any] | None = None
    if timeline and timeline.get("created_by_name"):
        requester = {
            "id": None,
            "name": str(timeline["created_by_name"]).strip(),
        }

    owner_id = getattr(booking, "company_id", None)
    owner = getattr(booking, "company", None)
    owner_name = getattr(owner, "name", None) if owner else None
    ownership = {
        "owner_company_id": owner_id,
        "owner_company_name": owner_name,
    }

    exec_id = getattr(booking, "executing_company_id", None) or owner_id
    exec_name: str | None = None
    if exec_id is not None:
        if (
            getattr(booking, "executing_company", None) is not None
            and int(getattr(booking.executing_company, "id", -1)) == int(exec_id)
        ):
            exec_name = getattr(booking.executing_company, "name", None)
        elif owner is not None and owner_id is not None and int(exec_id) == int(owner_id):
            exec_name = owner_name
        transfer = _get_active_transfer(booking)
        if not exec_name and transfer:
            if int(transfer.get("executing_company_id") or 0) == int(exec_id):
                exec_name = transfer.get("executing_company_name")
            elif int(transfer.get("owner_company_id") or 0) == int(exec_id):
                exec_name = transfer.get("owner_company_name")

    execution = {
        "executing_company_id": exec_id,
        "executing_company_name": exec_name,
    }

    upstream = _resolve_upstream(booking, source)

    display_category = derive_display_category(source.get("type"))
    primary_label, secondary_label = build_identity_labels(
        passenger=passenger,
        source=source,
        display_category=display_category,
    )

    return {
        "passenger": {
            "name": passenger,
            "birth_date": _passenger_birth_date(booking),
            "gender": _passenger_gender(booking),
        },
        "display_category": display_category,
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "source": source,
        "requester": requester,
        "ownership": ownership,
        "execution": execution,
        "upstream": upstream,
        "origin_channel": _created_via_value(booking),
    }


def derive_display_category(source_type: str | None) -> str:
    """Mappe source.type vers display_category (liste fermée INV-4)."""
    key = str(source_type or SOURCE_TYPE_LEGACY).lower()
    mapping = {
        SOURCE_TYPE_INSTITUTION: DISPLAY_CATEGORY_INSTITUTION_PATIENT,
        SOURCE_TYPE_PARTNER: DISPLAY_CATEGORY_PARTNER_CLIENT,
        SOURCE_TYPE_LIRIE_CLIENT: DISPLAY_CATEGORY_LIRIE_CLIENT,
        SOURCE_TYPE_LIRIE_GUEST: DISPLAY_CATEGORY_LIRIE_GUEST,
        SOURCE_TYPE_COMPANY_CLIENT: DISPLAY_CATEGORY_COMPANY_CLIENT,
        SOURCE_TYPE_COMPANY_ACCOUNT: DISPLAY_CATEGORY_COMPANY_CLIENT,
        SOURCE_TYPE_LEGACY: DISPLAY_CATEGORY_COMPANY_CLIENT,
    }
    return mapping.get(key, DISPLAY_CATEGORY_COMPANY_CLIENT)


def build_identity_labels(
    *,
    passenger: str,
    source: dict[str, Any],
    display_category: str,
) -> tuple[str, str]:
    """Produit primary_label / secondary_label (INV-8)."""
    primary = passenger or "Non spécifié"
    source_name = str(source.get("name") or "").strip()

    if display_category == DISPLAY_CATEGORY_INSTITUTION_PATIENT:
        return primary, source_name or "Institution"
    if display_category == DISPLAY_CATEGORY_PARTNER_CLIENT:
        return primary, source_name or "Partenaire"
    if display_category == DISPLAY_CATEGORY_LIRIE_CLIENT:
        return primary, LIRIE_PLATFORM_NAME
    if display_category == DISPLAY_CATEGORY_LIRIE_GUEST:
        return primary, GUEST_DISPLAY_NAME
    return primary, source_name or COMPANY_CLIENT_DISPLAY_NAME


def is_legacy_midnight_pickup_sentinel(
    scheduled_dt: datetime | None,
    *,
    time_confirmed: bool | None = None,
) -> bool:
    """Transition Phase 2→4 : T00:00:00 sans confirmation = pas d'heure métier (legacy).

    Minuit réel confirmé (BK-01c) n'est pas une sentinelle.
    """
    if scheduled_dt is None:
        return False
    st = (
        scheduled_dt.replace(tzinfo=None)
        if getattr(scheduled_dt, "tzinfo", None)
        else scheduled_dt
    )
    if not (st.hour == 0 and st.minute == 0 and st.second == 0):
        return False
    if time_confirmed is True:
        return False
    return True


def booking_has_scheduled_pickup_time(booking: Any) -> bool:
    """Existence d'une heure métier — indépendant de la confirmation workflow."""
    scheduled_dt = getattr(booking, "scheduled_time", None)
    if scheduled_dt is None:
        return False
    time_confirmed = getattr(booking, "time_confirmed", None)
    if isinstance(time_confirmed, bool):
        tc: bool | None = time_confirmed
    else:
        tc = None
    return not is_legacy_midnight_pickup_sentinel(scheduled_dt, time_confirmed=tc)


def booking_has_confirmed_pickup_time(booking: Any) -> bool:
    """INV-2 : heure confirmée workflow (= time_defined côté API)."""
    time_confirmed = getattr(booking, "time_confirmed", None)
    if time_confirmed is False:
        return False
    if time_confirmed is True:
        return booking_has_scheduled_pickup_time(booking)
    return booking_has_scheduled_pickup_time(booking)


def build_booking_scheduling(booking: Any) -> dict[str, Any]:
    scheduled_dt = getattr(booking, "scheduled_time", None)
    raw_time_confirmed = getattr(booking, "time_confirmed", None)
    time_confirmed = (
        bool(raw_time_confirmed)
        if isinstance(raw_time_confirmed, bool)
        else True
    )
    time_scheduled = booking_has_scheduled_pickup_time(booking)
    time_defined = booking_has_confirmed_pickup_time(booking)

    display_time = "À définir"
    display_datetime = "À définir"
    if scheduled_dt is not None and time_scheduled:
        date_local, time_local = split_date_time_local(scheduled_dt)
        if time_defined and time_local:
            display_time = time_local
            if date_local:
                display_datetime = f"{date_local} • {time_local}"
            else:
                display_datetime = time_local
        elif time_local:
            display_time = f"{time_local} (non confirmé)"
            if date_local:
                display_datetime = f"{date_local} • {display_time}"
            else:
                display_datetime = display_time
        elif date_local:
            display_datetime = date_local

    scheduled_iso = None
    if scheduled_dt is not None:
        from shared.time_utils import iso_utc_z, to_utc_from_db

        scheduled_iso = iso_utc_z(to_utc_from_db(scheduled_dt))

    return {
        "scheduled_time": scheduled_iso,
        "time_confirmed": time_confirmed,
        "time_scheduled": time_scheduled,
        "time_defined": time_defined,
        "display_time": display_time,
        "display_datetime": display_datetime,
    }


def _route_group_leg_count(booking: Any) -> int:
    cached = getattr(booking, "_route_group_leg_count", None)
    if cached is not None:
        try:
            return max(1, int(cached))
        except (TypeError, ValueError):
            pass
    group_id = getattr(booking, "route_group_id", None)
    if not group_id:
        return 1
    try:
        from models.booking import Booking

        count = (
            Booking.query.filter(Booking.route_group_id == group_id).count()
        )
        return max(1, int(count))
    except Exception:
        return 1


def build_booking_trip_flags(
    booking: Any,
    viewer_company_id: int | None = None,
) -> dict[str, Any]:
    is_return = bool(getattr(booking, "is_return", False))
    is_round_trip = bool(getattr(booking, "is_round_trip", False))
    has_return = getattr(booking, "return_trip", None) is not None
    round_trip = (not is_return) and (is_round_trip or has_return)

    leg_count = _route_group_leg_count(booking)
    route_group_id = getattr(booking, "route_group_id", None)
    leg_number = getattr(booking, "route_sequence_number", None)
    multi_stop = bool(route_group_id and leg_count > 1)

    transferred = False
    fn = getattr(booking, "_is_transferred", None)
    if callable(fn):
        transferred = bool(fn())
    elif getattr(booking, "executing_company_id", None) and getattr(
        booking, "company_id", None
    ):
        transferred = int(booking.executing_company_id) != int(booking.company_id)

    change_pending = False
    acr = getattr(booking, "active_change_request", None)
    if acr is not None:
        st = getattr(acr, "status", None)
        st_val = st.value if hasattr(st, "value") else str(st or "").lower()
        change_pending = st_val in _CHANGE_REQUEST_PENDING_STATUSES

    return {
        "round_trip": round_trip,
        "return_leg": is_return,
        "multi_stop": multi_stop,
        "leg_number": int(leg_number) if leg_number is not None else None,
        "leg_count": leg_count if multi_stop else (1 if route_group_id else None),
        "transferred": transferred,
        "change_request_pending": change_pending,
    }


def build_booking_search_index(identity: dict[str, Any]) -> list[str]:
    tokens: list[str] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text not in tokens:
            tokens.append(text)

    _add((identity.get("passenger") or {}).get("name"))
    source = identity.get("source") or {}
    _add(source.get("name"))
    _add(source.get("code"))
    upstream = identity.get("upstream")
    if isinstance(upstream, dict):
        _add(upstream.get("name"))
        _add(upstream.get("code"))
    requester = identity.get("requester")
    if isinstance(requester, dict):
        _add(requester.get("name"))
    ownership = identity.get("ownership") or {}
    _add(ownership.get("owner_company_name"))
    execution = identity.get("execution") or {}
    _add(execution.get("executing_company_name"))

    return tokens


def build_booking_display_blocks(
    booking: Any,
    viewer_company_id: int | None = None,
) -> dict[str, Any]:
    identity = build_booking_identity(booking, viewer_company_id)
    return {
        "display_model": DISPLAY_MODEL_BOOKING,
        "display_model_version": DISPLAY_MODEL_VERSION,
        "identity": identity,
        "trip_flags": build_booking_trip_flags(booking, viewer_company_id),
        "scheduling": build_booking_scheduling(booking),
        "search_index": build_booking_search_index(identity),
    }
