"""Tests unitaires — blocs d'affichage canoniques réservation entreprise."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from models.enums import BookingCreatedVia, ClientType
from services.companies.booking_display import (
    DISPLAY_CATEGORY_COMPANY_CLIENT,
    DISPLAY_CATEGORY_INSTITUTION_PATIENT,
    DISPLAY_CATEGORY_LIRIE_GUEST,
    DISPLAY_CATEGORY_PARTNER_CLIENT,
    DISPLAY_MODEL_BOOKING,
    SOURCE_TYPE_COMPANY_CLIENT,
    SOURCE_TYPE_INSTITUTION,
    SOURCE_TYPE_LIRIE_GUEST,
    SOURCE_TYPE_PARTNER,
    booking_has_confirmed_pickup_time,
    booking_has_scheduled_pickup_time,
    build_booking_display_blocks,
    build_booking_scheduling,
    is_legacy_midnight_pickup_sentinel,
    resolve_booking_source,
)


def _booking(**kwargs):
    defaults = {
        "id": 1,
        "customer_name": "Jean Dupont",
        "company_id": 10,
        "executing_company_id": 10,
        "client_id": 5,
        "booking_type": "standard",
        "is_return": False,
        "is_round_trip": False,
        "time_confirmed": True,
        "scheduled_time": None,
        "route_group_id": None,
        "route_sequence_number": None,
        "created_via": BookingCreatedVia.LEGACY,
    }
    defaults.update(kwargs)
    b = SimpleNamespace(**defaults)
    b.customer_full_name = defaults["customer_name"]
    b.client = kwargs.get("client")
    b.company = kwargs.get("company")
    b.executing_company = kwargs.get("executing_company")
    b.return_trip = kwargs.get("return_trip")
    b.active_change_request = kwargs.get("active_change_request")
    b.source_request = kwargs.get("source_request")
    b._get_institution_timeline = kwargs.get(
        "_get_institution_timeline",
        lambda: kwargs.get("institution_timeline"),
    )
    b._get_active_transfer_info = kwargs.get(
        "_get_active_transfer_info",
        lambda: kwargs.get("active_transfer"),
    )
    b._is_transferred = kwargs.get("_is_transferred", lambda: False)
    return b


def test_public_guest_source():
    b = _booking(created_via=BookingCreatedVia.PUBLIC_GUEST, client=None)
    source = resolve_booking_source(b, viewer_company_id=10)
    assert source["type"] == SOURCE_TYPE_LIRIE_GUEST
    assert source["id"] is None
    assert source["code"] == "GUEST"


def test_institution_source():
    client = SimpleNamespace(
        id=5,
        is_institution=True,
        institution_name="Clinique LHA",
        linked_institution_id=251,
        client_type=ClientType.TRANSPORT,
    )
    b = _booking(
        client=client,
        institution_timeline={
            "institution_name": "Clinique LHA",
            "created_by_name": "Marie Martin",
        },
        created_via=BookingCreatedVia.INSTITUTION_PORTAL,
    )
    source = resolve_booking_source(b, viewer_company_id=10)
    assert source["type"] == SOURCE_TYPE_INSTITUTION
    assert source["id"] == 251
    assert source["code"] == "CL"


def test_partner_source_for_executing_viewer():
    client = SimpleNamespace(
        id=5,
        is_institution=True,
        institution_name="HUG",
        linked_institution_id=123,
        client_type=ClientType.TRANSPORT,
    )
    company = SimpleNamespace(id=1, name="Emmenez-Moi")
    b = _booking(
        client=client,
        company_id=1,
        company=company,
        executing_company_id=7,
        institution_timeline={"institution_name": "HUG"},
        created_via=BookingCreatedVia.INSTITUTION_PORTAL,
        active_transfer={
            "owner_company_id": 1,
            "owner_company_name": "Emmenez-Moi",
            "executing_company_id": 7,
            "executing_company_name": "MT Genève",
            "status": "ACCEPTED",
        },
    )
    source = resolve_booking_source(b, viewer_company_id=7)
    assert source["type"] == SOURCE_TYPE_PARTNER
    assert source["id"] == 1
    assert source["name"] == "Emmenez-Moi"

    blocks = build_booking_display_blocks(b, viewer_company_id=7)
    assert blocks["identity"]["upstream"]["type"] == SOURCE_TYPE_INSTITUTION
    assert blocks["identity"]["ownership"]["owner_company_id"] == 1
    assert blocks["identity"]["execution"]["executing_company_id"] == 7


def test_company_client_never_equals_passenger_name():
    client = SimpleNamespace(
        id=99,
        is_institution=False,
        institution_name=None,
        linked_institution_id=None,
        client_type=ClientType.TRANSPORT,
    )
    company = SimpleNamespace(id=10, name="MT Genève")
    b = _booking(
        client=client,
        company=company,
        booking_type="manual",
        created_via=BookingCreatedVia.DISPATCHER,
    )
    source = resolve_booking_source(b, viewer_company_id=10)
    assert source["type"] == SOURCE_TYPE_COMPANY_CLIENT
    assert source["name"] != "Jean Dupont"


def test_scheduling_time_undefined_when_not_confirmed():
    b = _booking(
        scheduled_time=datetime(2026, 6, 12, 0, 0),
        time_confirmed=False,
    )
    scheduling = build_booking_scheduling(b)
    assert scheduling["time_scheduled"] is False
    assert scheduling["time_defined"] is False
    assert scheduling["display_time"] == "À définir"


def test_scheduling_unconfirmed_1330_has_time_scheduled():
    b = _booking(
        scheduled_time=datetime(2026, 6, 12, 13, 30),
        time_confirmed=False,
    )
    scheduling = build_booking_scheduling(b)
    assert scheduling["time_scheduled"] is True
    assert scheduling["time_defined"] is False
    assert scheduling["display_time"] == "13:30 (non confirmé)"
    assert booking_has_scheduled_pickup_time(b) is True
    assert booking_has_confirmed_pickup_time(b) is False


def test_legacy_midnight_sentinel_vs_real_midnight():
    legacy = datetime(2026, 6, 12, 0, 0)
    assert is_legacy_midnight_pickup_sentinel(legacy, time_confirmed=False) is True
    assert is_legacy_midnight_pickup_sentinel(legacy, time_confirmed=True) is False
    assert is_legacy_midnight_pickup_sentinel(datetime(2026, 6, 12, 13, 30)) is False


def test_scheduling_midnight_real_confirmed_bk01c():
    """BK-01c : minuit réel confirmé → 00:00, jamais À définir."""
    b = _booking(
        scheduled_time=datetime(2026, 6, 12, 0, 0),
        time_confirmed=True,
    )
    scheduling = build_booking_scheduling(b)
    assert scheduling["time_scheduled"] is True
    assert scheduling["time_defined"] is True
    assert scheduling["display_time"] == "00:00"
    assert scheduling["display_time"] != "À définir"


def test_scheduling_confirmed_1430_bk01a():
    b = _booking(
        scheduled_time=datetime(2026, 6, 12, 14, 30),
        time_confirmed=True,
    )
    scheduling = build_booking_scheduling(b)
    assert scheduling["time_scheduled"] is True
    assert scheduling["time_defined"] is True
    assert scheduling["display_time"] == "14:30"


def test_scheduling_undefined_null_bk01b():
    b = _booking(scheduled_time=None, time_confirmed=False)
    scheduling = build_booking_scheduling(b)
    assert scheduling["time_scheduled"] is False
    assert scheduling["time_defined"] is False
    assert scheduling["display_time"] == "À définir"


def test_display_model_envelope():
    b = _booking(scheduled_time=datetime(2026, 6, 12, 10, 0))
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    assert blocks["display_model"] == DISPLAY_MODEL_BOOKING
    assert blocks["display_model_version"] == 1


def test_identity_passenger_gender_from_institution_brief():
    b = _booking(
        _get_institution_passenger_brief=lambda: {
            "first_name": "Matsa",
            "last_name": "CHERIF",
            "gender": "FEMME",
            "birth_date": "1973-05-03",
        },
    )
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    assert blocks["identity"]["passenger"]["gender"] == "FEMME"


def test_identity_passenger_gender_from_client_user():
    from models.enums import GenderEnum

    client = SimpleNamespace(
        id=5,
        user=SimpleNamespace(gender=GenderEnum.HOMME),
        client_type=ClientType.TRANSPORT,
    )
    b = _booking(client=client, _get_institution_passenger_brief=lambda: None)
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    assert blocks["identity"]["passenger"]["gender"] == "HOMME"


def test_identity_labels_institution_bk01e():
    client = SimpleNamespace(
        id=5,
        is_institution=True,
        institution_name="Clinique LHA",
        linked_institution_id=251,
        client_type=ClientType.TRANSPORT,
    )
    b = _booking(
        client=client,
        institution_timeline={"institution_name": "Clinique LHA"},
        created_via=BookingCreatedVia.INSTITUTION_PORTAL,
    )
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    identity = blocks["identity"]
    assert identity["display_category"] == DISPLAY_CATEGORY_INSTITUTION_PATIENT
    assert identity["primary_label"] == "Jean Dupont"
    assert identity["secondary_label"] == "Clinique LHA"


def test_identity_labels_lirie_guest_bk01g():
    b = _booking(created_via=BookingCreatedVia.PUBLIC_GUEST, client=None)
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    assert blocks["identity"]["display_category"] == DISPLAY_CATEGORY_LIRIE_GUEST
    assert blocks["identity"]["secondary_label"] == "Invité LIRIE"


def test_identity_labels_partner_bk01f():
    """BK-01f : exécutant voit le partenaire propriétaire."""
    client = SimpleNamespace(
        id=5,
        is_institution=True,
        institution_name="HUG",
        linked_institution_id=123,
        client_type=ClientType.TRANSPORT,
    )
    company = SimpleNamespace(id=1, name="Emmenez-Moi")
    b = _booking(
        client=client,
        company_id=1,
        company=company,
        executing_company_id=10,
        executing_company=SimpleNamespace(id=10, name="MT Genève"),
        active_transfer={
            "owner_company_id": 1,
            "owner_company_name": "Emmenez-Moi",
            "executing_company_id": 10,
        },
        created_via=BookingCreatedVia.INSTITUTION_PORTAL,
    )
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    identity = blocks["identity"]
    assert identity["display_category"] == DISPLAY_CATEGORY_PARTNER_CLIENT
    assert identity["primary_label"] == "Jean Dupont"
    assert identity["secondary_label"] == "Emmenez-Moi"


def test_identity_labels_company_client_bk01h():
    client = SimpleNamespace(
        id=99,
        is_institution=False,
        institution_name=None,
        linked_institution_id=None,
        client_type=ClientType.TRANSPORT,
    )
    b = _booking(
        client=client,
        company=SimpleNamespace(id=10, name="MT Genève"),
        booking_type="manual",
        created_via=BookingCreatedVia.DISPATCHER,
    )
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    assert blocks["identity"]["display_category"] == DISPLAY_CATEGORY_COMPANY_CLIENT
    assert blocks["identity"]["primary_label"] == "Jean Dupont"


def test_search_index_deduplicates():
    client = SimpleNamespace(
        id=5,
        is_institution=True,
        institution_name="HUG",
        linked_institution_id=123,
        client_type=ClientType.TRANSPORT,
    )
    b = _booking(
        client=client,
        company=SimpleNamespace(id=10, name="MT Genève"),
        institution_timeline={
            "institution_name": "HUG",
            "created_by_name": "Marie Martin",
        },
        created_via=BookingCreatedVia.INSTITUTION_PORTAL,
    )
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    index = blocks["search_index"]
    assert "Jean Dupont" in index
    assert "Marie Martin" in index
    assert len(index) == len(set(index))


def test_trip_flags_multi_stop():
    b = _booking(
        route_group_id="grp-1",
        route_sequence_number=2,
    )
    b._route_group_leg_count = 3
    blocks = build_booking_display_blocks(b, viewer_company_id=10)
    flags = blocks["trip_flags"]
    assert flags["multi_stop"] is True
    assert flags["leg_number"] == 2
    assert flags["leg_count"] == 3


def test_trip_flags_return_leg_from_institution_topology():
    outbound = _booking(
        id=38906,
        route_group_id="grp-4464",
        route_sequence_number=1,
        is_return=False,
    )
    outbound._is_return_leg_from_topology = False
    return_booking = _booking(
        id=38907,
        route_group_id="grp-4464",
        route_sequence_number=2,
        is_return=False,
    )
    return_booking._is_return_leg_from_topology = True

    outbound_flags = build_booking_display_blocks(outbound, viewer_company_id=10)[
        "trip_flags"
    ]
    return_flags = build_booking_display_blocks(return_booking, viewer_company_id=10)[
        "trip_flags"
    ]

    assert outbound_flags["return_leg"] is False
    assert return_flags["return_leg"] is True


def test_trip_flags_classic_return_without_topology():
    classic = _booking(is_return=True)
    flags = build_booking_display_blocks(classic, viewer_company_id=10)["trip_flags"]
    assert flags["return_leg"] is True
