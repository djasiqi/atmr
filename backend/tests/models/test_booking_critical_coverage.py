"""Couverture critique de ``models.booking`` (seuil 80 %)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm.attributes import set_committed_value

from models.booking import (
    AMOUNT_MINIMUM,
    AMOUNT_ROUNDING_TARGET_1,
    AMOUNT_ROUNDING_TARGET_2,
    AMOUNT_ROUNDING_TARGET_3,
    CUSTOMER_NAME_MAX_LENGTH,
    LOCATION_MAX_LENGTH,
    Booking,
    _created_via_value,
)
from models.enums import BookingCreatedVia, BookingStatus
from shared.time_utils import now_local


@pytest.fixture
def booking(app):
    with app.app_context():
        yield Booking()


def _gender(value: str) -> SimpleNamespace:
    return SimpleNamespace(value=value)


# ---------------------------------------------------------------------------
# Helpers purs
# ---------------------------------------------------------------------------


def test_created_via_value_none_enum_et_str():
    assert _created_via_value(SimpleNamespace(created_via=None)) == (
        BookingCreatedVia.LEGACY.value
    )
    assert (
        _created_via_value(SimpleNamespace(created_via=BookingCreatedVia.CLIENT_APP))
        == "client_app"
    )
    assert _created_via_value(SimpleNamespace(created_via="DISPATCHER")) == "dispatcher"


def test_customer_full_name_branches():
    assert (
        Booking.customer_full_name.fget(SimpleNamespace(customer_name="Jean Dupont"))
        == "Jean Dupont"
    )
    user = SimpleNamespace(first_name="Ada", last_name="Lovelace", username="ada")
    assert (
        Booking.customer_full_name.fget(
            SimpleNamespace(customer_name="", client=SimpleNamespace(user=user))
        )
        == "Ada Lovelace"
    )
    user_username = SimpleNamespace(first_name="", last_name="", username="ada")
    assert (
        Booking.customer_full_name.fget(
            SimpleNamespace(
                customer_name=None, client=SimpleNamespace(user=user_username)
            )
        )
        == "ada"
    )
    assert (
        Booking.customer_full_name.fget(SimpleNamespace(customer_name="", client=None))
        == "Non spécifié"
    )


def test_get_effective_payer_company_client_et_fallback():
    company = SimpleNamespace(
        id=7,
        name="Clinique",
        address="Rue 1",
        contact_email="c@test.ch",
        contact_phone="021",
    )
    billed = Booking.get_effective_payer(
        SimpleNamespace(
            billed_to_type="clinic",
            billed_to_company=company,
            client=None,
            customer_name="X",
        )
    )
    assert billed["type"] == "clinic"
    assert billed["company_id"] == 7
    assert billed["name"] == "Clinique"

    user = SimpleNamespace(
        first_name="Ada",
        last_name="Lovelace",
        username="ada",
        email="a@test.ch",
        phone="079",
    )
    client = SimpleNamespace(
        id=3,
        user=user,
        billing_address="Facture 1",
        contact_email="cli@test.ch",
        contact_phone="022",
    )
    patient = Booking.get_effective_payer(
        SimpleNamespace(
            billed_to_type="patient",
            billed_to_company=None,
            client=client,
            customer_name="X",
        )
    )
    assert patient["type"] == "patient"
    assert patient["client_id"] == 3
    assert patient["email"] == "cli@test.ch"

    user_vide = SimpleNamespace(
        first_name="", last_name="", username="login", email=None, phone=None
    )
    via_username = Booking.get_effective_payer(
        SimpleNamespace(
            billed_to_type="patient",
            billed_to_company=None,
            client=SimpleNamespace(
                id=4,
                user=user_vide,
                billing_address=None,
                contact_email=None,
                contact_phone=None,
            ),
            customer_name="X",
        )
    )
    assert via_username["name"] == "login"
    assert via_username["email"] == user_vide.email

    fallback = Booking.get_effective_payer(
        SimpleNamespace(
            billed_to_type="patient",
            billed_to_company=None,
            client=None,
            customer_name="Invité",
            customer_full_name="Invité",
        )
    )
    assert fallback == {"type": "patient", "name": "Invité"}


# ---------------------------------------------------------------------------
# Dashboard / paiement / transfert
# ---------------------------------------------------------------------------


def _dashboard_ns(**overrides):
    data = {
        "id": 11,
        "customer_name": "Ada",
        "pickup_location": "A",
        "dropoff_location": "B",
        "pickup_lat": 46.2,
        "pickup_lon": 6.1,
        "dropoff_lat": 46.3,
        "dropoff_lon": 6.2,
        "scheduled_time": None,
        "status": BookingStatus.PENDING,
        "company_id": 1,
        "company": SimpleNamespace(name="Cie A"),
        "executing_company_id": 2,
        "executing_company": SimpleNamespace(name="Cie B"),
        "driver_id": None,
        "driver": None,
        "client": None,
        "is_return": False,
        "is_round_trip": False,
        "parent_booking_id": None,
        "return_trip": None,
        "time_confirmed": True,
        "created_via": None,
        "booking_type": None,
        "mission_type": None,
        "wheelchair_need": False,
        "amount": 12.5,
        "billed_to_type": None,
        "billed_to_company_id": None,
        "route_group_id": None,
        "route_sequence_number": None,
        "active_change_request_id": None,
        "active_change_request": None,
        "customer_full_name": "Ada",
        "driver_display_name": None,
        "_get_institution_timeline": lambda: None,
        "_get_route_journey": lambda: None,
        "_canonical_display_payload": lambda: {"identity": {"id": 11}},
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_serialize_dashboard_cache_et_sans_horaire():
    cached = _dashboard_ns(
        _transfer_cache={
            "is_transferred": True,
            "active_transfer": {"id": 99},
        }
    )
    payload = Booking.serialize_dashboard.fget(cached)
    assert payload["is_transferred"] is True
    assert payload["active_transfer"]["id"] == 99
    assert payload["date_formatted"] == "Non spécifié"
    assert payload["created_via"] == BookingCreatedVia.LEGACY.value

    change = SimpleNamespace(serialize=lambda: {"id": 4})
    client = SimpleNamespace(
        id=8,
        is_institution=True,
        institution_name="LHA",
        client_type=SimpleNamespace(value="transport"),
        linked_institution_id=251,
    )
    driver = SimpleNamespace(id=3)
    without_cache = _dashboard_ns(
        scheduled_time=datetime(2026, 8, 20, 14, 30, 0),
        client=client,
        driver=driver,
        driver_id=3,
        driver_display_name="Chauffeur",
        active_change_request=change,
        active_change_request_id=4,
    )
    dash = Booking.serialize_dashboard.fget(without_cache)
    assert dash["is_transferred"] is True
    assert dash["client"]["is_institution"] is True
    assert dash["client"]["client_type"] == "transport"
    assert dash["driver"]["id"] == 3
    assert dash["active_change_request"]["id"] == 4


def test_online_client_payment_brief_providers_et_erreur():
    safer = SimpleNamespace(
        id=3,
        status=SimpleNamespace(value="PENDING"),
        payment_provider="saferpay",
        saferpay_token="tok",
        saferpay_transaction_id=None,
        worldline_hosted_checkout_id=None,
    )
    brief = Booking._online_client_payment_brief(
        SimpleNamespace(payments=[safer], id=1)
    )
    assert brief is not None
    assert brief["has_pending_session"] is True
    assert brief["provider"] == "saferpay"

    worldline = SimpleNamespace(
        id=4,
        status="pending",
        payment_provider="worldline",
        saferpay_token=None,
        saferpay_transaction_id=None,
        worldline_hosted_checkout_id="hc-1",
    )
    wl = Booking._online_client_payment_brief(
        SimpleNamespace(payments=[worldline], id=1)
    )
    assert wl is not None
    assert wl["has_pending_session"] is True

    completed = SimpleNamespace(
        id=1,
        status="COMPLETED",
        payment_provider="saferpay",
        saferpay_token=None,
        saferpay_transaction_id=None,
        worldline_hosted_checkout_id=None,
    )
    done = Booking._online_client_payment_brief(
        SimpleNamespace(payments=[completed], id=1)
    )
    assert done is not None
    assert done["has_pending_session"] is False

    assert (
        Booking._online_client_payment_brief(SimpleNamespace(payments=[], id=1)) is None
    )
    assert (
        Booking._online_client_payment_brief(SimpleNamespace(payments=object(), id=1))
        is None
    )


def test_is_transferred_et_active_transfer_cache_et_erreur(monkeypatch):
    cached = SimpleNamespace(
        _transfer_cache={"is_transferred": True, "active_transfer": {"id": 2}}
    )
    assert Booking._is_transferred(cached) is True
    assert Booking._get_active_transfer_info(cached) == {"id": 2}

    boom = SimpleNamespace(
        id=1,
        company_id=1,
        executing_company_id=9,
        _transfer_cache=None,
    )

    class _BoomQuery:
        def filter_by(self, **_kwargs):
            raise RuntimeError("db down")

    fake_mod = SimpleNamespace(BookingTransfer=SimpleNamespace(query=_BoomQuery()))
    monkeypatch.setitem(__import__("sys").modules, "models.booking_transfer", fake_mod)
    assert Booking._is_transferred(boom) is True
    assert Booking._get_active_transfer_info(boom) is None


# ---------------------------------------------------------------------------
# Institution / parcours
# ---------------------------------------------------------------------------


def test_resolve_source_transport_request_branches(monkeypatch):
    req = SimpleNamespace(id=1)
    assert (
        Booking._resolve_source_transport_request(SimpleNamespace(source_request=req))
        is req
    )
    assert (
        Booking._resolve_source_transport_request(
            SimpleNamespace(source_request=[req, SimpleNamespace(id=2)])
        )
        is req
    )

    parent = SimpleNamespace(source_request=req)
    via_parent = Booking._resolve_source_transport_request(
        SimpleNamespace(
            source_request=None,
            is_return=True,
            parent_booking_id=8,
            return_trip=parent,
            route_group_id=None,
        )
    )
    assert via_parent is req

    assert (
        Booking._resolve_source_transport_request(
            SimpleNamespace(
                source_request=None,
                is_return=False,
                parent_booking_id=None,
                route_group_id=None,
            )
        )
        is None
    )

    fake_tr = MagicMock()
    fake_tr.query.filter_by.return_value.first.return_value = req
    monkeypatch.setattr("models.transport_request.TransportRequest", fake_tr)
    via_group = Booking._resolve_source_transport_request(
        SimpleNamespace(
            source_request=None,
            is_return=False,
            parent_booking_id=None,
            route_group_id="grp-1",
        )
    )
    assert via_group is req

    class _Boom:
        def __getattribute__(self, _name):
            raise RuntimeError("source_request inaccessible")

    assert Booking._resolve_source_transport_request(_Boom()) is None


def test_institution_passenger_leg_et_timeline():
    patient = SimpleNamespace(
        id=9,
        first_name="Eve",
        last_name="Patient",
        dob=date(1990, 1, 2),
        gender=_gender("F"),
        external_reference="EXT-1",
        phone="079",
    )
    req = SimpleNamespace(patient=patient)
    ns = SimpleNamespace(_resolve_source_transport_request=lambda: req)
    brief = Booking._get_institution_passenger_brief(ns)
    assert brief is not None
    assert brief["birth_date"] == "1990-01-02"
    assert brief["gender"] == "F"

    ns_str = SimpleNamespace(
        _resolve_source_transport_request=lambda: SimpleNamespace(
            patient=SimpleNamespace(
                id=1,
                first_name="A",
                last_name="B",
                dob=None,
                gender="M",
                external_reference=None,
                phone=None,
            )
        )
    )
    assert Booking._get_institution_passenger_brief(ns_str)["gender"] == "M"

    assert (
        Booking._get_institution_passenger_brief(
            SimpleNamespace(_resolve_source_transport_request=lambda: None)
        )
        is None
    )
    assert (
        Booking._get_institution_passenger_brief(
            SimpleNamespace(
                _resolve_source_transport_request=lambda: SimpleNamespace(patient=None)
            )
        )
        is None
    )
    assert (
        Booking._get_institution_passenger_brief(
            SimpleNamespace(
                _resolve_source_transport_request=lambda: (_ for _ in ()).throw(
                    RuntimeError("boom")
                )
            )
        )
        is None
    )

    leg_match = SimpleNamespace(
        sequence_index=2,
        route_sequence_number=2,
        dropoff_establishment="HUG",
        dropoff_service="Radio",
        dropoff_doctor="Dr X",
        scheduled_time=datetime(2026, 8, 20, 9, 0, 0),
        time_confirmed=True,
    )
    leg_other = SimpleNamespace(
        sequence_index=1,
        route_sequence_number=1,
        dropoff_establishment="Autre",
        dropoff_service=None,
        dropoff_doctor=None,
        scheduled_time=None,
        time_confirmed=False,
    )
    clinical = Booking._get_institution_leg_clinical_brief(
        SimpleNamespace(
            route_sequence_number=2,
            _resolve_source_transport_request=lambda: SimpleNamespace(
                legs=[leg_other, leg_match]
            ),
        )
    )
    assert clinical is not None
    assert clinical["establishment"] == "HUG"
    assert clinical["appointment_time"] is not None

    first_leg = Booking._get_institution_leg_clinical_brief(
        SimpleNamespace(
            route_sequence_number=None,
            _resolve_source_transport_request=lambda: SimpleNamespace(legs=[leg_other]),
        )
    )
    assert first_leg is not None
    assert first_leg["establishment"] == "Autre"

    assert (
        Booking._get_institution_leg_clinical_brief(
            SimpleNamespace(
                route_sequence_number=1,
                _resolve_source_transport_request=lambda: SimpleNamespace(legs=[]),
            )
        )
        is None
    )
    assert (
        Booking._get_institution_leg_clinical_brief(
            SimpleNamespace(
                _resolve_source_transport_request=lambda: (_ for _ in ()).throw(
                    RuntimeError("boom")
                )
            )
        )
        is None
    )

    created = datetime(2026, 8, 1, 10, 0, 0)
    timeline_req = SimpleNamespace(
        institution=SimpleNamespace(name="LHA"),
        accepted_by_company=SimpleNamespace(name="ATMR"),
        created_by=SimpleNamespace(first_name="Ann", last_name="Dispe"),
        created_at=created,
        sent_at=created,
        accepted_at=created,
        converted_at=created,
        cancelled_at=None,
    )
    timeline = Booking._get_institution_timeline(
        SimpleNamespace(_resolve_source_transport_request=lambda: timeline_req)
    )
    assert timeline is not None
    assert timeline["institution_name"] == "LHA"
    assert timeline["created_by_name"] == "Ann Dispe"
    assert timeline["cancelled_at"] is None

    assert (
        Booking._get_institution_timeline(
            SimpleNamespace(_resolve_source_transport_request=lambda: None)
        )
        is None
    )
    creator_vide = SimpleNamespace(
        institution=None,
        accepted_by_company=None,
        created_by=SimpleNamespace(first_name="", last_name=""),
        created_at=None,
        sent_at=None,
        accepted_at=None,
        converted_at=None,
        cancelled_at=None,
    )
    empty_tl = Booking._get_institution_timeline(
        SimpleNamespace(_resolve_source_transport_request=lambda: creator_vide)
    )
    assert empty_tl is not None
    assert empty_tl["created_by_name"] is None
    assert (
        Booking._get_institution_timeline(
            SimpleNamespace(
                _resolve_source_transport_request=lambda: (_ for _ in ()).throw(
                    RuntimeError("boom")
                )
            )
        )
        is None
    )


def test_route_journey_simple_et_multi_legs():
    boarded = datetime(2026, 8, 13, 10, 0, 0)
    completed = datetime(2026, 8, 13, 10, 40, 0)
    simple = SimpleNamespace(
        route_group_id=None,
        parent_booking_id=None,
        is_round_trip=False,
        boarded_at=boarded,
        completed_at=completed,
        route_sequence_number=None,
    )
    events = Booking._get_route_journey(simple)
    assert events is not None
    assert len(events) == 2
    assert events[0]["type"] == "pickup"

    leg1 = SimpleNamespace(
        route_sequence_number=1, boarded_at=boarded, completed_at=completed
    )
    ret = SimpleNamespace(
        route_sequence_number=1, boarded_at=boarded, completed_at=completed
    )
    leg2 = SimpleNamespace(
        route_sequence_number=2, boarded_at=None, completed_at=completed
    )
    multi = SimpleNamespace(
        route_group_id="g1",
        parent_booking_id=None,
        is_round_trip=True,
        _collect_journey_legs=lambda: [(leg1, False), (ret, True), (leg2, False)],
    )
    multi_events = Booking._get_route_journey(multi)
    assert multi_events is not None
    labels = [e["event"] for e in multi_events]
    assert any("retour" in e.lower() for e in labels)
    assert any("Trajet 2" in e for e in labels)

    only_return = SimpleNamespace(
        route_group_id="g1",
        parent_booking_id=9,
        is_round_trip=False,
        _collect_journey_legs=lambda: [
            (
                SimpleNamespace(
                    route_sequence_number=None,
                    boarded_at=boarded,
                    completed_at=None,
                ),
                True,
            )
        ],
    )
    ret_events = Booking._get_route_journey(only_return)
    assert ret_events is not None
    assert "Retour" in ret_events[0]["event"]

    broken = SimpleNamespace(
        route_group_id=None,
        parent_booking_id=None,
        is_round_trip=False,
        boarded_at=object(),
        completed_at=None,
        route_sequence_number=None,
    )
    assert Booking._get_route_journey(broken) is None


def test_collect_journey_legs_parent_et_groupe(booking, monkeypatch):
    parent = SimpleNamespace(id=10, route_group_id=None)
    retour = SimpleNamespace(id=11)

    class _Query:
        def get(self, pk):
            return parent if pk == 10 else None

        def filter_by(self, **kwargs):
            m = MagicMock()
            if kwargs.get("parent_booking_id") == 10:
                m.order_by.return_value.all.return_value = [retour]
            else:
                m.order_by.return_value.all.return_value = []
            return m

    monkeypatch.setattr(Booking, "query", _Query())
    set_committed_value(booking, "is_return", True)
    set_committed_value(booking, "parent_booking_id", 10)
    pairs = booking._collect_journey_legs()
    assert pairs[0][0] is parent
    assert pairs[1][0] is retour
    assert pairs[1][1] is True

    grouped_leg = SimpleNamespace(id=20, route_group_id="grp")

    class _GroupQuery:
        def get(self, _pk):
            return None

        def filter_by(self, **kwargs):
            m = MagicMock()
            if "route_group_id" in kwargs:
                m.order_by.return_value.all.return_value = [grouped_leg]
            else:
                m.order_by.return_value.all.return_value = []
            return m

    monkeypatch.setattr(Booking, "query", _GroupQuery())
    outbound = Booking()
    set_committed_value(outbound, "is_return", False)
    set_committed_value(outbound, "parent_booking_id", None)
    set_committed_value(outbound, "route_group_id", "grp")
    group_pairs = outbound._collect_journey_legs()
    assert group_pairs[0][0] is grouped_leg


# ---------------------------------------------------------------------------
# Validateurs
# ---------------------------------------------------------------------------


def test_validate_user_id(booking):
    assert booking.validate_user_id("user_id", None) is None
    assert booking.validate_user_id("user_id", 12) == 12
    with pytest.raises(ValueError, match="entier positif"):
        booking.validate_user_id("user_id", 0)
    with pytest.raises(ValueError, match="entier positif"):
        booking.validate_user_id("user_id", -1)
    with pytest.raises(ValueError, match="entier positif"):
        booking.validate_user_id("user_id", "x")


def test_validate_is_return_et_amount(booking):
    assert booking.validate_is_return("is_return", 1) is True
    assert booking.validate_is_return("is_return", None) is False
    assert booking.validate_amount("amount", None) is None
    set_committed_value(booking, "is_return", True)
    assert booking.validate_amount("amount", 0) == 0.0
    set_committed_value(booking, "is_return", False)
    with pytest.raises(ValueError, match="montant minimum"):
        booking.validate_amount("amount", 0.1)
    assert booking.validate_amount("amount", 0.55) == AMOUNT_ROUNDING_TARGET_1
    assert booking.validate_amount("amount", 0.77) == AMOUNT_ROUNDING_TARGET_2
    assert booking.validate_amount("amount", 39.99) == AMOUNT_ROUNDING_TARGET_3
    assert booking.validate_amount("amount", 12.345) == 12.35
    assert booking.validate_amount("amount", AMOUNT_MINIMUM) == AMOUNT_ROUNDING_TARGET_1


def test_validate_scheduled_time(booking, monkeypatch):
    set_committed_value(booking, "is_return", True)
    set_committed_value(booking, "time_confirmed", True)
    assert booking.validate_scheduled_time("scheduled_time", None) is None

    set_committed_value(booking, "is_return", False)
    set_committed_value(booking, "time_confirmed", False)
    assert booking.validate_scheduled_time("scheduled_time", None) is None

    set_committed_value(booking, "time_confirmed", True)
    with pytest.raises(ValueError, match="obligatoire"):
        booking.validate_scheduled_time("scheduled_time", None)

    past = datetime(2020, 1, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="passé"):
        booking.validate_scheduled_time("scheduled_time", past)

    sentinel = datetime(2020, 1, 1, 0, 0, 0)
    assert booking.validate_scheduled_time("scheduled_time", sentinel) == sentinel

    set_committed_value(booking, "time_confirmed", False)
    assert booking.validate_scheduled_time("scheduled_time", past) == past

    future = now_local() + timedelta(days=2)
    naive_future = future.replace(tzinfo=None) if future.tzinfo else future
    set_committed_value(booking, "time_confirmed", True)
    assert (
        booking.validate_scheduled_time("scheduled_time", naive_future) == naive_future
    )

    monkeypatch.setattr(
        "models.booking.api_scheduled_iso_to_naive_geneva", lambda _v: None
    )
    with pytest.raises(ValueError, match="invalide"):
        booking.validate_scheduled_time("scheduled_time", "x")


def test_validate_customer_name_et_location(booking):
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        booking.validate_customer_name("customer_name", "")
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        booking.validate_customer_name("customer_name", "   ")
    too_long = "x" * (CUSTOMER_NAME_MAX_LENGTH + 1)
    with pytest.raises(ValueError, match="ne peut pas dépasser"):
        booking.validate_customer_name("customer_name", too_long)
    assert booking.validate_customer_name("customer_name", "Ada") == "Ada"

    with pytest.raises(ValueError, match="ne peut pas être vide"):
        booking.validate_location("pickup_location", "")
    loc_too_long = "y" * (LOCATION_MAX_LENGTH + 1)
    with pytest.raises(ValueError, match="ne peut pas dépasser"):
        booking.validate_location("dropoff_location", loc_too_long)
    assert booking.validate_location("pickup_location", "Gare") == "Gare"


def test_validate_status_et_driver_id(booking):
    assert booking.validate_status("status", "pending") == BookingStatus.PENDING
    with pytest.raises(ValueError, match="Statut invalide"):
        booking.validate_status("status", "NOT_A_STATUS")
    with pytest.raises(ValueError, match="BookingStatus"):
        booking.validate_status("status", 123)
    set_committed_value(booking, "driver_id", None)
    with pytest.raises(ValueError, match="nécessite un driver_id"):
        booking.validate_status("status", BookingStatus.ASSIGNED)
    set_committed_value(booking, "driver_id", 4)
    assert (
        booking.validate_status("status", BookingStatus.ASSIGNED)
        == BookingStatus.ASSIGNED
    )

    with pytest.raises(ValueError, match="entier positif"):
        booking.validate_driver_id("driver_id", -2)
    with pytest.raises(ValueError, match="entier positif"):
        booking.validate_driver_id("driver_id", "x")
    set_committed_value(booking, "status", BookingStatus.ASSIGNED)
    with pytest.raises(ValueError, match="ne peut pas être NULL"):
        booking.validate_driver_id("driver_id", None)
    set_committed_value(booking, "status", BookingStatus.PENDING)
    assert booking.validate_driver_id("driver_id", None) is None
    assert booking.validate_driver_id("driver_id", 8) == 8


def test_validate_billed_to(booking):
    assert booking._v_billed_to_type("billed_to_type", None) == "patient"
    assert booking._v_billed_to_type("billed_to_type", " CLINIC ") == "clinic"
    with pytest.raises(ValueError, match="billed_to_type invalide"):
        booking._v_billed_to_type("billed_to_type", "iban")

    with pytest.raises(ValueError, match="entier positif"):
        booking._v_billed_to_company_id("billed_to_company_id", 0)
    with pytest.raises(ValueError, match="entier positif"):
        booking._v_billed_to_company_id("billed_to_company_id", "x")
    set_committed_value(booking, "billed_to_type", "patient")
    assert booking._v_billed_to_company_id("billed_to_company_id", 9) is None
    set_committed_value(booking, "billed_to_type", "clinic")
    assert booking._v_billed_to_company_id("billed_to_company_id", 9) == 9
    assert booking._v_billed_to_company_id("billed_to_company_id", None) is None


# ---------------------------------------------------------------------------
# State machine / métier
# ---------------------------------------------------------------------------


def test_validate_status_transition_autorisees_et_refusees(booking):
    set_committed_value(booking, "status", BookingStatus.PENDING)
    set_committed_value(booking, "driver_id", None)
    ok, err = booking.validate_status_transition(BookingStatus.ACCEPTED)
    assert ok is True
    assert err is None

    bad, msg = booking.validate_status_transition(BookingStatus.ASSIGNED)
    assert bad is False
    assert msg is not None
    assert "driver_id" in msg

    set_committed_value(booking, "driver_id", 5)
    ok_assigned, _ = booking.validate_status_transition(BookingStatus.ASSIGNED)
    assert ok_assigned is True

    set_committed_value(booking, "status", BookingStatus.ASSIGNED)
    set_committed_value(booking, "driver_id", None)
    en_route_ko, en_route_msg = booking.validate_status_transition(
        BookingStatus.EN_ROUTE
    )
    assert en_route_ko is False
    assert en_route_msg is not None
    assert "EN_ROUTE" in en_route_msg

    set_committed_value(booking, "status", BookingStatus.EN_ROUTE)
    in_progress_ko, in_progress_msg = booking.validate_status_transition(
        BookingStatus.IN_PROGRESS
    )
    assert in_progress_ko is False
    assert in_progress_msg is not None
    assert "IN_PROGRESS" in in_progress_msg

    set_committed_value(booking, "status", BookingStatus.IN_PROGRESS)
    completed_ko, completed_msg = booking.validate_status_transition(
        BookingStatus.COMPLETED
    )
    assert completed_ko is False
    assert completed_msg is not None
    assert "compléter" in completed_msg

    return_ko, _ = booking.validate_status_transition(BookingStatus.RETURN_COMPLETED)
    assert return_ko is False

    set_committed_value(booking, "driver_id", 5)
    completed_ok, _ = booking.validate_status_transition(BookingStatus.COMPLETED)
    assert completed_ok is True

    set_committed_value(booking, "status", BookingStatus.AWAITING_CLIENT_PAYMENT)
    pay_ok, _ = booking.validate_status_transition(BookingStatus.PENDING)
    assert pay_ok is True
    cancel_from_pay, _ = booking.validate_status_transition(BookingStatus.CANCELED)
    assert cancel_from_pay is True

    set_committed_value(booking, "status", BookingStatus.ACCEPTED)
    acc_cancel, _ = booking.validate_status_transition(BookingStatus.CANCELED)
    assert acc_cancel is True

    set_committed_value(booking, "status", BookingStatus.COMPLETED)
    term, term_msg = booking.validate_status_transition(BookingStatus.PENDING)
    assert term is False
    assert term_msg is not None
    assert "Transition invalide" in term_msg

    set_committed_value(booking, "status", BookingStatus.RETURN_COMPLETED)
    ret_term, _ = booking.validate_status_transition(BookingStatus.CANCELED)
    assert ret_term is False

    set_committed_value(booking, "status", BookingStatus.CANCELED)
    can_term, _ = booking.validate_status_transition(BookingStatus.PENDING)
    assert can_term is False


def test_update_status_duration_assign_cancel(booking):
    set_committed_value(booking, "status", BookingStatus.PENDING)
    set_committed_value(booking, "driver_id", None)
    with pytest.raises(ValueError, match="Transition invalide"):
        booking.update_status(BookingStatus.EN_ROUTE)
    booking.update_status(BookingStatus.ACCEPTED)
    assert booking.status == BookingStatus.ACCEPTED

    assert (
        Booking.duration_in_minutes.fget(
            SimpleNamespace(boarded_at=None, completed_at=None)
        )
        is None
    )
    start = datetime(2026, 8, 13, 10, 0, 0)
    end = datetime(2026, 8, 13, 10, 45, 0)
    assert (
        Booking.duration_in_minutes.fget(
            SimpleNamespace(boarded_at=start, completed_at=end)
        )
        == 45
    )

    future = now_local() + timedelta(days=1)
    naive_future = future.replace(tzinfo=None) if future.tzinfo else future
    set_committed_value(booking, "status", BookingStatus.PENDING)
    set_committed_value(booking, "scheduled_time", naive_future)
    set_committed_value(booking, "driver_id", 7)
    assert booking.is_assignable() is True
    booking.assign_driver(7)
    assert booking.driver_id == 7
    assert booking.status == BookingStatus.PENDING
    set_committed_value(booking, "driver_id", None)
    booking.assign_driver(7)
    assert booking.driver_id == 7
    assert booking.status == BookingStatus.ASSIGNED

    set_committed_value(booking, "status", BookingStatus.COMPLETED)
    set_committed_value(booking, "scheduled_time", now_local() + timedelta(days=1))
    with pytest.raises(ValueError, match="ne peut pas être attribuée"):
        booking.assign_driver(8)

    set_committed_value(booking, "status", BookingStatus.PENDING)
    with pytest.raises(ValueError, match="en cours"):
        booking.cancel_booking()
    set_committed_value(booking, "driver_id", 7)
    set_committed_value(booking, "status", BookingStatus.ASSIGNED)
    booking.cancel_booking()
    assert booking.status == BookingStatus.CANCELED


def test_enforce_billing_exclusive():
    patient = SimpleNamespace(billed_to_type="patient", billed_to_company_id=44)
    Booking._enforce_billing_exclusive(None, None, patient)
    assert patient.billed_to_company_id is None

    clinic = SimpleNamespace(billed_to_type="clinic", billed_to_company_id=3)
    Booking._enforce_billing_exclusive(None, None, clinic)

    with pytest.raises(ValueError, match="obligatoire"):
        Booking._enforce_billing_exclusive(
            None,
            None,
            SimpleNamespace(billed_to_type="clinic", billed_to_company_id=None),
        )
    with pytest.raises(ValueError, match="obligatoire"):
        Booking._enforce_billing_exclusive(
            None,
            None,
            SimpleNamespace(billed_to_type="insurance", billed_to_company_id=0),
        )


def test_to_dict_et_is_future(booking, monkeypatch):
    monkeypatch.setattr(
        Booking,
        "serialize",
        {"id": 1},
        raising=False,
    )
    assert booking.to_dict() == {"id": 1}

    set_committed_value(booking, "scheduled_time", now_local() + timedelta(days=3))
    assert booking.is_future() is True
    set_committed_value(booking, "scheduled_time", datetime(2020, 1, 1, 12, 0, 0))
    assert booking.is_future() is False
    set_committed_value(booking, "scheduled_time", None)
    assert booking.is_future() is False


def test_is_transferred_query_hit(monkeypatch):
    transfer = SimpleNamespace(owner_company_id=2)
    q = MagicMock()
    q.filter_by.return_value.filter.return_value.first.return_value = transfer
    monkeypatch.setattr(
        "models.booking_transfer.BookingTransfer",
        SimpleNamespace(query=q, status=MagicMock()),
        raising=False,
    )
    monkeypatch.setattr(
        "models.enums.TransferStatus",
        SimpleNamespace(ACCEPTED="ACCEPTED", COMPLETED="COMPLETED", PENDING="PENDING"),
        raising=False,
    )
    ns = SimpleNamespace(id=1, company_id=9, _transfer_cache=None)
    assert Booking._is_transferred(ns) is True

    q.filter_by.return_value.filter.return_value.first.return_value = None
    assert Booking._is_transferred(ns) is False

    info_row = MagicMock()
    info_row.to_dict.return_value = {"id": 5}
    q.filter_by.return_value.filter.return_value.first.return_value = info_row
    assert Booking._get_active_transfer_info(ns) == {"id": 5}
    q.filter_by.return_value.filter.return_value.first.return_value = None
    assert Booking._get_active_transfer_info(ns) is None
