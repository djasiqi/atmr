"""Tests pour resolve_invoice_booking_units (A/R strict, pas de fusion cross-sujet)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from application.invoices.invoice_booking_units import resolve_invoice_booking_units
from application.invoices.subject_identity import resolve_subject_identity


def _bk(
    bid: int,
    *,
    client_id: int = 23,
    institution_patient_id: int | None = None,
    amount: str = "80.00",
    pickup: str = "Rue A 1, 1200 Genève",
    dropoff: str = "Rue B 2, 1227 Carouge",
    scheduled: datetime | None = None,
    parent_booking_id: int | None = None,
    is_return: bool = False,
    route_group_id: str | None = None,
    billing_party_id: int | None = 901,
    billed_to_type: str = "patient",
    created_via: str | None = None,
    is_institution_client: bool = False,
) -> SimpleNamespace:
    client = None
    if is_institution_client:
        client = SimpleNamespace(is_institution=True, linked_institution_id=1)
    elif institution_patient_id is None and created_via is None:
        client = SimpleNamespace(is_institution=False, linked_institution_id=None)
    return SimpleNamespace(
        id=bid,
        client_id=client_id,
        institution_patient_id=institution_patient_id,
        amount=Decimal(amount),
        pickup_location=pickup,
        dropoff_location=dropoff,
        scheduled_time=scheduled or datetime(2026, 7, 5, 10, 0, 0),
        parent_booking_id=parent_booking_id,
        is_return=is_return,
        route_group_id=route_group_id,
        billing_party_id=billing_party_id,
        billed_to_type=billed_to_type,
        created_via=created_via,
        client=client,
        status="COMPLETED",
        _resolve_source_transport_request=lambda: None,
    )


def test_parent_child_same_subject_forms_round_trip_unit():
    aller = _bk(100, institution_patient_id=458, amount="80.00")
    retour = _bk(
        101,
        institution_patient_id=458,
        amount="80.00",
        pickup="Rue B 2, 1227 Carouge",
        dropoff="Rue A 1, 1200 Genève",
        scheduled=datetime(2026, 7, 5, 16, 0, 0),
        parent_booking_id=100,
        is_return=True,
    )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=[aller, retour],
        subject_key_fn=lambda b: resolve_subject_identity(b).key,
        amount_ht_fn=lambda b: Decimal(str(b.amount)),
    )
    assert len(units) == 1
    u = units[0]
    assert u.kind == "round_trip"
    assert set(u.booking_ids) == {100, 101}
    assert u.subject_key == "institution_patient:458"
    assert u.amount_ht == Decimal("160.00")
    assert len(u.booking_ids) == 2


def test_two_patients_same_carrier_never_merged():
    """Deux patients sur client_id=23 → 2 unités distinctes, jamais fusionnées."""
    a1 = _bk(1, institution_patient_id=10, amount="40.00")
    a2 = _bk(
        2,
        institution_patient_id=10,
        amount="40.00",
        pickup="Rue B 2, 1227 Carouge",
        dropoff="Rue A 1, 1200 Genève",
        scheduled=datetime(2026, 7, 5, 16, 0, 0),
        parent_booking_id=1,
        is_return=True,
    )
    b1 = _bk(3, institution_patient_id=20, amount="50.00")
    b2 = _bk(
        4,
        institution_patient_id=20,
        amount="50.00",
        pickup="Rue B 2, 1227 Carouge",
        dropoff="Rue A 1, 1200 Genève",
        scheduled=datetime(2026, 7, 5, 17, 0, 0),
        parent_booking_id=3,
        is_return=True,
    )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=[a1, a2, b1, b2],
        subject_key_fn=lambda b: resolve_subject_identity(b).key,
        amount_ht_fn=lambda b: Decimal(str(b.amount)),
    )
    assert len(units) == 2
    keys = {u.subject_key for u in units}
    assert keys == {"institution_patient:10", "institution_patient:20"}
    segments = sum(len(u.booking_ids) for u in units)
    assert segments == 4


def test_chain_abc_not_merged_as_single_round_trip():
    """A→B→C : pas une unité A/R unique (max 2 segments)."""
    a = _bk(
        1,
        institution_patient_id=1,
        pickup="A",
        dropoff="B",
        scheduled=datetime(2026, 7, 1, 9, 0, 0),
        amount="30.00",
    )
    b = _bk(
        2,
        institution_patient_id=1,
        pickup="B",
        dropoff="C",
        scheduled=datetime(2026, 7, 1, 10, 0, 0),
        amount="30.00",
    )
    c = _bk(
        3,
        institution_patient_id=1,
        pickup="C",
        dropoff="D",
        scheduled=datetime(2026, 7, 1, 11, 0, 0),
        amount="30.00",
    )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=[a, b, c],
        subject_key_fn=lambda b: resolve_subject_identity(b).key,
        amount_ht_fn=lambda b: Decimal(str(b.amount)),
    )
    assert len(units) == 3
    assert all(u.kind == "single" for u in units)


def test_same_patient_same_day_without_relation_never_merges():
    """Même patient + même date + adresses inversées, sans lien métier → 2 lignes."""
    aller = _bk(
        10,
        client_id=99,
        institution_patient_id=None,
        pickup="Foyer, Route 1",
        dropoff="Clinique, Chemin 2",
        scheduled=datetime(2026, 7, 10, 9, 0, 0),
    )
    retour = _bk(
        11,
        client_id=99,
        institution_patient_id=None,
        pickup="Clinique, Chemin 2",
        dropoff="Foyer, Route 1",
        scheduled=datetime(2026, 7, 10, 15, 0, 0),
    )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=[aller, retour],
        subject_key_fn=lambda b: resolve_subject_identity(b).key,
        amount_ht_fn=lambda b: Decimal(str(b.amount)),
    )
    assert len(units) == 2
    assert {u.kind for u in units} == {"single"}
    assert {i for u in units for i in u.booking_ids} == {10, 11}


def test_segments_vs_units_four_segments_two_round_trips():
    """C5 : 4 segments / 2 A/R → units_count=2, segments=4."""
    bookings = []
    for i, pid in enumerate((100, 200)):
        bookings.append(
            _bk(
                pid,
                institution_patient_id=7,
                amount="40.00",
                scheduled=datetime(2026, 7, 5 + i, 10, 0, 0),
            )
        )
        bookings.append(
            _bk(
                pid + 1,
                institution_patient_id=7,
                amount="40.00",
                pickup="Rue B 2, 1227 Carouge",
                dropoff="Rue A 1, 1200 Genève",
                scheduled=datetime(2026, 7, 5 + i, 16, 0, 0),
                parent_booking_id=pid,
                is_return=True,
            )
        )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=bookings,
        subject_key_fn=lambda b: resolve_subject_identity(b).key,
        amount_ht_fn=lambda b: Decimal(str(b.amount)),
    )
    assert len(units) == 2
    assert sum(len(u.booking_ids) for u in units) == 4
