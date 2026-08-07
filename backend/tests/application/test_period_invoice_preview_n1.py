"""Régression N+1 : preview période clinique ne charge pas 1 client par booking.

Fixes PYTHON-FLASK-DQ (invoices_billing_opportunities → build_period_invoice_preview S2).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import event
from sqlalchemy.engine import Engine

from application.invoices.billing_opportunities import list_billing_opportunities
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import Booking, Client, Company, User
from models.enums import BookingStatus, UserRole
from repositories.client_repository import ClientRepository


def _make_company(db, *, name: str | None = None) -> Company:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"co_{suffix}"
    user.email = f"co-{suffix}@test.ch"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.name = name or f"Transport {suffix}"
    company.address = "Rue Test 1, 1200 Genève"
    company.contact_phone = "0220000000"
    company.contact_email = f"contact-{suffix}@test.ch"
    company.user_id = user.id
    db.session.add(company)
    db.session.flush()
    return company


def _make_client(db, company: Company, *, idx: int) -> Client:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"cli_{suffix}"
    user.email = f"cli-{suffix}@test.ch"
    user.role = UserRole.client
    user.first_name = f"Patient{idx}"
    user.last_name = "Test"
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = company.id
    client.domicile_address = f"Rue {idx}"
    client.domicile_zip = "1200"
    client.domicile_city = "Genève"
    client.default_billed_to_type = "clinic"
    db.session.add(client)
    db.session.flush()
    return client


def _make_clinic_booking(
    db,
    *,
    transport: Company,
    clinic: Company,
    client: Client,
    scheduled: datetime,
) -> Booking:
    booking = Booking()
    booking.company_id = transport.id
    booking.client_id = client.id
    booking.customer_name = f"{client.user.first_name} {client.user.last_name}"
    booking.scheduled_time = scheduled
    booking.status = BookingStatus.COMPLETED.value
    booking.pickup_location = "Pickup"
    booking.dropoff_location = "Drop"
    booking.amount = 40.0
    booking.billed_to_type = "clinic"
    booking.billed_to_company_id = clinic.id
    booking.is_return = False
    db.session.add(booking)
    db.session.flush()
    return booking


def _count_client_by_id_lookups(fn):
    """Compte les SELECT client filtrés par ``client.id`` (pattern N+1 unitaire)."""
    lookups: list[str] = []

    def _on_execute(conn, cursor, statement, parameters, context, executemany):
        stmt = " ".join(statement.strip().lower().split())
        if not stmt.startswith("select"):
            return
        # Pattern unitaire : FROM client ... WHERE client.id = ...
        compact = stmt.replace('"', "")
        if " from client " not in f" {compact} " and not compact.startswith(
            "select client."
        ):
            return
        if "where client.id =" in compact or "where client.id=" in compact:
            lookups.append(statement[:160])

    event.listen(Engine, "before_cursor_execute", _on_execute)
    try:
        result = fn()
    finally:
        event.remove(Engine, "before_cursor_execute", _on_execute)
    return result, lookups


def test_find_models_by_ids_batch(db):
    transport = _make_company(db)
    clients = [_make_client(db, transport, idx=i) for i in range(5)]
    db.session.commit()

    found = ClientRepository().find_models_by_ids_and_company_with_user(
        {c.id for c in clients}, transport.id
    )
    assert set(found.keys()) == {c.id for c in clients}
    assert all(found[c.id].user is not None for c in clients)


def test_clinic_preview_batches_client_loads(db):
    transport = _make_company(db)
    clinic = _make_company(db, name="Clinique N1")
    n_bookings = 12
    for i in range(n_bookings):
        client = _make_client(db, transport, idx=i)
        _make_clinic_booking(
            db,
            transport=transport,
            clinic=clinic,
            client=client,
            scheduled=datetime(2026, 7, 1 + (i % 28), 10, 0),
        )
    db.session.commit()

    def _run():
        return build_period_invoice_preview(
            company_id=transport.id,
            period_year=2026,
            period_month=7,
            clinic_company_id=clinic.id,
            include_line_details=True,
        )

    prev, lookups = _count_client_by_id_lookups(_run)
    assert prev.transports_count == n_bookings
    assert len(prev.preview_lines) == n_bookings
    # Batch IN (...) : zéro lookup unitaire client.id =
    assert lookups == [], (
        f"N+1 clients détecté: {len(lookups)} SELECT client.id= "
        f"pour {n_bookings} bookings. Exemples: {lookups[:3]}"
    )


def test_clinic_preview_aggregates_skip_client_loads(db):
    transport = _make_company(db)
    clinic = _make_company(db, name="Clinique Agg")
    for i in range(8):
        client = _make_client(db, transport, idx=i)
        _make_clinic_booking(
            db,
            transport=transport,
            clinic=clinic,
            client=client,
            scheduled=datetime(2026, 7, 2 + i, 9, 0),
        )
    db.session.commit()

    def _run():
        return build_period_invoice_preview(
            company_id=transport.id,
            period_year=2026,
            period_month=7,
            clinic_company_id=clinic.id,
            include_line_details=False,
        )

    prev, lookups = _count_client_by_id_lookups(_run)
    assert prev.transports_count == 8
    assert prev.preview_lines == ()
    assert lookups == [], (
        f"include_line_details=False ne doit pas charger de clients: {lookups}"
    )


def test_billing_opportunities_clinic_no_client_n1(db):
    transport = _make_company(db)
    clinic = _make_company(db, name="Clinique Opp")
    for i in range(10):
        client = _make_client(db, transport, idx=i)
        _make_clinic_booking(
            db,
            transport=transport,
            clinic=clinic,
            client=client,
            scheduled=datetime(2026, 7, 3 + i, 11, 0),
        )
    db.session.commit()

    def _run():
        return list_billing_opportunities(
            company_id=transport.id, period_year=2026, period_month=7
        )

    result, lookups = _count_client_by_id_lookups(_run)
    assert len(result.clinic_items) == 1
    assert result.clinic_items[0].transports_count == 10
    assert lookups == [], (
        f"N+1 sur opportunités: {len(lookups)} SELECT client.id=. "
        f"Exemples: {lookups[:3]}"
    )
