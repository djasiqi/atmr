"""Isolation, clôture et invariant résumé = détail du temps de travail."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest

from models import Company, Driver, User
from models.booking import Booking
from models.driver_work_time import (
    DriverCompensationLedger,
    DriverCompensationPolicy,
    DriverManualWorkEntry,
    DriverWorkTimeAdjustment,
    DriverWorkTimePeriodClosure,
)
from models.enums import BookingStatus, UserRole

_PERIOD = {"from": "2026-09-01", "to": "2026-09-30"}


@pytest.fixture(autouse=True)
def _mois_de_septembre_termine(monkeypatch):
    """Les tests de clôture ne dépendent pas du jour réel de la machine."""
    monkeypatch.setattr(
        "routes.company_work_time.zurich_today",
        lambda: date(2026, 10, 1),
    )


def _user(db, role: UserRole) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"wt_{suffix}"
    user.email = f"wt_{suffix}@example.com"
    user.public_id = str(uuid.uuid4())
    user.role = role
    user.first_name = "Noah"
    user.last_name = "Bernard"
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def _other_company(db) -> tuple[Company, Driver]:
    owner = _user(db, UserRole.company)
    suffix = uuid.uuid4().hex[:8]
    company = Company()
    company.name = f"Autre {suffix}"
    company.address = "Rue Secrete 9"
    company.contact_phone = "0210000000"
    company.contact_email = f"autre_{suffix}@example.com"
    company.user_id = owner.id
    db.session.add(company)
    db.session.flush()
    driver = Driver()
    driver.user = _user(db, UserRole.driver)
    driver.company_id = company.id
    driver.is_active = True
    db.session.add(driver)
    db.session.flush()
    return company, driver


def _own_driver(db, company: Company) -> Driver:
    driver = Driver()
    driver.user = _user(db, UserRole.driver)
    driver.company_id = company.id
    driver.is_active = True
    db.session.add(driver)
    db.session.flush()
    return driver


def _booking(db, *, company, driver, client, **overrides) -> Booking:
    payload = {
        "company_id": company.id,
        "driver_id": driver.id,
        "client_id": client.id,
        "user_id": client.user_id,
        "customer_name": "Client visible",
        "pickup_location": "Rue A",
        "dropoff_location": "Rue B",
        "scheduled_time": datetime(2026, 9, 10, 10, 0),
        "status": BookingStatus.COMPLETED,
        "amount": 40.0,
        "is_round_trip": False,
        "is_return": False,
        "is_urgent": False,
        "time_confirmed": True,
        "arrived_at": datetime(2026, 9, 10, 8, 0, tzinfo=UTC),
        "completed_at": datetime(2026, 9, 10, 8, 40, tzinfo=UTC),
    }
    payload.update(overrides)
    booking = Booking(**payload)
    db.session.add(booking)
    db.session.flush()
    return booking


def _policy(client, headers, **overrides):
    body = {
        "effective_from": "2026-01-01",
        "transport_flat_minutes": 30,
        "work_type_rules": {"administrative": {"mode": "real_time"}},
    }
    if "one_way_minutes" in overrides and "transport_flat_minutes" not in overrides:
        overrides = {
            **overrides,
            "transport_flat_minutes": overrides["one_way_minutes"],
        }
    body.update(overrides)
    return client.post(
        "/api/v1/companies/me/work-time/compensation-policies",
        json=body,
        headers=headers,
    )


def _finalize(client, headers, period=None):
    return client.post(
        "/api/v1/companies/me/work-time/periods/finalize",
        json=period or _PERIOD,
        headers=headers,
    )


def _summary(client, headers, period=None):
    window = period or _PERIOD
    return client.get(
        "/api/v1/companies/me/work-time/summary"
        f"?from={window['from']}&to={window['to']}",
        headers=headers,
    )


def test_ressources_dune_autre_entreprise_sont_inaccessibles(
    client, auth_headers, db, sample_company, sample_client
):
    other_company, other_driver = _other_company(db)
    secret = _booking(
        db,
        company=other_company,
        driver=other_driver,
        client=sample_client,
        pickup_location="Rue Secrete 9",
        dropoff_location="Quai Cache",
        customer_name="Client secret",
    )
    own_driver = _own_driver(db, sample_company)
    manual = DriverManualWorkEntry(
        company_id=other_company.id,
        driver_id=other_driver.id,
        work_date=datetime(2026, 9, 12).date(),
        started_at=datetime(2026, 9, 12, 8, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 12, 8, 30, tzinfo=UTC),
        duration_minutes=30,
        work_type="administrative",
        description="note secrete",
    )
    adjustment = DriverWorkTimeAdjustment(
        company_id=other_company.id,
        driver_id=other_driver.id,
        booking_id=secret.id,
        corrected_arrived_at=datetime(2026, 9, 10, 8, 0, tzinfo=UTC),
        corrected_completed_at=datetime(2026, 9, 10, 8, 40, tzinfo=UTC),
        reason="motif secret",
    )
    policy = DriverCompensationPolicy(
        company_id=other_company.id,
        effective_from=datetime(2026, 1, 1).date(),
        transport_flat_minutes=99,
        one_way_minutes=99,
        round_trip_minutes=99,
        mode="flat_per_trip",
        notes="regle secrete",
    )
    closure = DriverWorkTimePeriodClosure(
        company_id=other_company.id,
        period_from=datetime(2026, 9, 1).date(),
        period_to=datetime(2026, 9, 30).date(),
        finalized_at=datetime(2026, 10, 2, tzinfo=UTC),
    )
    db.session.add_all([manual, adjustment, policy, closure])
    db.session.flush()

    explain = client.get(
        f"/api/v1/companies/me/work-time/bookings/{secret.id}/explain",
        headers=auth_headers,
    )
    assert explain.status_code == 404
    assert "Rue Secrete" not in explain.get_data(as_text=True)
    assert "Client secret" not in explain.get_data(as_text=True)

    created = client.post(
        "/api/v1/companies/me/work-time/adjustments",
        json={
            "booking_id": secret.id,
            "reason": "tentative",
            "corrected_arrived_at": "2026-09-10T08:00:00Z",
            "corrected_completed_at": "2026-09-10T09:00:00Z",
        },
        headers=auth_headers,
    )
    assert created.status_code == 404
    assert (
        db.session.query(DriverWorkTimeAdjustment)
        .filter(DriverWorkTimeAdjustment.company_id == sample_company.id)
        .count()
        == 0
    )

    listed = client.get(
        f"/api/v1/companies/me/work-time/adjustments?booking_id={secret.id}",
        headers=auth_headers,
    )
    assert listed.status_code == 200
    assert listed.get_json()["adjustments"] == []
    assert "motif secret" not in listed.get_data(as_text=True)

    by_driver = client.get(
        f"/api/v1/companies/me/work-time/adjustments?driver_id={other_driver.id}",
        headers=auth_headers,
    )
    assert by_driver.status_code == 200
    assert by_driver.get_json()["adjustments"] == []

    manual_post = client.post(
        "/api/v1/companies/me/work-time/manual-entries",
        json={
            "driver_id": other_driver.id,
            "work_type": "administrative",
            "started_at": "2026-09-12T08:00:00Z",
            "ended_at": "2026-09-12T08:40:00Z",
        },
        headers=auth_headers,
    )
    assert manual_post.status_code == 404

    linked = client.post(
        "/api/v1/companies/me/work-time/manual-entries",
        json={
            "driver_id": own_driver.id,
            "work_type": "administrative",
            "started_at": "2026-09-12T08:00:00Z",
            "ended_at": "2026-09-12T08:40:00Z",
            "booking_id": secret.id,
        },
        headers=auth_headers,
    )
    assert linked.status_code == 404
    assert "note secrete" not in linked.get_data(as_text=True)

    cancel = client.post(
        f"/api/v1/companies/me/work-time/manual-entries/{manual.id}/cancel",
        json={"reason": "tentative"},
        headers=auth_headers,
    )
    assert cancel.status_code == 404
    db.session.refresh(manual)
    assert manual.cancelled_at is None

    policies = client.get(
        "/api/v1/companies/me/work-time/compensation-policies",
        headers=auth_headers,
    )
    assert policies.status_code == 200
    assert "regle secrete" not in policies.get_data(as_text=True)
    assert all(row["id"] != policy.id for row in policies.get_json()["policies"])

    foreign_driver_policy = _policy(client, auth_headers, driver_id=other_driver.id)
    assert foreign_driver_policy.status_code == 404

    reopen = client.post(
        "/api/v1/companies/me/work-time/periods/reopen",
        json={**_PERIOD, "reason": "tentative croisee"},
        headers=auth_headers,
    )
    assert reopen.status_code == 404
    db.session.refresh(closure)
    assert closure.reopened_at is None


def test_invariant_resume_egal_detail_avec_pagination(
    client, auth_headers, db, sample_company, sample_client
):
    driver = _own_driver(db, sample_company)
    assert _policy(client, auth_headers).status_code == 201
    _booking(db, company=sample_company, driver=driver, client=sample_client)
    _booking(
        db,
        company=sample_company,
        driver=driver,
        client=sample_client,
        scheduled_time=datetime(2026, 9, 11, 9, 0),
        arrived_at=None,
        completed_at=None,
        pickup_location="Sans fin",
        dropoff_location="Inconnue",
    )
    group = uuid.uuid4().hex
    for seq, pickup, dropoff in ((1, "A", "B"), (2, "B", "C"), (3, "C", "A")):
        _booking(
            db,
            company=sample_company,
            driver=driver,
            client=sample_client,
            route_group_id=group,
            route_sequence_number=seq,
            pickup_location=pickup,
            dropoff_location=dropoff,
            scheduled_time=datetime(2026, 9, 20, 10, seq),
            arrived_at=datetime(2026, 9, 20, 8, seq, tzinfo=UTC),
            completed_at=datetime(2026, 9, 20, 8, seq + 20, tzinfo=UTC),
        )
    _booking(
        db,
        company=sample_company,
        driver=driver,
        client=sample_client,
        is_round_trip=True,
        pickup_location="Depot",
        dropoff_location="Hopital",
        scheduled_time=datetime(2026, 9, 28, 10, 0),
        arrived_at=datetime(2026, 9, 28, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 28, 8, 15, tzinfo=UTC),
    )
    _booking(
        db,
        company=sample_company,
        driver=driver,
        client=sample_client,
        is_return=True,
        pickup_location="Hopital",
        dropoff_location="Depot",
        scheduled_time=datetime(2026, 10, 2, 10, 0),
        arrived_at=None,
        completed_at=None,
        status=BookingStatus.ASSIGNED,
    )
    outside = _booking(
        db,
        company=sample_company,
        driver=driver,
        client=sample_client,
        scheduled_time=datetime(2026, 8, 15, 10, 0),
        arrived_at=datetime(2026, 8, 15, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 15, 9, 39, tzinfo=UTC),
        pickup_location="Aout",
    )
    created = client.post(
        "/api/v1/companies/me/work-time/manual-entries",
        json={
            "driver_id": driver.id,
            "work_type": "administrative",
            "started_at": "2026-09-12T08:00:00Z",
            "ended_at": "2026-09-12T08:40:00Z",
        },
        headers=auth_headers,
    )
    assert created.status_code == 201

    summary = _summary(client, auth_headers)
    assert summary.status_code == 200
    body = summary.get_json()
    worked = 0
    compensated = 0
    booking_ids = set()
    page = 1
    while True:
        detail = client.get(
            f"/api/v1/companies/me/work-time/drivers/{driver.id}"
            "?from=2026-09-01&to=2026-09-30&per_page=1&page=" + str(page),
            headers=auth_headers,
        )
        assert detail.status_code == 200
        payload = detail.get_json()
        assert len(payload["days"]) <= 1
        for day in payload["days"]:
            for entry in day["entries"]:
                worked += int(entry["worked_minutes"] or 0)
                compensated += int(entry["compensated_minutes"] or 0)
                if entry.get("booking_id"):
                    booking_ids.add(entry["booking_id"])
        if page * payload["per_page"] >= payload["total_days"]:
            break
        page += 1

    assert worked == body["kpis"]["total_worked_minutes"]
    assert compensated == body["kpis"]["compensated_minutes"]
    assert worked == 40 + 60 + 15 + 40
    assert compensated == 30 + 30 + 90 + 30 + 40
    assert outside.id not in booking_ids
    assert body["kpis"]["incomplete_segments_count"] >= 1
    assert body["kpis"]["completed_segments_count"] == 5
    assert body["kpis"]["transport_count"] == 6
    assert body["kpis"]["manual_minutes"] == 40
    assert body["kpis"]["pending_journeys_count"] == 0


def test_cloture_fige_la_remuneration_et_refuse_les_mutations(
    client, auth_headers, db, sample_company, sample_client
):
    driver = _own_driver(db, sample_company)
    assert _policy(client, auth_headers).status_code == 201
    booking = _booking(db, company=sample_company, driver=driver, client=sample_client)

    first = _finalize(client, auth_headers)
    assert first.status_code == 201
    second = _finalize(client, auth_headers)
    assert second.status_code == 409
    overlap = _finalize(
        client, auth_headers, {"from": "2026-09-15", "to": "2026-10-15"}
    )
    assert overlap.status_code == 409
    assert overlap.get_json()["code"] == "PERIOD_NOT_CLOSABLE"

    summary = _summary(client, auth_headers)
    assert summary.status_code == 200
    frozen = summary.get_json()
    assert frozen["compensation_source"] == "ledger"
    assert frozen["period_finalized"] is True
    assert frozen["kpis"]["compensated_minutes"] == 30

    ledger = (
        db.session.query(DriverCompensationLedger)
        .filter(DriverCompensationLedger.company_id == sample_company.id)
        .one()
    )
    assert ledger.line_key.startswith("bk:")
    assert ledger.journey_key.startswith("bk:")
    assert ledger.driver_id == driver.id
    assert ledger.policy_id is not None
    assert ledger.base_minutes == 30
    assert ledger.intermediate_stop_count == 0
    assert ledger.classification_source
    assert ledger.accounting_date.isoformat() == "2026-09-10"
    assert ledger.compensated_minutes == 30
    assert ledger.rule_type == "transport_flat"
    assert ledger.journey_status == "complete"

    booking.completed_at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    db.session.flush()
    after_edit = _summary(client, auth_headers).get_json()
    assert after_edit["kpis"]["compensated_minutes"] == 30
    db.session.refresh(ledger)
    assert ledger.compensated_minutes == 30

    adjustment = client.post(
        "/api/v1/companies/me/work-time/adjustments",
        json={
            "booking_id": booking.id,
            "reason": "apres cloture",
            "corrected_arrived_at": "2026-09-10T08:00:00Z",
            "corrected_completed_at": "2026-09-10T10:00:00Z",
        },
        headers=auth_headers,
    )
    assert adjustment.status_code == 409

    manual = client.post(
        "/api/v1/companies/me/work-time/manual-entries",
        json={
            "driver_id": driver.id,
            "work_type": "administrative",
            "started_at": "2026-09-12T08:00:00Z",
            "ended_at": "2026-09-12T09:00:00Z",
        },
        headers=auth_headers,
    )
    assert manual.status_code == 409

    retro = _policy(
        client, auth_headers, effective_from="2026-09-01", one_way_minutes=45
    )
    assert retro.status_code == 409
    still = _summary(client, auth_headers).get_json()
    assert still["kpis"]["compensated_minutes"] == 30

    already_open = client.post(
        "/api/v1/companies/me/work-time/periods/reopen",
        json={"from": "2026-08-01", "to": "2026-08-31", "reason": "rien a rouvrir"},
        headers=auth_headers,
    )
    assert already_open.status_code == 404

    reopened = client.post(
        "/api/v1/companies/me/work-time/periods/reopen",
        json={**_PERIOD, "reason": "correction de paie"},
        headers=auth_headers,
    )
    assert reopened.status_code == 200
    again = client.post(
        "/api/v1/companies/me/work-time/periods/reopen",
        json={**_PERIOD, "reason": "deja ouverte"},
        headers=auth_headers,
    )
    assert again.status_code == 404

    assert (
        _policy(
            client, auth_headers, effective_from="2026-09-01", one_way_minutes=45
        ).status_code
        == 201
    )
    refrozen = _finalize(client, auth_headers)
    assert refrozen.status_code == 201
    latest = _summary(client, auth_headers).get_json()
    assert latest["compensation_source"] == "ledger"
    assert latest["kpis"]["compensated_minutes"] == 45
    assert (
        db.session.query(DriverCompensationLedger)
        .filter(DriverCompensationLedger.company_id == sample_company.id)
        .count()
        == 2
    )
    neighbour = _finalize(
        client, auth_headers, {"from": "2026-08-01", "to": "2026-08-31"}
    )
    assert neighbour.status_code == 201


def _shown(summary):
    kpis = summary["kpis"]
    return {
        key: kpis[key]
        for key in (
            "transport_count",
            "flat_transport_minutes",
            "real_transport_minutes",
            "flat_added_minutes",
            "real_added_minutes",
            "review_count_flat",
            "review_count_real",
        )
    }


def test_relecture_et_recloture_sans_changement(
    client, auth_headers, db, sample_company, sample_client
):
    driver = _own_driver(db, sample_company)
    assert _policy(client, auth_headers).status_code == 201
    _booking(db, company=sample_company, driver=driver, client=sample_client)
    assert _finalize(client, auth_headers).status_code == 201
    first = _shown(_summary(client, auth_headers).get_json())
    second = _shown(_summary(client, auth_headers).get_json())
    assert second == first
    assert first["transport_count"] == 1
    assert first["flat_transport_minutes"] == 30
    reopened = client.post(
        "/api/v1/companies/me/work-time/periods/reopen",
        json={**_PERIOD, "reason": "vérification sans modification"},
        headers=auth_headers,
    )
    assert reopened.status_code == 200
    assert _finalize(client, auth_headers).status_code == 201
    assert _shown(_summary(client, auth_headers).get_json()) == first


def test_cloture_409_ne_persiste_ni_closure_ni_ledger(
    client, auth_headers, db, sample_company, sample_client, monkeypatch
):
    driver = _own_driver(db, sample_company)
    assert _policy(client, auth_headers).status_code == 201
    _booking(db, company=sample_company, driver=driver, client=sample_client)
    company_id = int(sample_company.id)

    def diverge(report, rows):
        from domain.work_time.report import apply_ledger_snapshot

        apply_ledger_snapshot(report, rows)
        report["kpis"]["flat_transport_minutes"] = 0

    monkeypatch.setattr("routes.company_work_time.apply_ledger_snapshot", diverge)
    refused = _finalize(client, auth_headers)
    assert refused.status_code == 409
    assert refused.get_json()["divergences"]
    assert (
        db.session.query(DriverWorkTimePeriodClosure)
        .filter(DriverWorkTimePeriodClosure.company_id == company_id)
        .count()
        == 0
    )
    assert (
        db.session.query(DriverCompensationLedger)
        .filter(DriverCompensationLedger.company_id == company_id)
        .count()
        == 0
    )


def test_seuls_les_mois_civils_termines_peuvent_etre_clotures(
    client, auth_headers, db, sample_company, monkeypatch
):
    company_id = int(sample_company.id)
    refused = [
        (date(2026, 10, 1), {"from": "2026-09-30", "to": "2026-09-30"}),
        (date(2026, 10, 1), {"from": "2026-09-28", "to": "2026-10-04"}),
        (date(2026, 10, 1), {"from": "2026-09-15", "to": "2026-09-30"}),
        (date(2026, 10, 1), {"from": "2026-09-01", "to": "2026-10-15"}),
        (date(2026, 10, 30), {"from": "2026-10-01", "to": "2026-10-31"}),
        (date(2026, 10, 31), {"from": "2026-10-01", "to": "2026-10-31"}),
    ]
    for today, period in refused:
        monkeypatch.setattr(
            "routes.company_work_time.zurich_today", lambda today=today: today
        )
        response = _finalize(client, auth_headers, period)
        assert response.status_code == 409
        assert response.get_json()["code"] == "PERIOD_NOT_CLOSABLE"
    assert (
        db.session.query(DriverWorkTimePeriodClosure)
        .filter(DriverWorkTimePeriodClosure.company_id == company_id)
        .count()
        == 0
    )

    allowed = [
        (date(2026, 11, 1), {"from": "2026-10-01", "to": "2026-10-31"}),
        (date(2026, 10, 1), {"from": "2026-09-01", "to": "2026-09-30"}),
        (date(2026, 12, 1), {"from": "2026-11-01", "to": "2026-11-30"}),
    ]
    for today, period in allowed:
        monkeypatch.setattr(
            "routes.company_work_time.zurich_today", lambda today=today: today
        )
        response = _finalize(client, auth_headers, period)
        assert response.status_code == 201, response.get_json()
