"""Couverture ciblée des branches historiques de ``routes.companies``."""

# ruff: noqa: F811

from __future__ import annotations

import inspect
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from flask_jwt_extended import verify_jwt_in_request

from models import Booking, Company, User
from models.enums import (
    BookingStatus,
    PartnershipStatus,
    TransferModel,
    UserRole,
)
from models.partnership import Partnership
from tests.routes.test_companies import _auth_headers, companies_world


def _second_company(db) -> Company:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"partner_{suffix}"
    user.email = f"partner_{suffix}@test.ch"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.name = f"Partenaire {suffix}"
    company.address = "Rue Partenaire 2"
    company.contact_email = user.email
    company.user_id = user.id
    company.is_approved = True
    company.dispatch_enabled = False
    db.session.add(company)
    db.session.flush()
    return company


def _partnership(
    db,
    owner_id: int,
    partner_id: int,
    *,
    status: PartnershipStatus,
) -> Partnership:
    partnership = Partnership()
    partnership.owner_company_id = owner_id
    partnership.partner_company_id = partner_id
    partnership.default_transfer_model = TransferModel.SUBCONTRACT
    partnership.default_partner_tariff_percent = Decimal("80.00")
    partnership.default_margin_percent = Decimal("20.00")
    partnership.auto_accept_rules = False
    partnership.auto_invoice = True
    partnership.payment_terms_days = 30
    partnership.status = status
    partnership.is_active = True
    db.session.add(partnership)
    db.session.flush()
    return partnership


def test_partnership_get_deduplicates_puts_and_deletes(
    client, app, companies_world, db, monkeypatch
):
    world = companies_world
    company = world["company"]
    other = _second_company(db)
    older = _partnership(
        db,
        company.id,
        other.id,
        status=PartnershipStatus.PENDING,
    )
    newer = _partnership(
        db,
        other.id,
        company.id,
        status=PartnershipStatus.ACCEPTED,
    )
    db.session.commit()

    monkeypatch.setattr(
        "services.partnerships.stats.PartnershipStatsService.get_partnership_stats",
        lambda *_args, **_kwargs: {"total_bookings": 0},
    )
    headers = _auth_headers(
        app,
        world["company_user"],
        role="company",
        company_id=company.id,
    )

    listed = client.get("/api/v1/companies/me/partnerships", headers=headers)
    assert listed.status_code == 200, listed.get_json()
    body = listed.get_json()
    data = body.get("data", body)
    assert len(data) == 1
    assert data[0]["id"] == newer.id
    assert data[0]["is_owner"] is False

    updated = client.put(
        f"/api/v1/companies/me/partnerships/{older.id}",
        headers=headers,
        json={
            "auto_invoice": False,
            "auto_accept": True,
            "default_partner_tariff_percent": 75,
            "default_margin_percent": 25,
            "default_transfer_model": "SUBCONTRACT",
        },
    )
    assert updated.status_code == 200, updated.get_json()
    db.session.refresh(older)
    assert older.auto_invoice is False
    assert float(older.default_partner_tariff_percent) == 75
    assert older.status == PartnershipStatus.PENDING

    deleted = client.delete(
        f"/api/v1/companies/me/partnerships/{older.id}",
        headers=headers,
    )
    assert deleted.status_code == 200, deleted.get_json()
    assert db.session.get(Partnership, older.id) is None


@pytest.mark.parametrize(
    ("url", "payload"),
    [
        (
            "/api/v1/companies/me/partnerships/statements/generate",
            {"year": "invalide"},
        ),
        (
            "/api/v1/companies/me/partnerships/statements/generate",
            {"month": "invalide"},
        ),
        (
            "/api/v1/companies/me/partnerships/statements/generate",
            {"start_date": "pas-une-date"},
        ),
        (
            "/api/v1/companies/me/partnerships/statements/generate",
            {"end_date": "pas-une-date"},
        ),
        (
            "/api/v1/companies/me/partnerships/999999/statements/generate",
            {"start_date": "pas-une-date"},
        ),
    ],
)
def test_statement_generation_validation_errors(
    client, app, companies_world, url, payload
):
    world = companies_world
    headers = _auth_headers(
        app,
        world["company_user"],
        role="company",
        company_id=world["company"].id,
    )
    response = client.post(url, headers=headers, json=payload)
    assert response.status_code == 400, response.get_json()


def test_statement_generation_successes(client, app, companies_world, db, monkeypatch):
    world = companies_world
    other = _second_company(db)
    partnership = _partnership(
        db,
        world["company"].id,
        other.id,
        status=PartnershipStatus.ACCEPTED,
    )
    db.session.commit()
    monkeypatch.setattr(
        "services.partnerships.statements.PartnershipStatementService."
        "generate_consolidated_statement",
        lambda _self, **_kwargs: "/uploads/statements/consolide.pdf",
    )
    monkeypatch.setattr(
        "services.partnerships.statements.PartnershipStatementService."
        "generate_partnership_statement",
        lambda _self, **_kwargs: "/uploads/statements/partenaire.pdf",
    )
    headers = _auth_headers(
        app,
        world["company_user"],
        role="company",
        company_id=world["company"].id,
    )
    payload = {
        "period_type": "custom",
        "year": "2026",
        "month": "8",
        "start_date": "2026-08-01T00:00:00Z",
        "end_date": "2026-08-31T23:59:59Z",
    }

    consolidated = client.post(
        "/api/v1/companies/me/partnerships/statements/generate",
        headers=headers,
        json=payload,
    )
    assert consolidated.status_code == 200, consolidated.get_json()

    individual = client.post(
        f"/api/v1/companies/me/partnerships/{partnership.id}/statements/generate",
        headers=headers,
        json=payload,
    )
    assert individual.status_code == 200, individual.get_json()


def test_delete_outbound_cascades_to_return_booking(client, app, companies_world, db):
    world = companies_world
    outbound = world["booking"]
    return_booking = Booking()
    return_booking.user_id = outbound.user_id
    return_booking.company_id = outbound.company_id
    return_booking.client_id = outbound.client_id
    return_booking.customer_name = outbound.customer_name
    return_booking.pickup_location = outbound.dropoff_location
    return_booking.dropoff_location = outbound.pickup_location
    return_booking.scheduled_time = datetime.now(UTC) + timedelta(hours=5)
    return_booking.status = BookingStatus.PENDING
    return_booking.is_return = True
    return_booking.parent_booking_id = outbound.id
    return_booking.amount = Decimal("0.00")
    return_booking.billed_to_type = "patient"
    db.session.add(return_booking)
    db.session.commit()
    return_id = return_booking.id

    headers = _auth_headers(
        app,
        world["company_user"],
        role="company",
        company_id=world["company"].id,
    )
    response = client.delete(
        f"/api/v1/companies/me/reservations/{outbound.id}",
        headers=headers,
        json={"reason_code": "OTHER", "reason_text": "cascade couverture"},
    )
    assert response.status_code == 200, response.get_json()
    assert db.session.get(Booking, outbound.id) is None
    assert db.session.get(Booking, return_id) is None


def test_manual_booking_iso8601_success(client, app, companies_world):
    world = companies_world
    headers = _auth_headers(
        app,
        world["company_user"],
        role="company",
        company_id=world["company"].id,
    )
    response = client.post(
        "/api/v1/companies/me/reservations/manual",
        headers=headers,
        json={
            "customer_name": "Réservation ISO",
            "pickup_location": "Rue ISO 1, Genève",
            "dropoff_location": "Rue ISO 2, Genève",
            "scheduled_time": "2026-08-20T10:00:00Z",
            "client_id": world["client"].id,
            "amount": 40,
        },
    )
    assert response.status_code == 201, response.get_json()
    body = response.get_json()
    assert body["reservation"]["id"]
    assert body["reservations"]


def test_update_reservation_legacy_handler_directly(app, companies_world, monkeypatch):
    import routes.companies as companies_route

    booking = companies_world["booking"]
    company = companies_world["company"]
    monkeypatch.setattr(
        companies_route,
        "_get_current_company_via_use_case",
        lambda: (company, None, None),
    )
    monkeypatch.setattr(
        "repositories.booking_repository.BookingRepository."
        "find_model_by_id_with_visibility",
        lambda _self, booking_id, company_id: (
            booking if booking_id == booking.id and company_id == company.id else None
        ),
    )
    monkeypatch.setattr(
        companies_route,
        "_maybe_trigger_dispatch",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "services.reservations_summary_cache."
        "invalidate_summary_cache_for_booking_after_day_change",
        lambda *_args, **_kwargs: None,
    )

    handler = inspect.unwrap(companies_route.UpdateReservation.put)
    with app.test_request_context(
        method="PUT",
        json={
            "pickup_location": "Nouvelle adresse de départ",
            "dropoff_location": "Nouvelle destination",
            "notes_medical": "Mise à jour directe",
        },
    ):
        body, status = handler(companies_route.UpdateReservation(), booking.id)

    assert status == 200, body
    assert body["reservation"]["id"] == booking.id


def test_debug_transfer_and_company_from_token(client, app, companies_world):
    import routes.companies as companies_route

    world = companies_world
    headers = _auth_headers(
        app,
        world["company_user"],
        role="company",
        company_id=world["company"].id,
    )
    debug = client.get(
        f"/api/v1/companies/debug/booking/{world['booking'].id}/transfer",
        headers=headers,
    )
    assert debug.status_code == 200, debug.get_json()
    assert debug.get_json()["booking"]["id"] == world["booking"].id
    assert debug.get_json()["transfers"] == []

    with app.test_request_context(headers=headers):
        verify_jwt_in_request()
        company, error, status = companies_route.get_company_from_token()
        assert company is not None
        assert company.id == world["company"].id
        assert error is None
        assert status is None


def test_respond_to_change_request_success_directly(app, companies_world, monkeypatch):
    import routes.companies as companies_route
    from application.institutions.respond_to_change_request import (
        RespondToChangeRequestResult,
    )

    company = companies_world["company"]
    monkeypatch.setattr(
        companies_route,
        "_get_current_company_via_use_case",
        lambda: (company, None, None),
    )
    monkeypatch.setattr(
        "application.institutions.respond_to_change_request."
        "RespondToChangeRequestUseCase.execute",
        lambda _self, command: RespondToChangeRequestResult(
            success=True,
            booking_id=command.booking_id,
            change_request_id=command.change_request_id,
            status="ACCEPTED",
            redispatched=True,
            status_code=200,
            payload={"billing_outcome": command.billing_outcome},
        ),
    )

    with app.test_request_context(
        method="POST",
        json={
            "version": "3",
            "reason": "Validé",
            "billing_outcome": "NO_FEE",
            "billing_comment": "Aucun frais",
            "policy_version": "v1",
            "respond_context_version": "2",
            "rejection_reason_code": "OTHER",
            "situation": "confirmed",
            "suggested_outcome": {"amount": "12.50"},
            "cancelable_booking_ids": [companies_world["booking"].id],
        },
    ):
        body, status = companies_route._respond_to_change_request(
            companies_world["booking"].id,
            987,
            "accept",
        )

    assert status == 200
    assert body["success"] is True
    assert body["redispatched"] is True
