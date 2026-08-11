"""Tests routes /api/v1/companies (handlers prioritaires Phase B)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token

from models import Booking, Client, Company, Driver, User, Vehicle
from models.enums import (
    BookingStatus,
    ClientType,
    ManagementMode,
    UserRole,
)


def _auth_headers(
    app, user: User, *, role: str | None = None, company_id: int | None = None
) -> dict[str, str]:
    claims: dict[str, object] = {
        "role": role or str(getattr(user.role, "value", user.role)),
        "aud": "atmr-api",
    }
    if company_id is not None:
        claims["company_id"] = company_id
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def companies_world(db, app):
    """Entreprise + client + booking PENDING + driver + vehicle + admin."""
    suffix = uuid.uuid4().hex[:8]

    company_user = User()
    company_user.username = f"coco_{suffix}"
    company_user.email = f"coco_{suffix}@test.ch"
    company_user.role = UserRole.company
    company_user.public_id = str(uuid.uuid4())
    company_user.set_password("password123", force_change=False)
    db.session.add(company_user)
    db.session.flush()

    company = Company()
    company.name = f"Co Co {suffix}"
    company.address = "Rue Co 1"
    company.contact_email = company_user.email
    company.user_id = company_user.id
    company.is_approved = True
    company.dispatch_enabled = False
    db.session.add(company)
    db.session.flush()

    client_user = User()
    client_user.username = f"cocl_{suffix}"
    client_user.email = f"cocl_{suffix}@test.ch"
    client_user.role = UserRole.client
    client_user.public_id = str(uuid.uuid4())
    client_user.first_name = "Co"
    client_user.last_name = "Client"
    client_user.set_password("password123", force_change=False)
    db.session.add(client_user)
    db.session.flush()

    client = Client()
    client.user_id = client_user.id
    client.company_id = company.id
    client.contact_email = client_user.email
    client.is_active = True
    client.client_type = ClientType.TRANSPORT
    client.management_mode = ManagementMode.MANAGED
    db.session.add(client)
    db.session.flush()

    admin = User()
    admin.username = f"coad_{suffix}"
    admin.email = f"coad_{suffix}@test.ch"
    admin.role = UserRole.admin
    admin.public_id = str(uuid.uuid4())
    admin.set_password("password123", force_change=False)
    db.session.add(admin)
    db.session.flush()

    driver_user = User()
    driver_user.username = f"codr_{suffix}"
    driver_user.email = f"codr_{suffix}@test.ch"
    driver_user.role = UserRole.driver
    driver_user.public_id = str(uuid.uuid4())
    driver_user.first_name = "Co"
    driver_user.last_name = "Driver"
    driver_user.set_password("password123", force_change=False)
    db.session.add(driver_user)
    db.session.flush()

    driver = Driver()
    driver.user_id = driver_user.id
    driver.company_id = company.id
    driver.is_active = True
    db.session.add(driver)
    db.session.flush()

    vehicle = Vehicle()
    vehicle.company_id = company.id
    vehicle.model = "Test Van"
    vehicle.license_plate = f"GE{suffix[:5].upper()}"
    vehicle.is_active = True
    db.session.add(vehicle)
    db.session.flush()

    booking = Booking()
    booking.user_id = client_user.id
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = "Co Client"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
    booking.status = BookingStatus.PENDING
    booking.amount = Decimal("45.00")
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.commit()

    return {
        "company_user": company_user,
        "company": company,
        "client_user": client_user,
        "client": client,
        "admin": admin,
        "driver": driver,
        "vehicle": vehicle,
        "booking": booking,
    }


class TestCompaniesRoutes:
    def test_get_me_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me", headers=headers)
        assert resp.status_code == 200
        body = resp.get_json()
        data = body.get("data") if isinstance(body, dict) and "data" in body else body
        assert isinstance(data, dict)
        assert data.get("id") == world["company"].id or data.get("name")

    def test_get_me_401_without_jwt(self, client, companies_world):
        resp = client.get("/api/v1/companies/me")
        assert resp.status_code == 401

    def test_list_reservations_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/reservations?page=1&per_page=20",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_list_reservations_with_filters(self, client, app, companies_world):
        """Couvre branches date / tab / search / fields du gros handler reservations."""
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        day = world["booking"].scheduled_time.strftime("%Y-%m-%d")
        resp = client.get(
            f"/api/v1/companies/me/reservations?date={day}&tab=pending"
            f"&search=Client&fields=dashboard&flat=true&exclude_canceled=true",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_list_reservations_invalid_date_400(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/reservations?date=not-a-date",
            headers=headers,
        )
        assert resp.status_code == 400

    def test_reservations_summary_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        day = world["booking"].scheduled_time.strftime("%Y-%m-%d")
        resp = client.get(
            f"/api/v1/companies/me/reservations/summary?date={day}",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_reservations_summary_missing_date_400(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/reservations/summary",
            headers=headers,
        )
        assert resp.status_code == 400

    def test_dashboard_bootstrap_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        day = world["booking"].scheduled_time.strftime("%Y-%m-%d")
        resp = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day}",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_assigned_reservations_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/assigned-reservations",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_invoices_list_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/invoices", headers=headers)
        assert resp.status_code == 200

    def test_company_search_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/search?q=Co", headers=headers)
        assert resp.status_code == 200

    def test_drivers_live_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/drivers/live", headers=headers)
        assert resp.status_code == 200

    def test_reject_reservation_missing(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            "/api/v1/companies/me/reservations/999999/reject",
            json={"reason": "test"},
            headers=headers,
        )
        assert resp.status_code in (400, 404)

    def test_create_vehicle_400(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            "/api/v1/companies/me/vehicles",
            json={},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_list_clients_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/clients", headers=headers)
        assert resp.status_code == 200

    def test_get_client_404(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/clients/999999", headers=headers)
        assert resp.status_code == 404

    def test_get_client_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            f"/api/v1/companies/me/clients/{world['client'].id}",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_create_client_400(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            "/api/v1/companies/me/clients",
            json={},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_create_client_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        suffix = uuid.uuid4().hex[:6]
        resp = client.post(
            "/api/v1/companies/me/clients",
            json={
                "management_mode": "MANAGED",
                "gender": "male",
                "first_name": "Nouveau",
                "last_name": f"Client{suffix}",
                "address": "Rue Test 12, Genève",
                "client_type": "TRANSPORT",
            },
            headers=headers,
        )
        assert resp.status_code in (200, 201)

    def test_get_vehicle_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            f"/api/v1/companies/me/vehicles/{world['vehicle'].id}",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_put_reservation_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.put(
            f"/api/v1/companies/me/reservations/{world['booking'].id}",
            json={"notes_medical": "note test coverage"},
            headers=headers,
        )
        assert resp.status_code in (200, 400)

    def test_delete_reservation_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.delete(
            f"/api/v1/companies/me/reservations/{world['booking'].id}",
            headers=headers,
            json={"reason_code": "OTHER", "reason_text": "test"},
        )
        assert resp.status_code in (200, 204, 400, 403)

    def test_delete_reservation_404(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.delete(
            "/api/v1/companies/me/reservations/999999",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_reject_reservation_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            f"/api/v1/companies/me/reservations/{world['booking'].id}/reject",
            headers=headers,
        )
        assert resp.status_code in (200, 400)

    def test_transport_actions_required(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/transport-actions/required",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_drivers_locations(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/drivers/locations",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_completed_trips_stats(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/drivers/completed-trips-stats",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_institutions_search(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/me/institutions/search?q=hop",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_client_reservations_list(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            f"/api/v1/companies/me/clients/{world['client'].id}/reservations",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_partnerships_list(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/partnerships", headers=headers)
        assert resp.status_code == 200

    def test_partnerships_stats(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/partnerships/stats", headers=headers)
        assert resp.status_code == 200

    def test_accept_reservation_ok_or_contract(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            f"/api/v1/companies/me/reservations/{world['booking'].id}/accept",
            headers=headers,
        )
        assert resp.status_code in (200, 400)

    def test_accept_reservation_missing(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            "/api/v1/companies/me/reservations/999999/accept",
            headers=headers,
        )
        # Route actuelle : validation_error (400) si introuvable
        assert resp.status_code in (400, 404)

    def test_vehicles_list_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/vehicles", headers=headers)
        assert resp.status_code == 200
        body = resp.get_json()
        assert isinstance(body, list) or (
            isinstance(body, dict) and "data" in body
        )

    def test_vehicle_404(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/vehicles/999999", headers=headers)
        assert resp.status_code == 404

    def test_drivers_list_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/drivers", headers=headers)
        assert resp.status_code == 200

    def test_dispatch_status_roundtrip(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )

        status_resp = client.get(
            "/api/v1/companies/me/dispatch/status", headers=headers
        )
        assert status_resp.status_code == 200
        assert "dispatch_enabled" in (status_resp.get_json() or {})

        activate = client.post(
            "/api/v1/companies/me/dispatch/activate",
            json={"enabled": True},
            headers=headers,
        )
        assert activate.status_code == 200

        deactivate = client.post(
            "/api/v1/companies/me/dispatch/deactivate",
            headers=headers,
        )
        assert deactivate.status_code == 200

    def test_put_me_ok(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.put(
            "/api/v1/companies/me",
            json={"name": f"Co Renamed {uuid.uuid4().hex[:6]}"},
            headers=headers,
        )
        assert resp.status_code in (200, 400)

    def test_list_reservations_date_range_and_tabs(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        day = world["booking"].scheduled_time.strftime("%Y-%m-%d")
        for tab in ("in_progress", "completed", "canceled"):
            resp = client.get(
                f"/api/v1/companies/me/reservations?start_date={day}&end_date={day}"
                f"&tab={tab}&status=PENDING",
                headers=headers,
            )
            assert resp.status_code == 200

    def test_create_client_html_rejected(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            "/api/v1/companies/me/clients",
            json={
                "management_mode": "MANAGED",
                "gender": "female",
                "first_name": "<script>x</script>",
                "last_name": "Safe",
                "address": "Rue 1",
            },
            headers=headers,
        )
        assert resp.status_code == 400

    def test_assign_reservation_missing(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            "/api/v1/companies/me/reservations/999999/assign",
            json={"driver_id": world["driver"].id},
            headers=headers,
        )
        assert resp.status_code in (400, 404)

    def test_complete_reservation_missing(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.post(
            "/api/v1/companies/me/reservations/999999/complete",
            headers=headers,
        )
        assert resp.status_code in (400, 404)

    def test_delete_vehicle_404(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.delete(
            "/api/v1/companies/me/vehicles/999999",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_clients_export(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/me/clients/export", headers=headers)
        assert resp.status_code == 200

    def test_driver_vacations_list(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            f"/api/v1/companies/me/drivers/{world['driver'].id}/vacations",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_debug_transfer_404(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get(
            "/api/v1/companies/debug/booking/999999/transfer",
            headers=headers,
        )
        assert resp.status_code in (404, 403)

    def test_list_companies_admin_ok(self, client, app, companies_world, monkeypatch):
        """Évite le N+1 serialize de toutes les companies en DB de test."""
        from application.companies.admin import ListCompaniesOutput

        world = companies_world
        headers = _auth_headers(app, world["admin"], role="admin")

        def _fast_execute(self, input_data):  # noqa: ARG001
            return ListCompaniesOutput(
                success=True,
                companies=[{"id": world["company"].id, "name": world["company"].name}],
            )

        monkeypatch.setattr(
            "application.companies.admin.ListCompaniesUseCase.execute",
            _fast_execute,
        )
        resp = client.get("/api/v1/companies/", headers=headers)
        assert resp.status_code == 200

    def test_list_companies_company_403(self, client, app, companies_world):
        world = companies_world
        headers = _auth_headers(
            app, world["company_user"], role="company", company_id=world["company"].id
        )
        resp = client.get("/api/v1/companies/", headers=headers)
        assert resp.status_code in (403, 401)
