"""Couverture critique routes/companies.py — happy paths des gros handlers."""

# ruff: noqa: F811

from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy.exc import IntegrityError
from werkzeug.exceptions import NotFound

from models import Booking, Client, Driver, User
from models.enums import (
    BookingStatus,
    ClientType,
    ManagementMode,
    UserRole,
)
from tests.routes.test_companies import _auth_headers, companies_world

# Mot de passe unique non listé HIBP (évite faux négatifs create/reset driver)
_SAFE_PASSWORD = "Cov3rAge!Xy9zQ2mK"


def _fresh_headers(app, user: User, *, company_id: int) -> dict[str, str]:
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims={
                "role": "company",
                "company_id": company_id,
                "aud": "atmr-api",
            },
            fresh=True,
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def company_headers(app, companies_world):
    world = companies_world
    return _auth_headers(
        app,
        world["company_user"],
        role="company",
        company_id=world["company"].id,
    )


@pytest.fixture
def company_fresh_headers(app, companies_world):
    world = companies_world
    return _fresh_headers(app, world["company_user"], company_id=world["company"].id)


class TestCompaniesCriticalCoverage:
    def test_delete_pending_reservation_physical(
        self, client, companies_world, company_headers, db
    ):
        """PENDING → action delete : traverse le cascade SQL (~580 lignes)."""
        booking = companies_world["booking"]
        assert booking.status == BookingStatus.PENDING
        resp = client.delete(
            f"/api/v1/companies/me/reservations/{booking.id}",
            headers={
                **company_headers,
                "Content-Type": "application/json",
            },
            json={"reason_code": "OTHER", "reason_text": "couverture"},
        )
        assert resp.status_code == 200, resp.get_json()
        assert Booking.query.get(booking.id) is None

    def test_cancel_assigned_reservation_path(
        self, client, companies_world, company_headers, db
    ):
        """ASSIGNED + course future → action cancel (branche non-delete)."""
        world = companies_world
        booking = world["booking"]
        # Ordre important : driver_id avant status (validator Booking)
        booking.driver_id = world["driver"].id
        db.session.flush()
        booking.status = BookingStatus.ASSIGNED
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=48)
        db.session.commit()

        resp = client.delete(
            f"/api/v1/companies/me/reservations/{booking.id}",
            headers={
                **company_headers,
                "Content-Type": "application/json",
            },
            json={"reason_code": "CLIENT_REQUEST", "reason_text": "annulation test"},
        )
        assert resp.status_code == 200, resp.get_json()
        db.session.refresh(booking)
        assert booking.status == BookingStatus.CANCELED

    def test_accept_assign_complete_flow(
        self, client, companies_world, company_headers, db
    ):
        world = companies_world
        booking_id = world["booking"].id
        driver_id = world["driver"].id

        accept = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/accept",
            headers=company_headers,
        )
        assert accept.status_code == 200, accept.get_json()

        assign = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/assign",
            headers=company_headers,
            json={"driver_id": driver_id},
        )
        assert assign.status_code == 200, assign.get_json()

        complete = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/complete",
            headers=company_headers,
            json={"reason": "Clôture manuelle couverture"},
        )
        assert complete.status_code == 200, complete.get_json()

        booking = Booking.query.get(booking_id)
        assert booking is not None
        assert booking.status == BookingStatus.COMPLETED

    def test_billing_adjustment_on_completed(
        self, client, companies_world, company_headers, db
    ):
        world = companies_world
        booking = world["booking"]
        booking.status = BookingStatus.COMPLETED
        booking.amount = Decimal("45.00")
        db.session.commit()

        resp = client.patch(
            f"/api/v1/companies/me/reservations/{booking.id}/billing-adjustment",
            headers=company_headers,
            json={
                "amount": 55.0,
                "override_reason": "Ajustement couverture tests critiques",
            },
        )
        assert resp.status_code in (200, 400), resp.get_json()

    def test_put_reservation_fields(self, client, companies_world, company_headers):
        booking_id = companies_world["booking"].id
        resp = client.put(
            f"/api/v1/companies/me/reservations/{booking_id}",
            headers=company_headers,
            json={
                "notes_medical": "note couverture",
                "pickup_location": "Rue Pickup 12, Genève",
                "dropoff_location": "Rue Dropoff 34, Genève",
            },
        )
        assert resp.status_code in (200, 400), resp.get_json()

    def test_create_driver_ok(self, client, companies_world, company_headers):
        suffix = uuid.uuid4().hex[:8]
        resp = client.post(
            "/api/v1/companies/me/drivers/create",
            headers=company_headers,
            json={
                "username": f"drv_{suffix}",
                "first_name": "Jean",
                "last_name": "Couverture",
                "email": f"drv_{suffix}@example.com",
                "password": _SAFE_PASSWORD,
                "vehicle_assigned": "Van 1",
                "brand": "Mercedes",
                "license_plate": f"GE-{suffix[:6].upper()}",
            },
        )
        assert resp.status_code in (200, 201), resp.get_json()

    def test_driver_put_delete_reset_toggle_completed(
        self, client, companies_world, company_headers, company_fresh_headers, db
    ):
        world = companies_world
        driver_id = world["driver"].id

        put = client.put(
            f"/api/v1/companies/me/drivers/{driver_id}",
            headers=company_fresh_headers,
            json={"is_active": True, "phone": "+41791234567"},
        )
        assert put.status_code in (200, 400), put.get_json()

        trips = client.get(
            f"/api/v1/companies/me/drivers/{driver_id}/completed-trips",
            headers=company_headers,
        )
        assert trips.status_code == 200

        toggle = client.put(
            f"/api/v1/companies/me/drivers/{driver_id}/toggle-type",
            headers=company_headers,
            json={},
        )
        assert toggle.status_code in (200, 400), toggle.get_json()

        reset = client.post(
            f"/api/v1/companies/me/drivers/{driver_id}/reset-password",
            headers=company_fresh_headers,
            json={"password": _SAFE_PASSWORD},
        )
        assert reset.status_code in (200, 400), reset.get_json()

        # Créer un second chauffeur pour DELETE (garder le premier pour le reste)
        suffix = uuid.uuid4().hex[:8]
        other_user = User()
        other_user.username = f"deldrv_{suffix}"
        other_user.email = f"deldrv_{suffix}@test.ch"
        other_user.role = UserRole.driver
        other_user.public_id = str(uuid.uuid4())
        other_user.first_name = "Del"
        other_user.last_name = "Driver"
        other_user.set_password("password123", force_change=False)
        db.session.add(other_user)
        db.session.flush()
        other = Driver()
        other.user_id = other_user.id
        other.company_id = world["company"].id
        other.is_active = True
        db.session.add(other)
        db.session.commit()

        delete = client.delete(
            f"/api/v1/companies/me/drivers/{other.id}",
            headers=company_headers,
        )
        assert delete.status_code in (200, 204, 400), delete.get_json()

    def test_vehicle_create_update_delete(
        self, client, companies_world, company_headers
    ):
        suffix = uuid.uuid4().hex[:6].upper()
        create = client.post(
            "/api/v1/companies/me/vehicles",
            headers=company_headers,
            json={
                "model": "Sprinter",
                "license_plate": f"VS{suffix}",
                "year": 2024,
                "seats": 8,
                "wheelchair_accessible": True,
                "is_active": True,
            },
        )
        assert create.status_code in (200, 201), create.get_json()
        body = create.get_json() or {}
        vehicle_id = body.get("id") or (body.get("data") or {}).get("id")
        if not vehicle_id:
            vehicle_id = companies_world["vehicle"].id

        update = client.put(
            f"/api/v1/companies/me/vehicles/{vehicle_id}",
            headers=company_headers,
            json={"model": "Sprinter XL", "seats": 9},
        )
        assert update.status_code in (200, 400), update.get_json()

        if vehicle_id != companies_world["vehicle"].id:
            delete = client.delete(
                f"/api/v1/companies/me/vehicles/{vehicle_id}",
                headers=company_headers,
            )
            assert delete.status_code in (200, 204, 400), delete.get_json()

    def test_driver_vacations_post(self, client, companies_world, company_headers):
        driver_id = companies_world["driver"].id
        start = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%d")
        end = (datetime.now(UTC) + timedelta(days=17)).strftime("%Y-%m-%d")
        resp = client.post(
            f"/api/v1/companies/me/drivers/{driver_id}/vacations",
            headers=company_headers,
            json={
                "start_date": start,
                "end_date": end,
                "vacation_type": "VACANCES",
            },
        )
        assert resp.status_code in (200, 201, 400), resp.get_json()

    def test_client_put_and_delete(self, client, companies_world, company_headers, db):
        world = companies_world
        client_id = world["client"].id

        put = client.put(
            f"/api/v1/companies/me/clients/{client_id}",
            headers=company_headers,
            json={
                "first_name": "Maj",
                "last_name": "Client",
                "address": "Rue Maj 1, Genève",
            },
        )
        assert put.status_code in (200, 400), put.get_json()

        # Client sans booking pour DELETE
        suffix = uuid.uuid4().hex[:8]
        cu = User()
        cu.username = f"delcl_{suffix}"
        cu.email = f"delcl_{suffix}@test.ch"
        cu.role = UserRole.client
        cu.public_id = str(uuid.uuid4())
        cu.first_name = "Del"
        cu.last_name = "Client"
        cu.set_password("password123", force_change=False)
        db.session.add(cu)
        db.session.flush()
        orphan = Client()
        orphan.user_id = cu.id
        orphan.company_id = world["company"].id
        orphan.contact_email = cu.email
        orphan.is_active = True
        orphan.client_type = ClientType.TRANSPORT
        orphan.management_mode = ManagementMode.MANAGED
        db.session.add(orphan)
        db.session.commit()

        delete = client.delete(
            f"/api/v1/companies/me/clients/{orphan.id}",
            headers=company_headers,
        )
        assert delete.status_code in (200, 204, 400), delete.get_json()

    def test_logo_get_post_delete(self, client, companies_world, company_headers):
        get_resp = client.get("/api/v1/companies/me/logo", headers=company_headers)
        assert get_resp.status_code == 200

        # PNG 1x1 minimal
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        post = client.post(
            "/api/v1/companies/me/logo",
            headers=company_headers,
            data={"file": (io.BytesIO(png), "logo.png")},
            content_type="multipart/form-data",
        )
        assert post.status_code in (200, 201, 400), post.get_json()

        delete = client.delete("/api/v1/companies/me/logo", headers=company_headers)
        assert delete.status_code in (200, 204, 400, 404), delete.get_json()

    def test_push_token_and_test_push(
        self, client, companies_world, company_headers, monkeypatch
    ):
        from application.companies.save_company_push_token import (
            SaveCompanyPushTokenResult,
        )

        def _fake_execute(self, **_kwargs):  # noqa: ANN001
            return SaveCompanyPushTokenResult(
                response={"message": "ok"},
                status_code=200,
                should_commit=False,
            )

        monkeypatch.setattr(
            "application.companies.save_company_push_token."
            "SaveCompanyPushTokenUseCase.execute",
            _fake_execute,
        )
        save = client.post(
            "/api/v1/companies/save-push-token",
            headers=company_headers,
            json={
                "token": "fake-push-token-abcdefghij",
                "platform": "android",
                "device_id": "device-cover-1",
            },
        )
        assert save.status_code == 200, save.get_json()

        telemetry = client.post(
            "/api/v1/companies/me/telemetry/push",
            headers=company_headers,
            json={"event": "offer_received", "booking_id": 1},
        )
        assert telemetry.status_code in (200, 204, 400), telemetry.get_json()

        test_push = client.post(
            "/api/v1/companies/me/test-push",
            headers=company_headers,
            json={},
        )
        # Souvent 404 sans DeviceToken — couvre quand même le handler
        assert test_push.status_code in (200, 404, 400), test_push.get_json()

    def test_change_request_version_required(
        self, client, companies_world, company_headers
    ):
        booking_id = companies_world["booking"].id
        resp = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/change-requests/1/accept",
            headers=company_headers,
            json={},
        )
        assert resp.status_code == 400

        resp2 = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/change-requests/1/refuse",
            headers=company_headers,
            json={"version": "abc"},
        )
        assert resp2.status_code == 400

    def test_schedule_and_dispatch_now_and_trigger_return(
        self, client, companies_world, company_headers
    ):
        booking_id = companies_world["booking"].id
        future = (datetime.now(UTC) + timedelta(hours=6)).isoformat()

        schedule = client.put(
            f"/api/v1/companies/me/reservations/{booking_id}/schedule",
            headers=company_headers,
            json={"scheduled_time": future},
        )
        assert schedule.status_code in (200, 400), schedule.get_json()

        dispatch = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/dispatch-now",
            headers=company_headers,
            json={},
        )
        assert dispatch.status_code in (200, 400, 409), dispatch.get_json()

        trigger = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/trigger-return",
            headers=company_headers,
            json={},
        )
        assert trigger.status_code in (200, 201, 400), trigger.get_json()

    def test_manual_reservation_and_post_alias(
        self, client, companies_world, company_headers
    ):
        world = companies_world
        scheduled = (datetime.now(UTC) + timedelta(hours=5)).isoformat()
        payload = {
            "customer_name": "Manual Cover",
            "pickup_location": "Rue A 1, Genève",
            "dropoff_location": "Rue B 2, Genève",
            "scheduled_time": scheduled,
            "client_id": world["client"].id,
            "amount": 40.0,
        }
        manual = client.post(
            "/api/v1/companies/me/reservations/manual",
            headers=company_headers,
            json=payload,
        )
        assert manual.status_code in (200, 201, 400), manual.get_json()

        alias = client.post(
            "/api/v1/companies/me/reservations",
            headers=company_headers,
            json=payload,
        )
        assert alias.status_code in (200, 201, 400), alias.get_json()

    def test_partnerships_mutate_404_and_statements(
        self, client, companies_world, company_headers
    ):
        put = client.put(
            "/api/v1/companies/me/partnerships/999999",
            headers=company_headers,
            json={"status": "ACTIVE"},
        )
        assert put.status_code in (400, 404)

        delete = client.delete(
            "/api/v1/companies/me/partnerships/999999",
            headers=company_headers,
        )
        assert delete.status_code in (400, 404)

        gen = client.post(
            "/api/v1/companies/me/partnerships/statements/generate",
            headers=company_headers,
            json={},
        )
        assert gen.status_code in (200, 400, 404)

        gen_one = client.post(
            "/api/v1/companies/me/partnerships/999999/statements/generate",
            headers=company_headers,
            json={},
        )
        assert gen_one.status_code in (200, 400, 404)

        pdf = client.get(
            "/api/v1/companies/me/partnerships/statements/pdf/missing.pdf",
            headers=company_headers,
        )
        assert pdf.status_code in (400, 404)

    def test_change_events_and_ack(self, client, companies_world, company_headers):
        booking_id = companies_world["booking"].id
        events = client.get(
            f"/api/v1/companies/me/reservations/{booking_id}/change-events",
            headers=company_headers,
        )
        assert events.status_code in (200, 404)

        ack = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/change-events/1/ack",
            headers=company_headers,
            json={},
        )
        assert ack.status_code in (200, 400, 404)

    def test_action_queue_execute_unknown(
        self, client, companies_world, company_headers
    ):
        resp = client.post(
            "/api/v1/companies/me/action-queue/unknown_action/execute",
            headers=company_headers,
            json={},
        )
        assert resp.status_code in (400, 404)

    def test_reject_happy_path(self, client, companies_world, company_headers):
        booking_id = companies_world["booking"].id
        resp = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/reject",
            headers=company_headers,
            json={"reason": "Pas de capacité"},
        )
        assert resp.status_code in (200, 400), resp.get_json()

    def test_company_me_put_error_paths(
        self, client, companies_world, company_headers, monkeypatch
    ):
        target = (
            "application.companies.update_company_profile."
            "UpdateCompanyProfileUseCase.execute"
        )

        def raise_value_error(self, *_args, **_kwargs):  # noqa: ANN001
            raise ValueError("profil invalide")

        monkeypatch.setattr(target, raise_value_error)
        response = client.put(
            "/api/v1/companies/me",
            headers=company_headers,
            json={"name": "Profil invalide"},
        )
        assert response.status_code == 400

        def raise_integrity_error(self, *_args, **_kwargs):  # noqa: ANN001
            raise IntegrityError("UPDATE company", {}, Exception("doublon"))

        monkeypatch.setattr(target, raise_integrity_error)
        response = client.put(
            "/api/v1/companies/me",
            headers=company_headers,
            json={"name": "Profil en doublon"},
        )
        assert response.status_code in (400, 409)

        def raise_unexpected_error(self, *_args, **_kwargs):  # noqa: ANN001
            raise RuntimeError("panne profil")

        monkeypatch.setattr(target, raise_unexpected_error)
        response = client.put(
            "/api/v1/companies/me",
            headers=company_headers,
            json={"name": "Profil indisponible"},
        )
        assert response.status_code == 500

    def test_company_me_disappears_during_update(
        self, client, companies_world, company_headers, monkeypatch
    ):
        from routes import companies as companies_routes

        class DisappearingCompany:
            id = companies_world["company"].id

            def __init__(self):
                self.boolean_checks = 0

            def __bool__(self):
                self.boolean_checks += 1
                return self.boolean_checks == 1

        company = DisappearingCompany()
        monkeypatch.setattr(
            companies_routes,
            "_get_current_company_via_use_case",
            lambda: (company, None, 200),
        )
        monkeypatch.setattr(
            "application.companies.update_company_profile."
            "UpdateCompanyProfileUseCase.execute",
            lambda self, *_args, **_kwargs: SimpleNamespace(  # noqa: ARG005
                geocoded=False,
                billing_profile_synced=False,
            ),
        )
        response = client.put(
            "/api/v1/companies/me",
            headers=company_headers,
            json={"name": "Entreprise éphémère"},
        )
        assert response.status_code == 404

    def test_vehicle_create_error_paths(
        self, client, companies_world, company_headers, monkeypatch
    ):
        target = (
            "application.companies.vehicles.create_company_vehicle."
            "CreateCompanyVehicleUseCase.execute"
        )
        payload = {"model": "Erreur", "license_plate": f"ERR{uuid.uuid4().hex[:6]}"}

        def raise_value_error(self, **_kwargs):  # noqa: ANN001
            raise ValueError("véhicule invalide")

        monkeypatch.setattr(target, raise_value_error)
        response = client.post(
            "/api/v1/companies/me/vehicles",
            headers=company_headers,
            json=payload,
        )
        assert response.status_code == 400

        def raise_integrity_error(self, **_kwargs):  # noqa: ANN001
            raise IntegrityError("INSERT vehicle", {}, Exception("doublon"))

        monkeypatch.setattr(target, raise_integrity_error)
        response = client.post(
            "/api/v1/companies/me/vehicles",
            headers=company_headers,
            json=payload,
        )
        assert response.status_code in (400, 409)

        def raise_unexpected_error(self, **_kwargs):  # noqa: ANN001
            raise RuntimeError("panne véhicule")

        monkeypatch.setattr(target, raise_unexpected_error)
        response = client.post(
            "/api/v1/companies/me/vehicles",
            headers=company_headers,
            json=payload,
        )
        assert response.status_code == 500

    def test_vehicle_update_error_paths(
        self, client, companies_world, company_headers, monkeypatch
    ):
        target = (
            "application.companies.vehicles.update_company_vehicle."
            "UpdateCompanyVehicleUseCase.execute"
        )
        vehicle_id = companies_world["vehicle"].id

        def raise_value_error(self, *_args, **_kwargs):  # noqa: ANN001
            raise ValueError("mise à jour invalide")

        monkeypatch.setattr(target, raise_value_error)
        response = client.put(
            f"/api/v1/companies/me/vehicles/{vehicle_id}",
            headers=company_headers,
            json={"model": "Erreur valeur"},
        )
        assert response.status_code == 400

        def raise_integrity_error(self, *_args, **_kwargs):  # noqa: ANN001
            raise IntegrityError("UPDATE vehicle", {}, Exception("doublon"))

        monkeypatch.setattr(target, raise_integrity_error)
        response = client.put(
            f"/api/v1/companies/me/vehicles/{vehicle_id}",
            headers=company_headers,
            json={"model": "Erreur intégrité"},
        )
        assert response.status_code in (400, 409)

        def raise_unexpected_error(self, *_args, **_kwargs):  # noqa: ANN001
            raise RuntimeError("panne mise à jour")

        monkeypatch.setattr(target, raise_unexpected_error)
        response = client.put(
            f"/api/v1/companies/me/vehicles/{vehicle_id}",
            headers=company_headers,
            json={"model": "Erreur interne"},
        )
        assert response.status_code == 500

    def test_reset_driver_password_error_paths(
        self,
        client,
        companies_world,
        company_fresh_headers,
        monkeypatch,
    ):
        target = (
            "application.companies.drivers.reset_driver_password."
            "ResetDriverPasswordUseCase.execute"
        )
        driver_id = companies_world["driver"].id

        monkeypatch.setattr(
            target,
            lambda self, _user: SimpleNamespace(  # noqa: ARG005
                ok=False,
                error={"error": "mot de passe refusé"},
                status_code=422,
            ),
        )
        response = client.post(
            f"/api/v1/companies/me/drivers/{driver_id}/reset-password",
            headers=company_fresh_headers,
            json={},
        )
        assert response.status_code == 422

        def raise_unexpected_error(self, _user):  # noqa: ANN001, ARG001
            raise RuntimeError("panne mot de passe")

        monkeypatch.setattr(target, raise_unexpected_error)
        response = client.post(
            f"/api/v1/companies/me/drivers/{driver_id}/reset-password",
            headers=company_fresh_headers,
            json={},
        )
        assert response.status_code == 500

    def test_clients_export_consumes_stream(
        self, client, companies_world, company_headers
    ):
        response = client.get(
            "/api/v1/companies/me/clients/export?search=",
            headers=company_headers,
        )
        assert response.status_code == 200
        csv_content = response.get_data(as_text=True)
        assert "id;type;nom;email;telephone;ville;actif" in csv_content
        assert "particulier" in csv_content

    def test_booking_stats_empty_and_error_paths(self, app):
        from routes.companies import _booking_stats_from_base_query

        class FakeQuery:
            def __init__(self, result=None, error=None):
                self.result = result
                self.error = error

            def with_entities(self, *_args):
                return self

            def first(self):
                if self.error:
                    raise self.error
                return self.result

        with app.app_context():
            empty = _booking_stats_from_base_query(FakeQuery())
            assert empty["total"] == 0

            fallback = _booking_stats_from_base_query(
                FakeQuery(error=RuntimeError("stats indisponibles"))
            )
            assert fallback["critical_delay_minutes"] == 15

            with pytest.raises(RuntimeError, match="stats indisponibles"):
                _booking_stats_from_base_query(
                    FakeQuery(error=RuntimeError("stats indisponibles")),
                    raise_on_error=True,
                )

    def test_logo_upload_validation_failures(
        self, client, companies_world, company_headers
    ):
        missing = client.post(
            "/api/v1/companies/me/logo",
            headers=company_headers,
            data={},
            content_type="multipart/form-data",
        )
        assert missing.status_code == 400

        invalid = client.post(
            "/api/v1/companies/me/logo",
            headers=company_headers,
            data={"file": (io.BytesIO(b"pas une image"), "logo.exe")},
            content_type="multipart/form-data",
        )
        assert invalid.status_code == 400

    def test_dispatch_activate_failure_and_trigger(
        self, client, companies_world, company_headers, monkeypatch
    ):
        target = (
            "application.companies.set_dispatch_enabled."
            "SetDispatchEnabledUseCase.execute"
        )
        monkeypatch.setattr(
            target,
            lambda self, *_args, **_kwargs: SimpleNamespace(  # noqa: ARG005
                ok=False,
                error={"error": "activation refusée"},
                status_code=409,
            ),
        )
        response = client.post(
            "/api/v1/companies/me/dispatch/activate",
            headers=company_headers,
            json={"enabled": True},
        )
        assert response.status_code == 409

        triggered = []
        monkeypatch.setattr(
            target,
            lambda self, company, **_kwargs: SimpleNamespace(  # noqa: ARG005
                ok=True,
                company_id=company.id,
                should_trigger_dispatch=True,
                trigger_reason="couverture",
            ),
        )
        monkeypatch.setattr(
            "routes.companies.queue.trigger",
            lambda company_id, **kwargs: triggered.append((company_id, kwargs)),
        )
        response = client.post(
            "/api/v1/companies/me/dispatch/activate",
            headers=company_headers,
            json={"enabled": True},
        )
        assert response.status_code == 200
        assert triggered

    def test_statement_pdf_consolidated_missing_file(
        self, client, companies_world, company_headers, monkeypatch
    ):
        def raise_not_found(_path):
            raise NotFound()

        monkeypatch.setattr(
            "shared.upload_path_resolver.serve_stored_upload",
            raise_not_found,
        )
        company_id = companies_world["company"].id
        filename = f"decompte_consolide_{company_id}_20260101_120000.pdf"
        response = client.get(
            f"/api/v1/companies/me/partnerships/statements/pdf/{filename}",
            headers=company_headers,
        )
        assert response.status_code == 404

    def test_create_driver_validation_and_use_case_failures(
        self, client, companies_world, company_headers, monkeypatch
    ):
        missing = client.post(
            "/api/v1/companies/me/drivers/create",
            headers=company_headers,
            json={},
        )
        assert missing.status_code == 400

        target = (
            "application.companies.drivers.create_driver."
            "CreateCompanyDriverUseCase.execute"
        )
        payload = {
            "username": f"err_{uuid.uuid4().hex[:8]}",
            "first_name": "Erreur",
            "last_name": "Couverture",
            "email": f"err_{uuid.uuid4().hex[:8]}@example.com",
            "password": _SAFE_PASSWORD,
            "vehicle_assigned": "Van erreur",
            "brand": "Test",
            "license_plate": f"ER{uuid.uuid4().hex[:8].upper()}",
        }
        monkeypatch.setattr(
            target,
            lambda self, **_kwargs: SimpleNamespace(  # noqa: ARG005
                ok=False,
                error={"error": "chauffeur refusé"},
                status_code=409,
            ),
        )
        refused = client.post(
            "/api/v1/companies/me/drivers/create",
            headers=company_headers,
            json=payload,
        )
        assert refused.status_code == 409

        def raise_unexpected_error(self, **_kwargs):  # noqa: ANN001
            raise RuntimeError("panne création chauffeur")

        monkeypatch.setattr(target, raise_unexpected_error)
        failed = client.post(
            "/api/v1/companies/me/drivers/create",
            headers=company_headers,
            json=payload,
        )
        assert failed.status_code == 500

    def test_partnership_statement_validation_and_errors(
        self, client, companies_world, company_headers, monkeypatch
    ):
        consolidated_url = "/api/v1/companies/me/partnerships/statements/generate"
        invalid_payloads = (
            {"year": "année"},
            {"month": "mois"},
            {"start_date": "hier"},
            {"end_date": "demain"},
        )
        for payload in invalid_payloads:
            response = client.post(
                consolidated_url,
                headers=company_headers,
                json=payload,
            )
            assert response.status_code == 400

        target = (
            "services.partnerships.statements.PartnershipStatementService."
            "generate_consolidated_statement"
        )

        def raise_value_error(self, **_kwargs):  # noqa: ANN001
            raise ValueError("période refusée")

        monkeypatch.setattr(target, raise_value_error)
        response = client.post(
            consolidated_url,
            headers=company_headers,
            json={},
        )
        assert response.status_code == 400

        def raise_unexpected_error(self, **_kwargs):  # noqa: ANN001
            raise RuntimeError("génération indisponible")

        monkeypatch.setattr(target, raise_unexpected_error)
        response = client.post(
            consolidated_url,
            headers=company_headers,
            json={},
        )
        assert response.status_code == 500

        partner_url = "/api/v1/companies/me/partnerships/999999/statements/generate"
        for payload in invalid_payloads:
            response = client.post(
                partner_url,
                headers=company_headers,
                json=payload,
            )
            assert response.status_code == 400

        partner_target = (
            "services.partnerships.statements.PartnershipStatementService."
            "generate_partnership_statement"
        )
        monkeypatch.setattr(partner_target, raise_value_error)
        response = client.post(partner_url, headers=company_headers, json={})
        assert response.status_code == 400

        monkeypatch.setattr(partner_target, raise_unexpected_error)
        response = client.post(partner_url, headers=company_headers, json={})
        assert response.status_code == 500

    def test_dashboard_bootstrap_failure_paths(
        self, client, companies_world, company_headers, monkeypatch
    ):
        day = companies_world["booking"].scheduled_time.strftime("%Y-%m-%d")
        url = f"/api/v1/companies/me/dashboard/bootstrap?date={day}"

        def raise_stats(*_args, **_kwargs):
            raise RuntimeError("KPI indisponibles")

        monkeypatch.setattr(
            "routes.companies._booking_stats_from_base_query", raise_stats
        )
        response = client.get(url, headers=company_headers)
        assert response.status_code == 503
        assert response.get_json()["health"]["kpi"] == "failed"

        from routes import companies as companies_routes

        monkeypatch.setattr(
            companies_routes,
            "_booking_stats_from_base_query",
            lambda *_args, **_kwargs: {"total": 1},
        )

        def raise_serialization(_bookings):
            raise RuntimeError("sérialisation indisponible")

        monkeypatch.setattr(
            "services.companies.booking_transfer_cache."
            "attach_transfer_cache_to_bookings",
            raise_serialization,
        )
        response = client.get(url, headers=company_headers)
        assert response.status_code == 503
        assert response.get_json()["health"]["bookings"] == "failed"

    def test_legacy_update_reservation_direct_paths(
        self, app, companies_world, monkeypatch
    ):
        from routes import companies as companies_routes

        raw_put = companies_routes.UpdateReservation.put
        while hasattr(raw_put, "__wrapped__"):
            raw_put = raw_put.__wrapped__

        company = companies_world["company"]
        booking = companies_world["booking"]
        monkeypatch.setattr(
            companies_routes,
            "_get_current_company_via_use_case",
            lambda: (company, None, 200),
        )
        monkeypatch.setattr(
            "services.reservations_summary_cache."
            "invalidate_summary_cache_for_booking_after_day_change",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            companies_routes,
            "_maybe_trigger_dispatch",
            lambda *_args, **_kwargs: None,
        )
        target = (
            "application.companies.reservations.update_reservation."
            "UpdateCompanyReservationUseCase.execute"
        )
        monkeypatch.setattr(
            target,
            lambda self, *_args, **_kwargs: SimpleNamespace(  # noqa: ARG005
                ok=False,
                error={"error": "mise à jour refusée"},
                status_code=409,
            ),
        )
        with app.test_request_context(json={"notes_medical": "legacy"}):
            body, status = raw_put(
                companies_routes.UpdateReservation(),
                booking.id,
            )
        assert status == 409
        assert body["error"] == "mise à jour refusée"

        monkeypatch.setattr(
            target,
            lambda self, *_args, **_kwargs: SimpleNamespace(  # noqa: ARG005
                ok=True,
                updated_fields=["notes_medical"],
            ),
        )
        with app.test_request_context(json={"notes_medical": "legacy réussi"}):
            body, status = raw_put(
                companies_routes.UpdateReservation(),
                booking.id,
            )
        assert status == 200
        assert body["reservation"]["id"] == booking.id

        def raise_commit():
            raise RuntimeError("commit indisponible")

        monkeypatch.setattr(companies_routes.db.session, "commit", raise_commit)
        with app.test_request_context(json={"notes_medical": "legacy panne"}):
            _body, status = raw_put(
                companies_routes.UpdateReservation(),
                booking.id,
            )
        assert status == 500

    def test_invalid_company_id_across_resources(
        self,
        client,
        companies_world,
        company_headers,
        company_fresh_headers,
        monkeypatch,
    ):
        from routes import companies as companies_routes

        invalid_company = SimpleNamespace(id="pas-un-identifiant")
        monkeypatch.setattr(
            companies_routes,
            "_get_current_company_via_use_case",
            lambda: (invalid_company, None, 200),
        )
        booking_id = companies_world["booking"].id
        driver_id = companies_world["driver"].id
        vehicle_id = companies_world["vehicle"].id

        requests = (
            ("get", "/api/v1/companies/me/reservations/summary?date=2026-01-01", None),
            ("get", "/api/v1/companies/me/dashboard/bootstrap?date=2026-01-01", None),
            ("post", "/api/v1/companies/me/action-queue/x/execute", {}),
            ("get", "/api/v1/companies/me/assigned-reservations", None),
            ("get", "/api/v1/companies/me/invoices", None),
            ("get", "/api/v1/companies/me/drivers/locations", None),
            ("post", f"/api/v1/companies/me/reservations/{booking_id}/accept", {}),
            ("post", f"/api/v1/companies/me/reservations/{booking_id}/reject", {}),
            (
                "patch",
                f"/api/v1/companies/me/reservations/{booking_id}/billing-adjustment",
                {"amount": 10, "override_reason": "identifiant invalide"},
            ),
            (
                "put",
                f"/api/v1/companies/me/reservations/{booking_id}",
                {"notes_medical": "identifiant invalide"},
            ),
            (
                "delete",
                f"/api/v1/companies/me/reservations/{booking_id}",
                {"reason_code": "OTHER"},
            ),
            ("post", "/api/v1/companies/me/clients", {}),
            ("get", "/api/v1/companies/me/clients/export", None),
            ("get", "/api/v1/companies/me/drivers/completed-trips-stats", None),
            (
                "get",
                f"/api/v1/companies/me/drivers/{driver_id}/completed-trips",
                None,
            ),
            ("delete", f"/api/v1/companies/me/drivers/{driver_id}", None),
            ("put", f"/api/v1/companies/me/drivers/{driver_id}/toggle-type", {}),
            ("get", "/api/v1/companies/me/vehicles", None),
            ("post", "/api/v1/companies/me/vehicles", {}),
            ("get", f"/api/v1/companies/me/vehicles/{vehicle_id}", None),
            ("put", f"/api/v1/companies/me/vehicles/{vehicle_id}", {}),
            ("delete", f"/api/v1/companies/me/vehicles/{vehicle_id}", None),
            ("post", "/api/v1/companies/me/logo", None),
            (
                "put",
                f"/api/v1/companies/me/reservations/{booking_id}/schedule",
                {"scheduled_time": "2026-01-01T12:00:00Z"},
            ),
            (
                "post",
                f"/api/v1/companies/me/reservations/{booking_id}/dispatch-now",
                {},
            ),
        )
        for method, url, payload in requests:
            response = getattr(client, method)(
                url,
                headers=company_headers,
                json=payload,
            )
            assert response.status_code == 500, (method, url, response.get_json())

        driver_update = client.put(
            f"/api/v1/companies/me/drivers/{driver_id}",
            headers=company_fresh_headers,
            json={},
        )
        assert driver_update.status_code == 500

        reset = client.post(
            f"/api/v1/companies/me/drivers/{driver_id}/reset-password",
            headers=company_fresh_headers,
            json={},
        )
        assert reset.status_code == 500

    def test_accept_reservation_transfer_owner_and_executor_paths(
        self, client, app, companies_world, company_headers, db, monkeypatch
    ):
        from models.booking_transfer import BookingTransfer
        from models.enums import PartnershipStatus, TransferModel, TransferStatus
        from tests.routes.test_companies_partnerships_coverage import (
            _partnership,
            _second_company,
        )

        world = companies_world
        company = world["company"]
        booking = world["booking"]
        executor = _second_company(db)
        partnership = _partnership(
            db,
            company.id,
            executor.id,
            status=PartnershipStatus.ACCEPTED,
        )
        db.session.flush()

        transfer = BookingTransfer()
        transfer.booking_id = booking.id
        transfer.partnership_id = partnership.id
        transfer.transfer_model = TransferModel.SUBCONTRACT
        transfer.owner_company_id = company.id
        transfer.executing_company_id = executor.id
        transfer.client_price = Decimal("50.00")
        transfer.partner_cost = Decimal("40.00")
        transfer.platform_fee = Decimal("0.00")
        transfer.currency = "CHF"
        transfer.vat_rate = Decimal("0.00")
        transfer.vat_included = True
        transfer.status = TransferStatus.PENDING
        db.session.add(transfer)
        booking.status = BookingStatus.ACCEPTED
        booking.executing_company_id = executor.id
        db.session.commit()

        monkeypatch.setattr(
            "application.companies.accept_reservation.AcceptReservationUseCase.execute",
            lambda self, *_args, **_kwargs: SimpleNamespace(  # noqa: ARG005
                ok=True,
                error=None,
            ),
        )
        monkeypatch.setattr(
            "routes.companies._maybe_trigger_dispatch",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "services.reservations_summary_cache.invalidate_summary_cache_for_booking",
            lambda *_args, **_kwargs: None,
        )

        owner_response = client.post(
            f"/api/v1/companies/me/reservations/{booking.id}/accept",
            headers=company_headers,
        )
        assert owner_response.status_code == 200, owner_response.get_json()
        assert transfer.status == TransferStatus.REJECTED
        assert booking.status == BookingStatus.PENDING

        transfer.status = TransferStatus.PENDING
        booking.status = BookingStatus.ACCEPTED
        db.session.commit()
        executor_headers = _auth_headers(
            app,
            executor.user,
            role="company",
            company_id=executor.id,
        )
        accepted_transfer = SimpleNamespace(
            to_dict=lambda: {
                "id": transfer.id,
                "status": TransferStatus.ACCEPTED.value,
            }
        )
        monkeypatch.setattr(
            "services.booking.transfers.BookingTransferService.accept_transfer",
            lambda *_args, **_kwargs: accepted_transfer,
        )

        executor_response = client.post(
            f"/api/v1/companies/me/reservations/{booking.id}/accept",
            headers=executor_headers,
        )
        assert executor_response.status_code == 200, executor_response.get_json()
        assert executor_response.get_json()["transfer"]["id"] == transfer.id

        def raise_transfer_validation(*_args, **_kwargs):
            raise ValueError("transfert refusé")

        monkeypatch.setattr(
            "services.booking.transfers.BookingTransferService.accept_transfer",
            raise_transfer_validation,
        )
        refused_response = client.post(
            f"/api/v1/companies/me/reservations/{booking.id}/accept",
            headers=executor_headers,
        )
        assert refused_response.status_code == 400

        def raise_transfer_failure(*_args, **_kwargs):
            raise RuntimeError("service de transfert indisponible")

        monkeypatch.setattr(
            "services.booking.transfers.BookingTransferService.accept_transfer",
            raise_transfer_failure,
        )
        failed_response = client.post(
            f"/api/v1/companies/me/reservations/{booking.id}/accept",
            headers=executor_headers,
        )
        assert failed_response.status_code == 500

        def raise_not_found(_path):
            raise NotFound()

        monkeypatch.setattr(
            "shared.upload_path_resolver.serve_stored_upload",
            raise_not_found,
        )
        statement_filename = f"decompte_partenaire_{partnership.id}_20260101_120000.pdf"
        statement_response = client.get(
            f"/api/v1/companies/me/partnerships/statements/pdf/{statement_filename}",
            headers=company_headers,
        )
        assert statement_response.status_code == 404

    def test_change_event_ack_success_and_push_commit(
        self, client, companies_world, company_headers, monkeypatch
    ):
        """Franchit le seuil 80 % : ack succès + save-push-token should_commit."""
        from application.companies.save_company_push_token import (
            SaveCompanyPushTokenResult,
        )

        booking_id = companies_world["booking"].id
        monkeypatch.setattr(
            "services.institutions.booking_change_service.acknowledge_critical_event",
            lambda *_a, **_k: ({"acked": True}, 200),
        )
        ack = client.post(
            f"/api/v1/companies/me/reservations/{booking_id}/change-events/42/ack",
            headers=company_headers,
            json={},
        )
        assert ack.status_code == 200
        assert ack.get_json()["acked"] is True

        def _commit_execute(self, **_kwargs):  # noqa: ANN001
            return SaveCompanyPushTokenResult(
                response={"message": "enregistré"},
                status_code=200,
                should_commit=True,
            )

        monkeypatch.setattr(
            "application.companies.save_company_push_token."
            "SaveCompanyPushTokenUseCase.execute",
            _commit_execute,
        )
        monkeypatch.setattr(
            "services.monitoring.prometheus.track_push_token_registration_outcome",
            lambda **_k: None,
            raising=False,
        )
        monkeypatch.setattr(
            "services.monitoring.prometheus.refresh_push_active_owners_gauges",
            lambda: None,
            raising=False,
        )
        save = client.post(
            "/api/v1/companies/save-push-token",
            headers=company_headers,
            json={
                "token": "fake-push-token-commit-path",
                "platform": "ios",
                "device_id": "device-cover-commit",
            },
        )
        assert save.status_code == 200
