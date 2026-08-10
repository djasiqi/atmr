"""Lot 3 perf espace entreprise — GET /companies/me/dashboard/bootstrap.

Vérifie la forme de la réponse agrégée (KPI + réservations + mode dispatch +
notifications + curseur temps réel) et la présence de `snapshot_cursor`
(entier monotone, pas un timestamp) — voir docs/perf-company-space-lot3-dashboard.md.
"""

from __future__ import annotations

import uuid

import pytest
from flask_jwt_extended import create_access_token

from models import Company, User, UserRole
from models.booking import Booking
from models.enums import BookingStatus
from shared.time_utils import now_local


def _company_headers(client, user, company_id: int) -> dict[str, str]:
    claims = {
        "role": user.role.value,
        "company_id": company_id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def bootstrap_company(db, sample_user):
    existing = Company.query.filter_by(user_id=sample_user.id).first()
    if existing:
        return existing
    company = Company()
    company.name = "Entreprise Bootstrap"
    company.user_id = sample_user.id
    company.address = "Rue Bootstrap 1"
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    db.session.refresh(company)
    return company


@pytest.fixture
def bootstrap_booking(db, bootstrap_company):
    booking = Booking()
    booking.customer_name = f"Client Bootstrap {uuid.uuid4().hex[:6]}"
    booking.pickup_location = "Rue Alpha 1, Genève"
    booking.dropoff_location = "Rue Beta 2, Genève"
    booking.pickup_lat = 46.2
    booking.pickup_lon = 6.1
    booking.dropoff_lat = 46.21
    booking.dropoff_lon = 6.15
    booking.booking_type = "standard"
    booking.scheduled_time = now_local()
    booking.amount = 42.0
    booking.status = BookingStatus.PENDING
    booking.company_id = bootstrap_company.id
    db.session.add(booking)
    db.session.flush()
    db.session.refresh(booking)
    return booking


class TestCompanyDashboardBootstrap:
    def test_requires_auth(self, client):
        response = client.get("/api/v1/companies/me/dashboard/bootstrap")
        assert response.status_code == 401

    def test_returns_bootstrap_shape_with_snapshot_cursor(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")

        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )

        assert response.status_code == 200
        data = response.get_json()

        # Enveloppe attendue par le frontend (CompanyDashboard critical path).
        assert data["schema_version"] == 1
        assert isinstance(data["generated_at"], str) and data["generated_at"]
        assert data["date"] == day_str
        assert data["company_id"] == bootstrap_company.id

        # snapshot_cursor : curseur entier monotone (Redis INCR), pas updated_at.
        assert "snapshot_cursor" in data
        assert isinstance(data["snapshot_cursor"], int)
        assert data["snapshot_cursor"] >= 0

        # KPI du jour (mêmes agrégats que /me/reservations/summary).
        kpi = data["kpi"]
        for key in (
            "total",
            "pending",
            "inProgress",
            "completed",
            "canceled",
            "revenue",
        ):
            assert key in kpi
        assert kpi["total"] >= 1
        assert kpi["pending"] >= 1

        # Projection réservations (mêmes champs que fields=dashboard).
        bookings = data["bookings"]
        assert isinstance(bookings, list)
        assert any(b["id"] == bootstrap_booking.id for b in bookings)
        matched = next(b for b in bookings if b["id"] == bootstrap_booking.id)
        assert matched["status"] == "pending"
        assert "client_name" in matched

        # Mode dispatch + notifications (résumés chrome).
        assert data["dispatch_mode"] in (
            "manual",
            "semi_auto",
            "fully_auto",
            "autonomous",
        )
        assert isinstance(data["notifications"]["unread_count"], int)

        # KPI opérationnels étendus (PR2) — jamais absents même si zéro.
        for key in (
            "pending_decision",
            "unassigned",
            "in_service",
            "delay_count",
            "critical_delay_count",
            "critical_delay_minutes",
        ):
            assert key in kpi
        assert kpi["critical_delay_minutes"] == 15

        # Health explicite (PR1) — jamais absent, jamais une valeur trompeuse.
        assert data["health"]["realtime_sequence"] in ("ok", "degraded")
        assert data["health"]["notifications"] in ("ok", "degraded")

        # Troncature explicite (PR2).
        assert data["bookings_truncated"] is False
        assert data["bookings_limit"] >= 1
        assert data["bookings_returned"] == len(bookings)
        assert data["bookings_total"] >= 1

    def test_defaults_to_today_without_date_param(
        self, client, sample_user, bootstrap_company
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        response = client.get(
            "/api/v1/companies/me/dashboard/bootstrap", headers=headers
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["date"] == now_local().strftime("%Y-%m-%d")

    def test_rejects_invalid_date_format(self, client, sample_user, bootstrap_company):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        response = client.get(
            "/api/v1/companies/me/dashboard/bootstrap?date=not-a-date",
            headers=headers,
        )
        assert response.status_code == 400

    def test_company_a_does_not_see_company_b_bookings(
        self, client, db, sample_user, bootstrap_company, bootstrap_booking
    ):
        """Isolation multi-tenant : une autre entreprise ne voit pas ces réservations."""
        uid = str(uuid.uuid4())[:8]
        other_user = User()
        other_user.username = f"company_b_{uid}"
        other_user.email = f"company-b-{uid}@test.ch"
        other_user.role = UserRole.company
        other_user.public_id = str(uuid.uuid4())
        other_user.set_password("password123", force_change=False)
        db.session.add(other_user)
        db.session.flush()

        other_company = Company()
        other_company.name = "Entreprise B Bootstrap"
        other_company.user_id = other_user.id
        other_company.address = "Rue B 1"
        other_company.is_approved = True
        db.session.add(other_company)
        db.session.flush()

        headers = _company_headers(client, other_user, other_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")
        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["kpi"]["total"] == 0
        assert data["bookings"] == []

    def test_extended_kpi_fields_and_health_ok(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        """KPI étendus (pending_decision, unassigned, ...) + health nominal (Redis dispo)."""
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")

        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.get_json()

        kpi = data["kpi"]
        for key in (
            "pending_decision",
            "unassigned",
            "in_service",
            "delay_count",
            "critical_delay_count",
            "critical_delay_minutes",
        ):
            assert key in kpi
        assert kpi["critical_delay_minutes"] == 15
        # bootstrap_booking est PENDING → compte dans pending_decision.
        assert kpi["pending_decision"] >= 1

        # Troncature : champs de contrat toujours présents, même sans troncature réelle.
        assert data["bookings_truncated"] is False
        assert data["bookings_limit"] >= 1
        assert data["bookings_returned"] == len(data["bookings"])
        assert data["bookings_total"] >= 1

        assert data["health"]["realtime_sequence"] == "ok"
        assert data["health"]["notifications"] == "ok"

    def test_realtime_degraded_when_redis_unavailable(
        self, client, sample_user, bootstrap_company, monkeypatch
    ):
        """Redis down → snapshot_cursor=None (jamais 0) + health.realtime_sequence=degraded."""
        import ext

        monkeypatch.setattr(ext, "redis_client", None)
        headers = _company_headers(client, sample_user, bootstrap_company.id)

        response = client.get(
            "/api/v1/companies/me/dashboard/bootstrap", headers=headers
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["snapshot_cursor"] is None
        assert data["health"]["realtime_sequence"] == "degraded"

    def test_notifications_degraded_on_failure(
        self, client, sample_user, bootstrap_company, monkeypatch
    ):
        """Échec comptage notifications → unread_count=null + health.notifications=degraded
        (le reste du bootstrap doit rester utilisable, pas de 503)."""
        from models.company_notification import CompanyNotification

        class _BoomQuery:
            def filter_by(self, **kwargs):
                raise RuntimeError("db indisponible")

        monkeypatch.setattr(CompanyNotification, "query", _BoomQuery())
        headers = _company_headers(client, sample_user, bootstrap_company.id)

        response = client.get(
            "/api/v1/companies/me/dashboard/bootstrap", headers=headers
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["notifications"]["unread_count"] is None
        assert data["health"]["notifications"] == "degraded"

    def test_kpi_failure_returns_503_with_error(
        self, client, sample_user, bootstrap_company, monkeypatch
    ):
        """Échec KPI (SQL/agrégat) → 503 avec un message d'erreur, pas de payload partiel."""
        import routes.companies as companies_module

        def _boom(*_args, **_kwargs):
            raise RuntimeError("agrégat KPI indisponible")

        monkeypatch.setattr(companies_module, "_booking_stats_from_base_query", _boom)
        headers = _company_headers(client, sample_user, bootstrap_company.id)

        response = client.get(
            "/api/v1/companies/me/dashboard/bootstrap", headers=headers
        )
        assert response.status_code == 503
        data = response.get_json()
        assert "error" in data

    def test_schema_version_2_exposes_action_queue(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        """schema_version=2 (additif) : action_queue + total/troncature/curseur + summary.to_handle."""
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")

        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}&schema_version=2",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.get_json()

        assert data["schema_version"] == 2
        assert isinstance(data["action_queue"], list)
        assert "action_queue_total" in data
        assert "action_queue_truncated" in data
        assert "action_queue_next_cursor" in data
        assert data["summary"]["to_handle"] == data["action_queue_total"]
        # bootstrap_booking (PENDING) doit apparaître dans la file d'action.
        assert data["action_queue_total"] >= 1
        assert any(
            item.get("entity_id") == bootstrap_booking.id
            for item in data["action_queue"]
        )


class TestCompanyDashboardBootstrapReliability:
    """PR1 — jamais de KPI à 0 / bookings vides trompeurs en cas d'échec critique."""

    def test_kpi_failure_returns_503_with_explicit_error(
        self, client, monkeypatch, sample_user, bootstrap_company, bootstrap_booking
    ):
        import routes.companies as companies_module

        def _boom(*_args, **_kwargs):
            raise RuntimeError("DB indisponible (test)")

        monkeypatch.setattr(companies_module, "_booking_stats_from_base_query", _boom)

        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")
        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )

        assert response.status_code == 503
        data = response.get_json()
        assert "error" in data
        assert "kpi" not in data
        assert "bookings" not in data
        assert data["health"]["kpi"] == "failed"

    def test_bookings_serialization_failure_returns_503(
        self, client, monkeypatch, sample_user, bootstrap_company, bootstrap_booking
    ):
        import services.companies.booking_transfer_cache as transfer_cache_module

        def _boom(*_args, **_kwargs):
            raise RuntimeError("Sérialisation cassée (test)")

        monkeypatch.setattr(
            transfer_cache_module, "attach_transfer_cache_to_bookings", _boom
        )

        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")
        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )

        assert response.status_code == 503
        data = response.get_json()
        assert "error" in data
        assert data["health"]["bookings"] == "failed"

    def test_notifications_failure_degrades_without_failing_bootstrap(
        self, client, monkeypatch, sample_user, bootstrap_company, bootstrap_booking
    ):
        from models.company_notification import CompanyNotification

        class _RaisingQuery:
            def filter_by(self, **_kwargs):
                raise RuntimeError("Notifications indisponibles (test)")

        monkeypatch.setattr(CompanyNotification, "query", _RaisingQuery())

        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")
        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )

        # Une notification en panne ne doit jamais faire échouer tout le bootstrap.
        assert response.status_code == 200
        data = response.get_json()
        assert data["notifications"]["unread_count"] is None
        assert data["health"]["notifications"] == "degraded"
        # Le reste du payload (KPI, bookings) reste fiable.
        assert data["kpi"]["total"] >= 1

    def test_bookings_truncation_uses_limit_plus_one(
        self, client, db, monkeypatch, sample_user, bootstrap_company, bootstrap_booking
    ):
        # Deuxième réservation le même jour pour dépasser une limite artificiellement basse.
        second = Booking()
        second.customer_name = f"Client Bootstrap {uuid.uuid4().hex[:6]}"
        second.pickup_location = "Rue Gamma 3, Genève"
        second.dropoff_location = "Rue Delta 4, Genève"
        second.pickup_lat = 46.22
        second.pickup_lon = 6.12
        second.dropoff_lat = 46.23
        second.dropoff_lon = 6.16
        second.booking_type = "standard"
        second.scheduled_time = bootstrap_booking.scheduled_time
        second.amount = 12.0
        second.status = BookingStatus.PENDING
        second.company_id = bootstrap_company.id
        db.session.add(second)
        db.session.flush()

        monkeypatch.setenv("LIRIE_DASHBOARD_BOOTSTRAP_MAX_BOOKINGS", "1")

        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")
        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["bookings_limit"] == 1
        assert data["bookings_returned"] == 1
        assert data["bookings_total"] == 2
        assert data["bookings_truncated"] is True
        assert len(data["bookings"]) == 1

    def test_schema_version_2_action_queue_matches_summary_to_handle(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")
        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}&schema_version=2",
            headers=headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["schema_version"] == 2

        # v1 reste inchangé (additif uniquement).
        assert "kpi" in data and "bookings" in data

        assert "action_queue" in data
        assert isinstance(data["action_queue"], list)
        assert any(
            item["booking_id"] == bootstrap_booking.id
            and item["kind"] == "pending_decision"
            for item in data["action_queue"]
        )
        assert "action_queue_total" in data
        assert "action_queue_truncated" in data
        assert "action_queue_next_cursor" in data
        assert data["summary"]["to_handle"] == data["action_queue_total"]


class TestCompanyActionQueueExecute:
    """PR3 — exécution idempotente + concurrence optimiste de la file d'actions."""

    def _execute(self, client, headers, action_id, **body):
        return client.post(
            f"/api/v1/companies/me/action-queue/{action_id}/execute",
            headers=headers,
            json=body,
        )

    def test_accept_pending_decision_transitions_booking(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        action_id = f"pending_decision:{bootstrap_booking.id}"

        response = self._execute(
            client,
            headers,
            action_id,
            action="accept",
            expected_version=1,
            idempotency_key="accept-key-1",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "accepted"
        assert data["new_version"] == 2

    def test_same_key_same_payload_replays_result(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        action_id = f"pending_decision:{bootstrap_booking.id}"
        body = {
            "action": "accept",
            "expected_version": 1,
            "idempotency_key": "accept-key-2",
        }

        first = self._execute(client, headers, action_id, **body)
        second = self._execute(client, headers, action_id, **body)

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.get_json() == second.get_json()

    def test_same_key_different_payload_returns_conflict(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        action_id = f"pending_decision:{bootstrap_booking.id}"

        first = self._execute(
            client,
            headers,
            action_id,
            action="accept",
            expected_version=1,
            idempotency_key="accept-key-3",
        )
        assert first.status_code == 200

        second = self._execute(
            client,
            headers,
            action_id,
            action="reject",
            expected_version=1,
            idempotency_key="accept-key-3",
        )

        assert second.status_code == 409
        assert second.get_json()["error"] == "idempotency_conflict"

    def test_stale_expected_version_returns_409(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        action_id = f"pending_decision:{bootstrap_booking.id}"

        response = self._execute(
            client,
            headers,
            action_id,
            action="accept",
            expected_version=999,
            idempotency_key="accept-key-4",
        )

        assert response.status_code == 409
        assert response.get_json()["error"] == "stale_action"

    def test_missing_idempotency_key_is_rejected(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        action_id = f"pending_decision:{bootstrap_booking.id}"

        response = self._execute(
            client, headers, action_id, action="accept", expected_version=1
        )

        assert response.status_code == 400
