"""Tests routes /api/v1/payments (GET/PUT/POST)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token

from models import Booking, Client, Company, Payment, User
from models.enums import BookingStatus, PaymentStatus, UserRole


def _auth_headers(app, user: User, *, role: str | None = None) -> dict[str, str]:
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims={
                "role": role or str(getattr(user.role, "value", user.role)),
                "aud": "atmr-api",
            },
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def payments_world(db, app):
    """Client + booking + paiement pour les routes payments."""
    suffix = uuid.uuid4().hex[:8]

    company_user = User()
    company_user.username = f"payco_{suffix}"
    company_user.email = f"payco_{suffix}@test.ch"
    company_user.role = UserRole.company
    company_user.public_id = str(uuid.uuid4())
    company_user.set_password("password123", force_change=False)
    db.session.add(company_user)
    db.session.flush()

    company = Company()
    company.name = f"Pay Co {suffix}"
    company.address = "Rue Pay 1"
    company.contact_email = company_user.email
    company.user_id = company_user.id
    db.session.add(company)
    db.session.flush()

    client_user = User()
    client_user.username = f"paycl_{suffix}"
    client_user.email = f"paycl_{suffix}@test.ch"
    client_user.role = UserRole.client
    client_user.public_id = str(uuid.uuid4())
    client_user.first_name = "Pay"
    client_user.last_name = "Client"
    client_user.set_password("password123", force_change=False)
    db.session.add(client_user)
    db.session.flush()

    client = Client()
    client.user_id = client_user.id
    client.company_id = company.id
    client.contact_email = client_user.email
    client.is_active = True
    db.session.add(client)
    db.session.flush()

    other_user = User()
    other_user.username = f"payot_{suffix}"
    other_user.email = f"payot_{suffix}@test.ch"
    other_user.role = UserRole.client
    other_user.public_id = str(uuid.uuid4())
    other_user.set_password("password123", force_change=False)
    db.session.add(other_user)
    db.session.flush()

    other_client = Client()
    other_client.user_id = other_user.id
    other_client.company_id = company.id
    other_client.contact_email = other_user.email
    other_client.is_active = True
    db.session.add(other_client)
    db.session.flush()

    admin = User()
    admin.username = f"payad_{suffix}"
    admin.email = f"payad_{suffix}@test.ch"
    admin.role = UserRole.admin
    admin.public_id = str(uuid.uuid4())
    admin.set_password("password123", force_change=False)
    db.session.add(admin)
    db.session.flush()

    booking = Booking()
    booking.user_id = client_user.id
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = "Pay Client"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
    booking.status = BookingStatus.COMPLETED
    booking.amount = Decimal("45.00")
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()

    payment = Payment(
        amount=45.0,
        method="credit_card",
        status=PaymentStatus.PENDING,
        user_id=client_user.id,
        client_id=client.id,
        booking_id=booking.id,
    )
    db.session.add(payment)
    db.session.commit()

    return {
        "company": company,
        "company_user": company_user,
        "client_user": client_user,
        "client": client,
        "other_user": other_user,
        "admin": admin,
        "booking": booking,
        "payment": payment,
    }


class TestPaymentsRoutes:
    def test_get_my_payments_success(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["client_user"], role="client")
        resp = client.get("/api/v1/payments/me", headers=headers)
        assert resp.status_code == 200
        body = resp.get_json()
        data = body.get("data") if isinstance(body, dict) else body
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_my_payments_401_without_jwt(self, client, payments_world):
        resp = client.get("/api/v1/payments/me")
        assert resp.status_code == 401

    def test_get_my_payments_403_non_client(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["company_user"], role="company")
        resp = client.get("/api/v1/payments/me", headers=headers)
        assert resp.status_code in (403, 401)

    def test_get_my_payments_empty_404(self, client, app, payments_world, db):
        # Autre client sans paiement
        headers = _auth_headers(app, payments_world["other_user"], role="client")
        resp = client.get("/api/v1/payments/me", headers=headers)
        assert resp.status_code == 404

    def test_get_payment_owner_ok(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["client_user"], role="client")
        pid = payments_world["payment"].id
        resp = client.get(f"/api/v1/payments/{pid}", headers=headers)
        assert resp.status_code == 200

    def test_get_payment_401_without_jwt(self, client, payments_world):
        pid = payments_world["payment"].id
        resp = client.get(f"/api/v1/payments/{pid}")
        assert resp.status_code == 401

    def test_get_payment_not_found_404(self, client, app, payments_world):
        """Contrat produit : paiement inexistant → 404 (pas 400 validation)."""
        headers = _auth_headers(app, payments_world["admin"], role="admin")
        resp = client.get("/api/v1/payments/999999", headers=headers)
        assert resp.status_code == 404

    def test_get_payment_forbidden_other_client(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["other_user"], role="client")
        pid = payments_world["payment"].id
        resp = client.get(f"/api/v1/payments/{pid}", headers=headers)
        assert resp.status_code in (403, 401)

    def test_get_payment_admin_ok(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["admin"], role="admin")
        pid = payments_world["payment"].id
        resp = client.get(f"/api/v1/payments/{pid}", headers=headers)
        assert resp.status_code == 200

    def test_put_payment_status_admin_ok(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["admin"], role="admin")
        pid = payments_world["payment"].id
        resp = client.put(
            f"/api/v1/payments/{pid}",
            json={"status": "completed"},
            headers=headers,
        )
        assert resp.status_code == 200

    def test_put_payment_status_non_admin_forbidden(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["client_user"], role="client")
        pid = payments_world["payment"].id
        resp = client.put(
            f"/api/v1/payments/{pid}",
            json={"status": "completed"},
            headers=headers,
        )
        assert resp.status_code in (403, 401)

    def test_put_payment_missing_404(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["admin"], role="admin")
        resp = client.put(
            "/api/v1/payments/999999",
            json={"status": "completed"},
            headers=headers,
        )
        assert resp.status_code == 404

    def test_put_payment_status_invalid_schema_400(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["admin"], role="admin")
        pid = payments_world["payment"].id
        resp = client.put(
            f"/api/v1/payments/{pid}",
            json={"status": "not-a-status"},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_post_payment_for_booking_success(self, client, app, payments_world, db):
        # Nouveau booking sans paiement pour éviter conflits métier
        world = payments_world
        booking = Booking()
        booking.user_id = world["client_user"].id
        booking.company_id = world["company"].id
        booking.client_id = world["client"].id
        booking.customer_name = "Pay Client"
        booking.pickup_location = "X"
        booking.dropoff_location = "Y"
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=3)
        booking.status = BookingStatus.COMPLETED
        booking.amount = Decimal("20.00")
        booking.billed_to_type = "patient"
        db.session.add(booking)
        db.session.commit()

        headers = _auth_headers(app, world["client_user"], role="client")
        resp = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={"amount": 20.0, "method": "credit_card"},
            headers=headers,
        )
        assert resp.status_code in (200, 201)

    def test_post_payment_booking_not_owned_404(self, client, app, payments_world):
        headers = _auth_headers(app, payments_world["other_user"], role="client")
        booking_id = payments_world["booking"].id
        resp = client.post(
            f"/api/v1/payments/booking/{booking_id}",
            json={"amount": 10.0, "method": "credit_card"},
            headers=headers,
        )
        assert resp.status_code == 404

    def test_post_payment_restx_empty_body_400(self, client, app, payments_world):
        """Contrat HTTP RESTX (validate=True) : {} → 400 avant le handler métier."""
        headers = _auth_headers(app, payments_world["client_user"], role="client")
        booking_id = payments_world["booking"].id
        resp = client.post(
            f"/api/v1/payments/booking/{booking_id}",
            json={},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_post_payment_marshmallow_handler_400(self, client, app, payments_world):
        """Payload accepté par RESTX mais rejeté par PaymentCreateSchema (handler)."""
        headers = _auth_headers(app, payments_world["client_user"], role="client")
        booking_id = payments_world["booking"].id
        resp = client.post(
            f"/api/v1/payments/booking/{booking_id}",
            json={
                "amount": 10.0,
                "method": "credit_card",
                "booking_id": 0,  # Range(min=1) Marshmallow
            },
            headers=headers,
        )
        assert resp.status_code == 400
