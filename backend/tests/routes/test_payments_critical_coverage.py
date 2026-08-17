"""Couverture critique ``routes/payments.py`` (seuil 80 %)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from marshmallow import ValidationError

from routes import payments as payments_mod
from tests.routes.test_payments import _auth_headers
from tests.routes.test_payments import payments_world as _payments_world_fixture


@pytest.fixture
def payments_world(_payments_world_fixture):  # noqa: F811
    return _payments_world_fixture


def _fake_uc(result):
    class _UC:
        def __init__(self, **_kwargs):
            pass

        def execute(self, _input):
            return result

    return _UC


def _boom_uc(exc: Exception | None = None):
    error = exc or RuntimeError("db down")

    class _UC:
        def __init__(self, **_kwargs):
            pass

        def execute(self, _input):
            raise error

    return _UC


def test_get_me_user_absent_et_usecase_en_echec(
    client, app, payments_world, monkeypatch
):
    headers = _auth_headers(app, payments_world["client_user"], role="client")
    monkeypatch.setattr(payments_mod, "get_current_user_via_use_case", lambda: None)
    resp = client.get("/api/v1/payments/me", headers=headers)
    assert resp.status_code in (401, 403)

    monkeypatch.setattr(
        payments_mod,
        "get_current_user_via_use_case",
        lambda: payments_world["client_user"],
    )
    monkeypatch.setattr(
        payments_mod,
        "ListPaymentsUseCase",
        _fake_uc(
            SimpleNamespace(success=False, error={"message": "liste indisponible"})
        ),
    )
    resp = client.get("/api/v1/payments/me", headers=headers)
    assert resp.status_code == 400

    monkeypatch.setattr(
        payments_mod,
        "ListPaymentsUseCase",
        _fake_uc(SimpleNamespace(success=False, error=None)),
    )
    resp = client.get("/api/v1/payments/me", headers=headers)
    assert resp.status_code == 400


def test_get_me_exception(client, app, payments_world, monkeypatch):
    monkeypatch.setattr(payments_mod.sentry_sdk, "capture_exception", lambda _e: None)
    monkeypatch.setattr(payments_mod, "ListPaymentsUseCase", _boom_uc())
    headers = _auth_headers(app, payments_world["client_user"], role="client")
    resp = client.get("/api/v1/payments/me", headers=headers)
    assert resp.status_code >= 400


def test_get_payment_id_invalide_et_variantes_introuvable(
    client, app, payments_world, monkeypatch
):
    headers = _auth_headers(app, payments_world["admin"], role="admin")
    resp = client.get("/api/v1/payments/0", headers=headers)
    assert resp.status_code == 400

    monkeypatch.setattr(
        payments_mod,
        "GetPaymentUseCase",
        _fake_uc(
            SimpleNamespace(
                found=False,
                error={"error": "format"},
                status_code=400,
                payment=None,
            )
        ),
    )
    resp = client.get("/api/v1/payments/12", headers=headers)
    assert resp.status_code == 400

    monkeypatch.setattr(
        payments_mod,
        "GetPaymentUseCase",
        _fake_uc(
            SimpleNamespace(
                found=True,
                error=None,
                status_code=None,
                payment=None,
            )
        ),
    )
    resp = client.get("/api/v1/payments/12", headers=headers)
    assert resp.status_code == 404


def test_get_payment_user_absent_et_exception(client, app, payments_world, monkeypatch):
    headers = _auth_headers(app, payments_world["client_user"], role="client")
    payment = SimpleNamespace(
        client_id=payments_world["client"].id, to_dict=lambda: {"id": 1}
    )
    monkeypatch.setattr(
        payments_mod,
        "GetPaymentUseCase",
        _fake_uc(SimpleNamespace(found=True, payment=payment, error=None)),
    )
    monkeypatch.setattr(payments_mod, "get_current_user_via_use_case", lambda: None)
    resp = client.get("/api/v1/payments/12", headers=headers)
    assert resp.status_code in (401, 403)

    monkeypatch.setattr(payments_mod.sentry_sdk, "capture_exception", lambda _e: None)
    monkeypatch.setattr(payments_mod, "GetPaymentUseCase", _boom_uc())
    resp = client.get("/api/v1/payments/12", headers=headers)
    assert resp.status_code >= 400


def test_put_validation_marshmallow_et_id_invalide(
    client, app, payments_world, monkeypatch
):
    headers = _auth_headers(app, payments_world["admin"], role="admin")

    def _raise_validation(_schema, _data):
        raise ValidationError({"status": ["invalide"]})

    monkeypatch.setattr("schemas.validation_utils.validate_request", _raise_validation)
    resp = client.put(
        f"/api/v1/payments/{payments_world['payment'].id}",
        json={"status": "pending"},
        headers=headers,
    )
    assert resp.status_code == 400

    monkeypatch.undo()
    resp = client.put(
        "/api/v1/payments/0",
        json={"status": "failed"},
        headers=headers,
    )
    assert resp.status_code == 400


def test_put_update_echec_500_validation_et_introuvable_apres(
    client, app, payments_world, monkeypatch
):
    headers = _auth_headers(app, payments_world["admin"], role="admin")
    pid = payments_world["payment"].id

    monkeypatch.setattr(
        payments_mod,
        "UpdatePaymentStatusUseCase",
        _fake_uc(
            SimpleNamespace(success=False, error={"message": "crash"}, status_code=500)
        ),
    )
    resp = client.put(
        f"/api/v1/payments/{pid}", json={"status": "completed"}, headers=headers
    )
    assert resp.status_code >= 400

    monkeypatch.setattr(
        payments_mod,
        "UpdatePaymentStatusUseCase",
        _fake_uc(SimpleNamespace(success=False, error={"x": "y"}, status_code=400)),
    )
    resp = client.put(
        f"/api/v1/payments/{pid}", json={"status": "failed"}, headers=headers
    )
    assert resp.status_code == 400

    monkeypatch.setattr(
        payments_mod,
        "UpdatePaymentStatusUseCase",
        _fake_uc(SimpleNamespace(success=True, error=None, status_code=None)),
    )
    monkeypatch.setattr(payments_mod.payment_repo, "find_by_id", lambda _i: None)
    resp = client.put(
        f"/api/v1/payments/{pid}", json={"status": "pending"}, headers=headers
    )
    assert resp.status_code == 404


def test_put_exception(client, app, payments_world, monkeypatch):
    monkeypatch.setattr(payments_mod.sentry_sdk, "capture_exception", lambda _e: None)
    monkeypatch.setattr(payments_mod, "UpdatePaymentStatusUseCase", _boom_uc())
    headers = _auth_headers(app, payments_world["admin"], role="admin")
    resp = client.put(
        f"/api/v1/payments/{payments_world['payment'].id}",
        json={"status": "completed"},
        headers=headers,
    )
    assert resp.status_code >= 400


def test_post_user_et_client_absents(client, app, payments_world, monkeypatch):
    headers = _auth_headers(app, payments_world["client_user"], role="client")
    booking_id = payments_world["booking"].id
    monkeypatch.setattr(payments_mod, "get_current_user_via_use_case", lambda: None)
    resp = client.post(
        f"/api/v1/payments/booking/{booking_id}",
        json={"amount": 10.0, "method": "credit_card"},
        headers=headers,
    )
    assert resp.status_code in (401, 403)


def test_post_client_absent(client, app, payments_world):
    co_headers = _auth_headers(app, payments_world["company_user"], role="company")
    resp = client.post(
        f"/api/v1/payments/booking/{payments_world['booking'].id}",
        json={"amount": 10.0, "method": "credit_card"},
        headers=co_headers,
    )
    assert resp.status_code in (401, 403)


def test_post_booking_id_invalide_create_echec_et_sans_paiement(
    client, app, payments_world, monkeypatch
):
    headers = _auth_headers(app, payments_world["client_user"], role="client")
    monkeypatch.setattr(
        payments_mod.booking_repo,
        "find_model_by_id_and_client",
        lambda *_a, **_k: SimpleNamespace(id=0),
    )
    resp = client.post(
        "/api/v1/payments/booking/0",
        json={"amount": 10.0, "method": "credit_card"},
        headers=headers,
    )
    assert resp.status_code == 400

    monkeypatch.setattr(
        payments_mod.booking_repo,
        "find_model_by_id_and_client",
        lambda *_a, **_k: SimpleNamespace(id=payments_world["booking"].id),
    )
    monkeypatch.setattr(
        payments_mod,
        "CreatePaymentUseCase",
        _fake_uc(
            SimpleNamespace(success=False, error={"message": "refusé"}, payment=None)
        ),
    )
    resp = client.post(
        f"/api/v1/payments/booking/{payments_world['booking'].id}",
        json={"amount": 10.0, "method": "credit_card"},
        headers=headers,
    )
    assert resp.status_code == 400

    monkeypatch.setattr(
        payments_mod,
        "CreatePaymentUseCase",
        _fake_uc(SimpleNamespace(success=False, error=None, payment=None)),
    )
    resp = client.post(
        f"/api/v1/payments/booking/{payments_world['booking'].id}",
        json={"amount": 10.0, "method": "credit_card"},
        headers=headers,
    )
    assert resp.status_code == 400

    monkeypatch.setattr(
        payments_mod,
        "CreatePaymentUseCase",
        _fake_uc(SimpleNamespace(success=True, error=None, payment=None)),
    )
    resp = client.post(
        f"/api/v1/payments/booking/{payments_world['booking'].id}",
        json={"amount": 10.0, "method": "credit_card"},
        headers=headers,
    )
    assert resp.status_code >= 400


def test_post_exception(client, app, payments_world, monkeypatch):
    monkeypatch.setattr(payments_mod.sentry_sdk, "capture_exception", lambda _e: None)
    monkeypatch.setattr(payments_mod, "CreatePaymentUseCase", _boom_uc())
    headers = _auth_headers(app, payments_world["client_user"], role="client")
    resp = client.post(
        f"/api/v1/payments/booking/{payments_world['booking'].id}",
        json={"amount": 10.0, "method": "credit_card"},
        headers=headers,
    )
    assert resp.status_code >= 400
