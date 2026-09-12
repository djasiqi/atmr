"""Aucune exception technique brute ne doit sortir en HTTP."""

from __future__ import annotations

from marshmallow import ValidationError
from sqlalchemy.exc import IntegrityError
from werkzeug.exceptions import BadRequest, Forbidden

from routes.api_error_utils import create_error_response, create_internal_error
from routes.companies import _safe_company_value_error
from routes.db_error_utils import format_integrity_error
from shared.error_handlers import APIErrorHandler

SENTINEL = "SUPER_SECRET_INTERNAL_SENTINEL"


def _assert_no_sentinel(body: dict, *extra: str) -> None:
    dumped = str(body)
    assert SENTINEL not in dumped
    for token in extra:
        assert token not in dumped


def test_handle_exception_generic_500_blocks_sentinel():
    body, status = APIErrorHandler.handle_exception(RuntimeError(SENTINEL))
    assert status == 500
    _assert_no_sentinel(body, "RuntimeError")
    assert body.get("error_code") == "internal_error"


def test_handle_exception_value_error_is_generic_validation():
    body, status = APIErrorHandler.handle_exception(ValueError(SENTINEL))
    assert status == 400
    _assert_no_sentinel(body)
    assert "Données invalides" in str(body)


def test_handle_exception_key_error_does_not_echo_key():
    body, status = APIErrorHandler.handle_exception(KeyError(SENTINEL))
    assert status == 400
    _assert_no_sentinel(body)
    assert "Champ manquant" in str(body)


def test_handle_exception_marshmallow_is_generic():
    body, status = APIErrorHandler.handle_exception(
        ValidationError({"email": [SENTINEL]})
    )
    assert status == 400
    _assert_no_sentinel(body)


def test_handle_exception_http_exception_uses_controlled_message():
    body, status = APIErrorHandler.handle_exception(BadRequest(SENTINEL))
    assert status == 400
    _assert_no_sentinel(body)
    assert body["error"] == "bad_request"
    assert body["message"] == "Requête invalide"


def test_handle_exception_password_change_required_is_controlled():
    class _Resp:
        status_code = 403

        def get_json(self, silent=True):
            return {
                "error": "password_change_required",
                "message": SENTINEL,
                "redirect_to": SENTINEL,
            }

    forbidden = Forbidden()
    forbidden.response = _Resp()
    body, status = APIErrorHandler.handle_exception(forbidden)
    assert status == 403
    assert body == {
        "error": "password_change_required",
        "message": "Vous devez modifier votre mot de passe avant de continuer.",
        "redirect_to": "/force-reset-password",
    }
    _assert_no_sentinel(body)


def test_handle_exception_file_not_found_is_internal():
    body, status = APIErrorHandler.handle_exception(
        FileNotFoundError(f"/secret/path/{SENTINEL}.pem")
    )
    assert status == 500
    _assert_no_sentinel(body, "/secret/path")


def test_integrity_error_has_no_sql():
    class _Orig:
        pgcode = "23505"

    err = IntegrityError(
        f"SELECT * FROM users WHERE email={SENTINEL}",
        params={"email": SENTINEL},
        orig=_Orig(),
    )
    body, status = format_integrity_error(err)
    assert status == 400
    _assert_no_sentinel(body, "users_email_key", "SELECT")
    assert body["error"] == "unique_constraint_violation"


def test_create_internal_error_blocks_sentinel_and_type():
    body, status = create_internal_error(exception=RuntimeError(SENTINEL))
    assert status == 500
    _assert_no_sentinel(body, "RuntimeError")


def test_create_error_response_blocks_sentinel_in_logs_payload():
    body, status = create_error_response(
        "Une erreur interne s'est produite",
        500,
        exception=RuntimeError(SENTINEL),
    )
    assert status == 500
    _assert_no_sentinel(body)


def test_safe_company_value_error_maps_controlled_messages():
    assert _safe_company_value_error(ValueError("IBAN invalide (checksum).")) == (
        "IBAN invalide."
    )
    assert _safe_company_value_error(ValueError(SENTINEL)) == "Données invalides"


def test_handle_validation_error_allows_controlled_message():
    body, status = APIErrorHandler.handle_validation_error(
        "Departure time must be in the future."
    )
    assert status == 400
    assert "Departure time must be in the future." in str(body)
