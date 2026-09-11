"""Réponses d'erreur API : aucune fuite de traceback vers le client."""

from routes.api_error_utils import create_error_response, create_internal_error


def test_create_error_response_never_includes_traceback():
    body, status = create_error_response(
        "Une erreur interne s'est produite",
        500,
        exception=RuntimeError("secret-stack-trace"),
    )
    assert status == 500
    assert "debug" not in body
    assert "traceback" not in body
    dumped = str(body)
    assert "secret-stack-trace" not in dumped
    assert "RuntimeError" not in dumped


def test_create_internal_error_never_includes_traceback():
    body, status = create_internal_error(
        exception=ValueError("internal-detail")
    )
    assert status == 500
    assert "debug" not in body
    assert "internal-detail" not in str(body)
