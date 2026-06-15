from flask import Flask

from middleware.silent_json_request import SilentJSONRequest, redact_json_body_preview


def test_silent_json_request_invalid_body_does_not_raise():
    app = Flask(__name__)
    app.request_class = SilentJSONRequest

    with app.test_request_context(
        "/x", method="POST", data="not-json", content_type="application/json"
    ):
        from flask import request

        assert request.get_json() is None


def test_redact_json_body_preview_masks_password():
    preview = redact_json_body_preview('{"username":"a","password":"secret"}')
    assert "secret" not in preview
    assert "***" in preview
