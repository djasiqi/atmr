"""Le code OAuth ne doit pas apparaître dans une ligne de journal."""

from __future__ import annotations

import logging

from shared.oauth_log_redaction import OAuthLogFilter, redact_oauth_query


def test_ligne_dacces_masque_le_code_et_conserve_le_chemin():
    raw = (
        "GET /api/v1/auth/external/callback?code=4/0Acode-secret"
        "&state=state-public HTTP/1.1"
    )
    masked = redact_oauth_query(raw)
    assert "4/0Acode-secret" not in masked
    assert "code=[filtre]" in masked
    assert "/api/v1/auth/external/callback" in masked
    assert "state=state-public" in masked


def test_filtre_gunicorn_conserve_le_dictionnaire_d_arguments():
    record = logging.LogRecord(
        name="gunicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="placeholder",
        args=(),
        exc_info=None,
    )
    record.msg = "%(r)s"
    record.args = {
        "r": "GET /callback?code=secret-oauth&id_token=jeton-secret HTTP/1.1",
    }
    assert OAuthLogFilter().filter(record) is True
    assert isinstance(record.args, dict)
    rendered = record.getMessage()
    assert "secret-oauth" not in rendered
    assert "jeton-secret" not in rendered
    assert "code=[filtre]" in rendered
    assert "id_token=[filtre]" in rendered
