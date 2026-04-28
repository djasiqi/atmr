from __future__ import annotations

import logging
import os

from services.saferpay.config import (
    saferpay_api_url_looks_like_test_host,
    saferpay_configured,
    warn_saferpay_test_api_url_in_production,
)
import services.saferpay.config as saferpay_config_module


def test_saferpay_api_url_looks_like_test_host():
    assert saferpay_api_url_looks_like_test_host("https://test.saferpay.com/api")
    assert not saferpay_api_url_looks_like_test_host("https://www.saferpay.com/api")


def test_warn_saferpay_test_url_skips_non_production():
    log = logging.getLogger("test_saferpay_cfg")
    warn_saferpay_test_api_url_in_production(log, config_name="development")


def test_warn_saferpay_test_url_respects_allow_flag(monkeypatch):
    monkeypatch.setenv("SAFERPAY_ALLOW_TEST_API_IN_PRODUCTION", "1")
    log = logging.getLogger("test_saferpay_cfg2")
    warn_saferpay_test_api_url_in_production(log, config_name="production")


def test_warn_saferpay_test_url_logs_in_production(monkeypatch, caplog):
    monkeypatch.delenv("SAFERPAY_ALLOW_TEST_API_IN_PRODUCTION", raising=False)
    monkeypatch.setenv("SAFERPAY_API_BASE_URL", "https://test.saferpay.com/api")
    with caplog.at_level(logging.ERROR):
        warn_saferpay_test_api_url_in_production(
            logging.getLogger("saferpay_cfg_warn"), config_name="production"
        )
    assert any("SAFERPAY_API_BASE_URL" in r.message for r in caplog.records)


def test_saferpay_configured_merges_missing_keys_from_root_dotenv(
    monkeypatch, tmp_path
):
    """Si l'OS a une cle SAFERPAY vide, on relit le .env racine (dotenv_values)."""
    monkeypatch.setattr(
        saferpay_config_module,
        "_repo_root_env_path",
        lambda: tmp_path / ".env",
    )
    monkeypatch.setattr(saferpay_config_module, "_saferpay_repo_merge_state", {"done": False})
    (tmp_path / ".env").write_text(
        "SAFERPAY_CUSTOMER_ID=cid\n"
        "SAFERPAY_TERMINAL_ID=tid\n"
        "SAFERPAY_API_USERNAME=user\n"
        "SAFERPAY_API_PASSWORD=secret\n",
        encoding="utf-8",
    )
    for key in saferpay_config_module._SAFERPAY_REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SAFERPAY_CUSTOMER_ID", "")

    assert saferpay_configured()
    assert os.environ["SAFERPAY_CUSTOMER_ID"] == "cid"
    assert os.environ["SAFERPAY_TERMINAL_ID"] == "tid"
