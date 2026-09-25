"""7D — Gate d'activation coordonnée (anti-hybride au boot + 4 combinaisons)."""

from __future__ import annotations

import pytest

from services.legal.portal_terms_catalog import (
    PortalTermsActivationError,
    assert_activation_coordination,
    current_portal_terms,
    enforce_portal_activation_coordination_at_startup,
    prepared_portal_terms_v2,
)


def test_anti_hybrid_1_0_off_pass():
    assert_activation_coordination(
        effective_version="1.0", double_validation_enabled=False
    )


def test_anti_hybrid_1_0_on_refuse():
    with pytest.raises(PortalTermsActivationError):
        assert_activation_coordination(
            effective_version="1.0", double_validation_enabled=True
        )


def test_anti_hybrid_2_0_off_refuse():
    with pytest.raises(PortalTermsActivationError):
        assert_activation_coordination(
            effective_version="2.0", double_validation_enabled=False
        )


def test_anti_hybrid_2_0_on_pass():
    assert_activation_coordination(
        effective_version="2.0", double_validation_enabled=True
    )


def test_startup_enforce_refuses_hybrid(app):
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "2.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    with pytest.raises(PortalTermsActivationError):
        enforce_portal_activation_coordination_at_startup(app)

    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "1.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
    with pytest.raises(PortalTermsActivationError):
        enforce_portal_activation_coordination_at_startup(app)

    # Restaurer état prod baseline pour les autres fixtures
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "1.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    enforce_portal_activation_coordination_at_startup(app)


def test_isolated_env_2_0_on_makes_current(monkeypatch, app):
    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "2.0")
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "true")
    prev_v = app.config.get("PORTAL_TERMS_EFFECTIVE_VERSION")
    prev_dv = app.config.get("PORTAL_DOUBLE_VALIDATION_ENABLED")
    try:
        app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "2.0"
        app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
        enforce_portal_activation_coordination_at_startup(app)
        specs = current_portal_terms()
        assert all(s.terms_version == "2.0" for s in specs)
        prepared = prepared_portal_terms_v2()
        assert specs[0].terms_hash == prepared[0].terms_hash
        assert specs[1].terms_hash == prepared[1].terms_hash
    finally:
        app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = prev_v or "1.0"
        app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = bool(prev_dv)
        monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
        monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "false")
