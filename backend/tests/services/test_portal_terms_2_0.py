"""7C — Conditions PORTAL 2.0 (PREPARED) vs CURRENT 1.0 + gate d'activation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.client_terms_acceptance import (
    DOCUMENT_TERMS_OF_SERVICE,
    DOCUMENT_TRANSPORT_TERMS,
    ClientTermsAcceptance,
)
from services.legal.portal_terms_catalog import (
    TERMS_OF_SERVICE_V1_SHA256,
    TERMS_OF_SERVICE_V2_SHA256,
    TRANSPORT_TERMS_V1_SHA256,
    TRANSPORT_TERMS_V2_SHA256,
    CatalogIntegrityError,
    PortalTermsActivationError,
    assert_activation_coordination,
    canonical_sha256,
    current_portal_terms,
    effective_portal_terms_version,
    prepared_portal_terms_v2,
)
from services.legal.portal_terms_status import (
    STATUS_CURRENT,
    STATUS_REACCEPTANCE_REQUIRED,
    accept_current_required_portal_terms,
    resolve_portal_terms_status,
)
from services.legal.record_terms_acceptance import (
    ensure_document_version,
    record_portal_terms_acceptance,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = BACKEND_ROOT / "legal" / "canonical" / "fr"


def test_v1_hashes_unchanged(monkeypatch):
    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "false")
    monkeypatch.setenv("PORTAL_CONDITIONAL_ORDER_ENABLED", "false")
    tos = (CANONICAL / "terms_of_service_v1.0.txt").read_text(encoding="utf-8")
    tr = (CANONICAL / "transport_terms_v1.0.txt").read_text(encoding="utf-8")
    assert canonical_sha256(tos) == TERMS_OF_SERVICE_V1_SHA256
    assert canonical_sha256(tr) == TRANSPORT_TERMS_V1_SHA256
    current = current_portal_terms()
    assert all(s.terms_version == "1.0" for s in current)
    assert current[0].terms_hash == TERMS_OF_SERVICE_V1_SHA256
    assert current[1].terms_hash == TRANSPORT_TERMS_V1_SHA256


def test_v2_prepared_hashes_stable_and_distinct():
    prepared = prepared_portal_terms_v2()
    assert all(s.terms_version == "2.0" for s in prepared)
    assert all(s.requires_reacceptance is True for s in prepared)
    assert all(s.status == "prepared" for s in prepared)
    assert prepared[0].terms_hash == TERMS_OF_SERVICE_V2_SHA256
    assert prepared[1].terms_hash == TRANSPORT_TERMS_V2_SHA256
    assert TERMS_OF_SERVICE_V2_SHA256 != TERMS_OF_SERVICE_V1_SHA256
    assert TRANSPORT_TERMS_V2_SHA256 != TRANSPORT_TERMS_V1_SHA256
    # Contenu métier 2.0
    body_tr = prepared[1].canonical_body
    assert "CLIENT_TRANSPORT_CONFIRMED" in body_tr
    assert "Confirmer le transport à CHF" in body_tr
    assert "n'est pas encore le contrat de transport" in body_tr or (
        "ne constitue pas encore le contrat" in body_tr
    )
    assert "LIRIE ne facture pas" in body_tr
    assert "créancier" in body_tr.lower()


def test_v2_tamper_without_version_bump_rejected(tmp_path, monkeypatch):
    # Recharge via empreinte figée : un corps altéré lève CatalogIntegrityError.
    from services.legal import portal_terms_catalog as cat

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
    with pytest.raises(CatalogIntegrityError):
        cat._load_frozen(  # noqa: SLF001 — test d'intégrité
            "terms_of_service_v2.0.txt",
            "0" * 64,
        )


def test_adding_v2_does_not_make_it_current(monkeypatch):
    monkeypatch.delenv("PORTAL_TERMS_EFFECTIVE_VERSION", raising=False)
    assert effective_portal_terms_version() == "1.0"
    prepared = prepared_portal_terms_v2()
    assert prepared[0].terms_version == "2.0"
    current = current_portal_terms()
    assert current[0].terms_version == "1.0"
    assert current[1].terms_version == "1.0"


def test_old_client_remains_current_while_v2_prepared(db, monkeypatch, app):
    from tests.routes.test_auth_sms_02_portal_contract import _make_portal_user

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "false")
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "1.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    user, client = _make_portal_user(db)
    record_portal_terms_acceptance(user, client)  # 1.0 courant
    for spec in prepared_portal_terms_v2():
        ensure_document_version(spec)
    db.session.flush()
    resolved = resolve_portal_terms_status(user)
    assert resolved.status == STATUS_CURRENT
    assert all(d.current_version == "1.0" for d in resolved.documents)


def test_activation_v2_requires_reacceptance(db, monkeypatch, app):
    from tests.routes.test_auth_sms_02_portal_contract import _make_portal_user

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "false")
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "1.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    user, client = _make_portal_user(db)
    record_portal_terms_acceptance(user, client)
    assert resolve_portal_terms_status(user).status == STATUS_CURRENT

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "2.0")
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "true")
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "2.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True

    try:
        stale = resolve_portal_terms_status(user)
        assert stale.status == STATUS_REACCEPTANCE_REQUIRED
        assert {d.document_type for d in stale.documents if d.acceptance_required} == {
            DOCUMENT_TERMS_OF_SERVICE,
            DOCUMENT_TRANSPORT_TERMS,
        }

        only_tos = [
            s
            for s in prepared_portal_terms_v2()
            if s.document_type == DOCUMENT_TERMS_OF_SERVICE
        ]
        record_portal_terms_acceptance(user, client, documents=only_tos)
        still = resolve_portal_terms_status(user)
        assert still.status == STATUS_REACCEPTANCE_REQUIRED

        created, wrote = accept_current_required_portal_terms(user, client)
        assert wrote is True
        assert len(created) == 1
        assert resolve_portal_terms_status(user).status == STATUS_CURRENT

        rows = ClientTermsAcceptance.query.filter_by(user_id=user.id).all()
        versions = {(r.document_type, r.terms_version) for r in rows}
        assert (DOCUMENT_TERMS_OF_SERVICE, "1.0") in versions
        assert (DOCUMENT_TRANSPORT_TERMS, "1.0") in versions
        assert (DOCUMENT_TERMS_OF_SERVICE, "2.0") in versions
        assert (DOCUMENT_TRANSPORT_TERMS, "2.0") in versions
    finally:
        app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "1.0"
        app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
        monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
        monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "false")


def test_activation_coordination_gate(monkeypatch, app):
    # effective_portal_terms_version() privilégie app.config sous contexte Flask.
    app.config["PORTAL_CONDITIONAL_ORDER_ENABLED"] = False
    monkeypatch.setenv("PORTAL_CONDITIONAL_ORDER_ENABLED", "false")

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "2.0")
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "2.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "false")
    with pytest.raises(PortalTermsActivationError):
        assert_activation_coordination()

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "1.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "true")
    with pytest.raises(PortalTermsActivationError):
        assert_activation_coordination()

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "2.0")
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "2.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "true")
    assert_activation_coordination()  # OK

    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
    app.config["PORTAL_TERMS_EFFECTIVE_VERSION"] = "1.0"
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    monkeypatch.setenv("PORTAL_DOUBLE_VALIDATION_ENABLED", "false")
    assert_activation_coordination()  # état prod actuel OK


def test_dv_off_and_v1_current_legacy(monkeypatch, app):
    monkeypatch.setenv("PORTAL_TERMS_EFFECTIVE_VERSION", "1.0")
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    specs = current_portal_terms()
    assert specs[0].terms_version == "1.0"
    assert app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] is False
