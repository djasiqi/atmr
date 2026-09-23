"""Registre append-only des acceptations CGU / CGV du client privé."""

from __future__ import annotations

import ast
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from models.client import Client
from models.client_terms_acceptance import (
    DOCUMENT_TRANSPORT_TERMS,
    VERIFICATION_OTP_SMS,
    ClientTermsAcceptance,
)
from models.enums import ClientType, UserRole
from models.user import User
from services.legal.portal_terms_catalog import (
    TRANSPORT_TERMS_V1_SHA256,
    CatalogIntegrityError,
    PublishedTerms,
    canonical_sha256,
    current_portal_terms,
)
from services.legal.record_terms_acceptance import (
    ClientSuppliedTermsError,
    list_portal_terms_acceptances,
    record_portal_terms_acceptance,
    reject_client_supplied_terms,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _portal_user(db) -> tuple[User, Client]:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"terms_{suffix}"
    user.email = f"terms-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791234567"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    client.contact_phone = user.phone
    db.session.add(client)
    db.session.flush()
    return user, client


def test_accepting_version_1_creates_rows(db) -> None:
    user, client = _portal_user(db)
    rows = record_portal_terms_acceptance(user, client)
    assert len(rows) == 2
    assert {row.terms_version for row in rows} == {"1.0"}
    assert {row.terms_hash for row in rows} == {
        current_portal_terms()[0].terms_hash,
        current_portal_terms()[1].terms_hash,
    }
    assert rows[0].verification_method == VERIFICATION_OTP_SMS
    assert rows[0].email_snapshot == user.email
    assert rows[0].phone_snapshot == user.phone


def test_second_version_appends_and_keeps_the_first(db) -> None:
    user, client = _portal_user(db)
    record_portal_terms_acceptance(user, client)
    body = (
        "LIRIE — Conditions générales de réservation et de transport (client privé)\n"
        "terms_version: 1.1\n"
        "La version 1.1 remplace l'opposabilité future, pas l'historique.\n"
    )
    spec = PublishedTerms(
        document_type=DOCUMENT_TRANSPORT_TERMS,
        terms_version="1.1",
        terms_hash=canonical_sha256(body),
        canonical_body=body,
    )
    record_portal_terms_acceptance(user, client, documents=[spec])

    stored = list_portal_terms_acceptances(user.id)
    transport = [row for row in stored if row.document_type == DOCUMENT_TRANSPORT_TERMS]
    assert [row.terms_version for row in transport] == ["1.0", "1.1"]
    assert transport[0].terms_hash == TRANSPORT_TERMS_V1_SHA256
    assert transport[0].terms_hash != transport[1].terms_hash


def test_client_cannot_choose_version_or_hash() -> None:
    with pytest.raises(ClientSuppliedTermsError):
        reject_client_supplied_terms({"terms_version": "9.9", "terms_hash": "ab" * 32})


def test_catalog_refuses_a_silently_changed_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.legal.portal_terms_catalog.TERMS_OF_SERVICE_V1_SHA256",
        "0" * 64,
    )
    with pytest.raises(CatalogIntegrityError):
        current_portal_terms()


def test_later_profile_changes_do_not_rewrite_snapshots(db) -> None:
    user, client = _portal_user(db)
    original_email = user.email
    original_phone = user.phone
    record_portal_terms_acceptance(user, client)

    user.email = "renamed@example.com"
    user.phone = "+41790000000"
    user.phone_verified_at = None
    db.session.flush()

    stored = list_portal_terms_acceptances(user.id)
    assert stored
    for row in stored:
        assert row.email_snapshot == original_email
        assert row.phone_snapshot == original_phone
        assert row.phone_verified_at_snapshot is not None


def test_business_update_and_delete_are_rejected(db) -> None:
    user, client = _portal_user(db)
    rows = record_portal_terms_acceptance(user, client)
    acceptance_id = rows[0].id

    nested = db.session.begin_nested()
    rows[0].terms_version = "1.1"
    with pytest.raises(SQLAlchemyError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()

    kept = db.session.get(ClientTermsAcceptance, acceptance_id)
    assert kept is not None
    assert kept.terms_version == "1.0"

    nested = db.session.begin_nested()
    db.session.delete(kept)
    with pytest.raises(SQLAlchemyError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    assert db.session.get(ClientTermsAcceptance, acceptance_id) is not None


def test_user_delete_does_not_remove_acceptances(db) -> None:
    user, client = _portal_user(db)
    rows = record_portal_terms_acceptance(user, client)
    acceptance_id = rows[0].id
    nested = db.session.begin_nested()
    db.session.delete(user)
    with pytest.raises(IntegrityError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    assert db.session.get(ClientTermsAcceptance, acceptance_id) is not None


def test_terms_route_exposes_only_insert_and_read() -> None:
    source = (BACKEND_ROOT / "routes" / "client_terms.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ClientMyTermsAcceptances":
            methods = {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            }
    assert methods == {"get", "post"}
