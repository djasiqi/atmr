"""Persistance de la politique sur atmr_test. Transaction annulée."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    "lirie_e2e_" not in os.environ.get("DATABASE_URL", "")
    or "atmr_test" in os.environ.get("DATABASE_URL", ""),
    reason="rejeu sur base jetable uniquement, jamais sur atmr_test",
)


def test_politique_inseree_puis_annulee_sans_toucher_la_reservation() -> None:
    from tests.support.disposable_bootstrap import boot_testing_app

    application = boot_testing_app()
    engine = create_engine(os.environ["DATABASE_URL"])
    with application.app_context(), engine.connect() as conn:
        booking_id = conn.execute(
            text("SELECT id FROM booking ORDER BY id DESC LIMIT 1")
        ).scalar()
        user_id = conn.execute(
            text('SELECT id FROM "user" ORDER BY id DESC LIMIT 1')
        ).scalar()
        before_booking = conn.execute(
            text("SELECT updated_at FROM booking WHERE id = :id"),
            {"id": booking_id},
        ).scalar()
        payer_columns = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'booking_settlement_policy' "
                "AND column_name IN "
                "('billed_to_type', 'billing_party_id', 'billing_intent')"
            )
        ).all()
        assert booking_id is not None
        assert user_id is not None
        assert payer_columns == []
        try:
            conn.execute(
                text(
                    "INSERT INTO booking_settlement_policy ("
                    "booking_id, revision, modality, authorized_modes, currency, "
                    "decision_origin, policy_version, validated_at, "
                    "validated_by_user_id, status"
                    ") VALUES ("
                    ":booking_id, 1, 'on_site', "
                    "'[\"tap_to_pay\", \"cash\"]'::jsonb, 'CHF', "
                    "'client', 'projet-2026-10', now(), :user_id, 'active'"
                    ")"
                ),
                {"booking_id": booking_id, "user_id": user_id},
            )
            with pytest.raises(IntegrityError), conn.begin_nested():
                conn.execute(
                    text(
                        "INSERT INTO booking_settlement_policy ("
                        "booking_id, revision, modality, authorized_modes, "
                        "currency, decision_origin, policy_version, "
                        "validated_at, validated_by_user_id, status"
                        ") VALUES ("
                        ":booking_id, 2, 'on_site', '[\"cash\"]'::jsonb, "
                        "'CHF', 'client', 'projet-2026-10', now(), "
                        ":user_id, 'active')"
                    ),
                    {"booking_id": booking_id, "user_id": user_id},
                )
            with pytest.raises(IntegrityError), conn.begin_nested():
                conn.execute(
                    text(
                        "INSERT INTO booking_settlement_policy ("
                        "booking_id, revision, modality, authorized_modes, "
                        "currency, decision_origin, policy_version, "
                        "financial_responsibility_locked, validated_at, "
                        "validated_by_user_id, status"
                        ") VALUES ("
                        ":booking_id, 3, 'on_site', '[\"cash\"]'::jsonb, "
                        "'CHF', 'institution', 'projet-2026-10', true, "
                        "now(), :user_id, 'superseded')"
                    ),
                    {"booking_id": booking_id, "user_id": user_id},
                )
        finally:
            conn.rollback()
        with engine.connect() as check:
            left = check.execute(
                text(
                    "SELECT count(*) FROM booking_settlement_policy "
                    "WHERE booking_id = :id"
                ),
                {"id": booking_id},
            ).scalar()
            updated = check.execute(
                text("SELECT updated_at FROM booking WHERE id = :id"),
                {"id": booking_id},
            ).scalar()
    assert left == 0
    assert updated == before_booking
