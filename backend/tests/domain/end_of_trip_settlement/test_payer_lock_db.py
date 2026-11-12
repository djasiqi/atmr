"""Verrou financier sur atmr_test. Transaction annulée, réservation intacte."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

pytestmark = pytest.mark.skipif(
    "lirie_e2e_" not in os.environ.get("DATABASE_URL", "")
    or "atmr_test" in os.environ.get("DATABASE_URL", ""),
    reason="rejeu sur base jetable uniquement, jamais sur atmr_test",
)

_INSERT = text(
    "INSERT INTO booking_settlement_policy ("
    "booking_id, revision, modality, authorized_modes, currency, "
    "decision_origin, policy_version, financial_responsibility_locked, "
    "locked_at, locked_by_user_id, validated_at, validated_by_user_id, status"
    ") VALUES ("
    ":booking_id, 1, 'on_site', '[\"tap_to_pay\", \"cash\"]'::jsonb, 'CHF', "
    "'institution', 'projet-2026-10', :locked, "
    "CASE WHEN :locked THEN now() ELSE NULL END, "
    "CASE WHEN :locked THEN :user_id ELSE NULL END, "
    "now(), :user_id, 'active')"
)


def _candidate(conn):
    row = conn.execute(
        text(
            "SELECT b.id, b.billed_to_type, b.updated_at "
            "FROM booking b "
            "WHERE b.billing_locked_at IS NULL "
            "AND b.invoice_line_id IS NULL "
            "AND NOT EXISTS ("
            "SELECT 1 FROM invoice_lines il "
            "JOIN invoices i ON i.id = il.invoice_id "
            "WHERE il.reservation_id = b.id "
            "AND i.status::text <> 'cancelled'"
            ") "
            "AND NOT EXISTS ("
            "SELECT 1 FROM booking_settlement_policy p "
            "WHERE p.booking_id = b.id"
            ") "
            "ORDER BY b.id DESC LIMIT 1"
        )
    ).one()
    user_id = conn.execute(
        text('SELECT id FROM "user" ORDER BY id DESC LIMIT 1')
    ).scalar()
    return int(row.id), row.billed_to_type, row.updated_at, int(user_id)


def test_changement_ordinaire_refuse_apres_verrou_puis_annule() -> None:
    from tests.support.disposable_bootstrap import boot_testing_app

    from application.billing.settlement_payer_lock import (
        lock_financial_responsibility,
        refuse_ordinary_payer_change,
    )
    from application.institutions.change_booking_payer import change_booking_payer
    from ext import db

    application = boot_testing_app()
    with application.app_context():
        booking_id, billed_before, updated_before, user_id = _candidate(db.session)
        db.session.execute(
            _INSERT,
            {"booking_id": booking_id, "user_id": user_id, "locked": False},
        )
        assert refuse_ordinary_payer_change(booking_id) is None
        assert (
            lock_financial_responsibility(booking_id=booking_id, actor_user_id=user_id)
            == "locked"
        )
        assert refuse_ordinary_payer_change(booking_id) == "payer_locked"
        from models import Booking

        result = change_booking_payer(
            db.session.get(Booking, booking_id),
            target_payer="clinic",
            transport_request=None,
            institution_id=1,
            actor_user_id=user_id,
            actor_role="institution_admin",
            actor_display_name="Test",
            override_reason="tentative après verrou",
            billing_change_reason_code="payer_correction",
        )
        booking = db.session.get(Booking, booking_id)
        assert result.ok is False
        assert result.status_code == 409
        assert booking.billed_to_type == billed_before
        db.session.rollback()
        left = db.session.execute(
            text(
                "SELECT count(*) FROM booking_settlement_policy WHERE booking_id = :id"
            ),
            {"id": booking_id},
        ).scalar()
        updated = db.session.execute(
            text("SELECT updated_at FROM booking WHERE id = :id"),
            {"id": booking_id},
        ).scalar()
        current_type = db.session.execute(
            text("SELECT billed_to_type FROM booking WHERE id = :id"),
            {"id": booking_id},
        ).scalar()
    assert left == 0
    assert updated == updated_before
    assert current_type == billed_before


def test_deux_transactions_se_serialisent_sur_la_reservation() -> None:
    from tests.support.disposable_bootstrap import boot_testing_app

    application = boot_testing_app()
    engine = create_engine(os.environ["DATABASE_URL"])
    with application.app_context():
        with engine.connect() as probe:
            booking_id, billed_before, updated_before, user_id = _candidate(probe)
        holder = engine.connect()
        waiter = engine.connect()
        try:
            held = holder.begin()
            holder.execute(
                text("SELECT id FROM booking WHERE id = :id FOR UPDATE"),
                {"id": booking_id},
            )
            holder.execute(
                _INSERT,
                {"booking_id": booking_id, "user_id": user_id, "locked": True},
            )
            waiter.execute(text("SET lock_timeout = '1500ms'"))
            with pytest.raises(OperationalError):
                waiter.execute(
                    text("SELECT id FROM booking WHERE id = :id FOR UPDATE"),
                    {"id": booking_id},
                )
            waiter.rollback()
            held.rollback()
        finally:
            holder.close()
            waiter.close()
        with engine.connect() as probe:
            left = probe.execute(
                text(
                    "SELECT count(*) FROM booking_settlement_policy "
                    "WHERE booking_id = :id"
                ),
                {"id": booking_id},
            ).scalar()
            updated = probe.execute(
                text("SELECT updated_at, billed_to_type FROM booking WHERE id = :id"),
                {"id": booking_id},
            ).one()
    assert left == 0
    assert updated.updated_at == updated_before
    assert updated.billed_to_type == billed_before
