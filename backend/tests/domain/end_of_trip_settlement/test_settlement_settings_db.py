"""Préférences et revue d'acceptation sur atmr_test. Tout est annulé."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    "lirie_e2e_" not in os.environ.get("DATABASE_URL", "")
    or "atmr_test" in os.environ.get("DATABASE_URL", ""),
    reason="rejeu sur base jetable uniquement, jamais sur atmr_test",
)


def _app():
    from tests.support.disposable_bootstrap import boot_testing_app

    return boot_testing_app()


def test_preferences_n_activent_pas_les_moyens() -> None:
    from application.companies.settlement_settings import update_preferences
    from ext import db

    application = _app()
    with application.app_context():
        company_id = int(
            db.session.execute(text("SELECT id FROM company ORDER BY id LIMIT 1")).scalar()
        )
        user_id = int(
            db.session.execute(text('SELECT id FROM "user" ORDER BY id LIMIT 1')).scalar()
        )
        before = db.session.execute(
            text(
                "SELECT count(*) FROM company_settlement_capability "
                "WHERE company_id = :id"
            ),
            {"id": company_id},
        ).scalar()
        try:
            with pytest.raises(PermissionError):
                update_preferences(
                    company_id=company_id,
                    actor_role="driver",
                    actor_user_id=user_id,
                    same_company=True,
                    payload={"preferences": {"cash": True}},
                )
            with pytest.raises(PermissionError):
                update_preferences(
                    company_id=company_id,
                    actor_role="company",
                    actor_user_id=user_id,
                    same_company=False,
                    payload={"preferences": {"cash": True}},
                )
            with pytest.raises(ValueError, match="ne rend pas le moyen"):
                update_preferences(
                    company_id=company_id,
                    actor_role="company",
                    actor_user_id=user_id,
                    same_company=True,
                    payload={"cash_enabled": True, "preferences": {"cash": True}},
                )
            view = update_preferences(
                company_id=company_id,
                actor_role="company",
                actor_user_id=user_id,
                same_company=True,
                payload={
                    "preferences": {
                        "tap_to_pay": True,
                        "cash": True,
                        "direct_patient_invoice": True,
                    }
                },
            )
            assert view["preferences"] == {
                "tap_to_pay": True,
                "cash": True,
                "direct_patient_invoice": True,
            }
            assert view["effectively_available"] == {
                "tap_to_pay": False,
                "cash": False,
                "direct_patient_invoice": False,
            }
            assert view["activation_enabled"] is False
            assert len(view["history"]) == 1
            enabled = db.session.execute(
                text(
                    "SELECT tap_to_pay_enabled, cash_enabled, "
                    "direct_patient_invoice_enabled "
                    "FROM company_settlement_capability WHERE company_id = :id"
                ),
                {"id": company_id},
            ).one()
            assert list(enabled) == [False, False, False]
        finally:
            db.session.rollback()
        after = db.session.execute(
            text(
                "SELECT count(*) FROM company_settlement_capability "
                "WHERE company_id = :id"
            ),
            {"id": company_id},
        ).scalar()
        assert after == before


def test_declaration_patient_n_est_pas_une_preuve() -> None:
    from application.companies.settlement_settings import record_patient_notice
    from ext import db

    application = _app()
    with application.app_context():
        booking_id = int(
            db.session.execute(text("SELECT id FROM booking ORDER BY id LIMIT 1")).scalar()
        )
        company_id = int(
            db.session.execute(text("SELECT id FROM company ORDER BY id LIMIT 1")).scalar()
        )
        user_id = int(
            db.session.execute(text('SELECT id FROM "user" ORDER BY id LIMIT 1')).scalar()
        )
        try:
            with pytest.raises(ValueError, match="incomplète ou précochée"):
                record_patient_notice(
                    booking_id=booking_id,
                    company_id=company_id,
                    actor_role="company",
                    actor_user_id=user_id,
                    same_company=True,
                    payload={
                        "confirmed": True,
                        "auto_checked": True,
                        "conditions_version": "projet-2026-10",
                        "amount_chf": "45.00",
                        "debtor_label": "Patient",
                        "channel": "verbal",
                    },
                )
            view = record_patient_notice(
                booking_id=booking_id,
                company_id=company_id,
                actor_role="company",
                actor_user_id=user_id,
                same_company=True,
                payload={
                    "confirmed": True,
                    "auto_checked": False,
                    "conditions_version": "projet-2026-10",
                    "amount_chf": "45.00",
                    "debtor_label": "Patient",
                    "channel": "verbal",
                    "presented_modes": ["cash"],
                },
            )
            assert view["receipt_proven"] is False
            assert view["patient_acceptance_proven"] is False
            assert view["author_user_id"] == user_id
        finally:
            db.session.rollback()
        left = db.session.execute(
            text(
                "SELECT count(*) FROM booking_settlement_patient_notice "
                "WHERE booking_id = :id"
            ),
            {"id": booking_id},
        ).scalar()
        assert left == 0


def test_acceptation_ne_cree_pas_de_politique_active() -> None:
    from application.billing import settlement_acceptance
    from domain.end_of_trip_settlement import activation
    from ext import db
    from models import Booking

    application = _app()
    with application.app_context():
        row = db.session.execute(
            text(
                "SELECT id, company_id, billed_to_type::text AS payer, "
                "updated_at FROM booking "
                "WHERE company_id IS NOT NULL ORDER BY id DESC LIMIT 1"
            )
        ).one()
        booking = db.session.get(Booking, int(row.id))
        before = db.session.execute(
            text(
                "SELECT count(*) FROM booking_settlement_policy "
                "WHERE booking_id = :id"
            ),
            {"id": int(row.id)},
        ).scalar()
        updated_before = row.updated_at
        payer_before = row.payer
        activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = True
        try:
            reason = settlement_acceptance.note_acceptance_financial_review(
                booking, company_id=int(row.company_id)
            )
            assert reason != "skipped"
            assert reason != "refused"
            other = db.session.execute(
                text("SELECT id FROM company WHERE id <> :id ORDER BY id LIMIT 1"),
                {"id": int(row.company_id)},
            ).scalar()
            if other is not None:
                mismatch = settlement_acceptance.note_acceptance_financial_review(
                    booking, company_id=int(other)
                )
                assert mismatch == "company_mismatch"
            again = settlement_acceptance.note_acceptance_financial_review(
                booking, company_id=int(row.company_id)
            )
            assert again == reason

            def _boom(*_args, **_kwargs):
                raise RuntimeError("incident de préparation")

            original = settlement_acceptance._review
            settlement_acceptance._review = _boom
            try:
                failed = settlement_acceptance.note_acceptance_financial_review(
                    booking, company_id=int(row.company_id)
                )
            finally:
                settlement_acceptance._review = original
            assert failed == "financial_review_unavailable"
        finally:
            activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = False
            db.session.rollback()
        after = db.session.execute(
            text(
                "SELECT count(*) FROM booking_settlement_policy "
                "WHERE booking_id = :id"
            ),
            {"id": int(row.id)},
        ).scalar()
        current = db.session.execute(
            text("SELECT billed_to_type::text AS payer, updated_at FROM booking WHERE id = :id"),
            {"id": int(row.id)},
        ).one()
        assert after == before
        assert current.payer == payer_before
        assert current.updated_at == updated_before


def test_politique_confirmee_n_est_pas_ecrasee() -> None:
    from application.billing import settlement_acceptance
    from application.companies.settlement_settings import update_preferences
    from domain.end_of_trip_settlement import activation
    from ext import db
    from models import Booking

    application = _app()
    with application.app_context():
        row = db.session.execute(
            text(
                "SELECT b.id, b.company_id FROM booking b "
                "WHERE b.company_id IS NOT NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM booking_settlement_policy p "
                "WHERE p.booking_id = b.id) "
                "ORDER BY b.id DESC LIMIT 1"
            )
        ).one()
        user_id = int(
            db.session.execute(text('SELECT id FROM "user" ORDER BY id LIMIT 1')).scalar()
        )
        activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = True
        try:
            db.session.execute(
                text(
                    "INSERT INTO booking_settlement_policy ("
                    "booking_id, revision, modality, authorized_modes, currency, "
                    "decision_origin, policy_version, financial_responsibility_locked, "
                    "validated_at, validated_by_user_id, status"
                    ") VALUES ("
                    ":booking_id, 1, 'on_site', '[\"cash\"]'::jsonb, 'CHF', "
                    "'company', 'projet-2026-10', false, now(), :user_id, 'active')"
                ),
                {"booking_id": int(row.id), "user_id": user_id},
            )
            reason = settlement_acceptance.note_acceptance_financial_review(
                db.session.get(Booking, int(row.id)),
                company_id=int(row.company_id),
            )
            assert reason == "policy_already_recorded"
            update_preferences(
                company_id=int(row.company_id),
                actor_role="company",
                actor_user_id=user_id,
                same_company=True,
                payload={"preferences": {"tap_to_pay": True}},
            )
            kept = db.session.execute(
                text(
                    "SELECT modality, authorized_modes::text FROM booking_settlement_policy "
                    "WHERE booking_id = :id AND status = 'active'"
                ),
                {"id": int(row.id)},
            ).one()
            assert kept.modality == "on_site"
            assert "cash" in kept[1]
            active_count = db.session.execute(
                text(
                    "SELECT count(*) FROM booking_settlement_policy "
                    "WHERE booking_id = :id AND status = 'active'"
                ),
                {"id": int(row.id)},
            ).scalar()
            assert active_count == 1
        finally:
            activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = False
            db.session.rollback()
        left = db.session.execute(
            text(
                "SELECT count(*) FROM booking_settlement_policy WHERE booking_id = :id"
            ),
            {"id": int(row.id)},
        ).scalar()
        assert left == 0
