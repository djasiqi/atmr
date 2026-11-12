"""Persistance de règlement sur une base PostgreSQL jetable.

Les sept tests historiques qui exigent atmr_test ne sont pas lancés ici :
ils sélectionnent des réservations partagées. Les mêmes contrôles sont
rejoués sur des lignes fictives, puis la base est supprimée.
"""

from __future__ import annotations

from tests.support.recipe_credentials import RECIPE_PASSWORD

import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from flask import Flask
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from ext import db

FORBIDDEN_DATABASES = frozenset({"atmr", "atmr_test", "atmrstg", "postgres", ""})
HISTORICAL_IDS = (34994, 35027, 35028)

_INSERT = text(
    "INSERT INTO booking_settlement_policy ("
    "booking_id, revision, modality, authorized_modes, currency, "
    "decision_origin, policy_version, financial_responsibility_locked, "
    "locked_at, locked_by_user_id, validated_at, validated_by_user_id, status"
    ") VALUES ("
    ":booking_id, :revision, 'on_site', '[\"cash\"]'::jsonb, 'CHF', "
    "'institution', 'projet-2026-10', :locked, "
    "CASE WHEN :locked THEN now() ELSE NULL END, "
    "CASE WHEN :locked THEN :user_id ELSE NULL END, "
    "now(), :user_id, 'active')"
)


def _database_name() -> str:
    url = os.environ.get("DATABASE_URL", "")
    name = url.rsplit("/", 1)[-1].split("?", 1)[0].strip()
    if name in FORBIDDEN_DATABASES or not name.startswith("lirie_e2e_"):
        pytest.skip(f"Base refusée pour cette recette : {name or 'absente'}")
    if "atmr_test" in url or url.rstrip("/").endswith("/atmr"):
        pytest.fail("L'URL vise une base partagée.")
    return name


@pytest.fixture(scope="module")
def isolated():
    expected = _database_name()
    app = Flask("lirie_e2e_s14")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URL"]
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    context = app.app_context()
    context.push()
    current = db.session.execute(text("SELECT current_database()")).scalar()
    if current != expected:
        pytest.fail(f"Connexion inattendue : {current}")
    historical = db.session.execute(
        text("SELECT id FROM booking WHERE id IN (34994, 35027, 35028)")
    ).all()
    if historical:
        pytest.fail("Des réservations historiques sont présentes.")
    for table in (
        "booking_settlement_policy",
        "company_settlement_capability",
        "booking_settlement_patient_notice",
    ):
        ready = db.session.execute(
            text("SELECT to_regclass(:name)"),
            {"name": f"public.{table}"},
        ).scalar()
        if ready is None:
            pytest.fail(f"Table absente du schéma cloné : {table}")
    world = _seed()
    yield world
    db.session.remove()
    context.pop()


def _user(role: str, label: str):
    from models import User, UserRole

    row = User(
        public_id=str(uuid.uuid4()),
        username=label,
        email=f"{label}@lirie-e2e.invalid",
        password=RECIPE_PASSWORD,
        role=UserRole(role),
        first_name="Fictif",
        last_name=label,
    )
    db.session.add(row)
    db.session.flush()
    return row


def _seed():
    from models import Company, Driver

    company_user = _user("COMPANY", f"s14-co-{uuid.uuid4().hex[:8]}")
    company = Company(
        name="Entreprise fictive justificatif",
        user_id=company_user.id,
        platform_billing_access_state="active",
    )
    other_user = _user("COMPANY", f"s14-b-{uuid.uuid4().hex[:8]}")
    other = Company(
        name="Autre entreprise fictive",
        user_id=other_user.id,
        platform_billing_access_state="active",
    )
    db.session.add_all([company, other])
    db.session.flush()
    driver_user = _user("DRIVER", f"s14-ch-{uuid.uuid4().hex[:8]}")
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    db.session.add(driver)
    db.session.commit()
    return {
        "company": company,
        "other": other,
        "driver": driver,
        "user_id": company_user.id,
    }


def _booking(world):
    from models import Booking, BookingStatus

    row = Booking(
        company_id=world["company"].id,
        executing_company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount=45.0,
        status=BookingStatus.ASSIGNED,
        scheduled_time=datetime(2026, 10, 26, 9, 0),
        pickup_location="Clinique fictive",
        dropoff_location="Domicile fictif",
        customer_name="Patient fictif",
        billed_to_type="patient",
    )
    db.session.add(row)
    db.session.commit()
    assert row.id not in HISTORICAL_IDS
    return row


def _policy_count(booking_id: int) -> int:
    return int(
        db.session.execute(
            text(
                "SELECT count(*) FROM booking_settlement_policy WHERE booking_id = :id"
            ),
            {"id": booking_id},
        ).scalar()
    )


def _payment_count(booking_id: int) -> int:
    return int(
        db.session.execute(
            text("SELECT count(*) FROM payment WHERE booking_id = :id"),
            {"id": booking_id},
        ).scalar()
    )


def test_verrou_refuse_le_changement_puis_disparait(isolated):
    from application.billing.settlement_payer_lock import (
        lock_financial_responsibility,
        refuse_ordinary_payer_change,
    )
    from application.institutions.change_booking_payer import change_booking_payer

    booking = _booking(isolated)
    updated_before = booking.updated_at
    billed_before = booking.billed_to_type
    amount_before = booking.amount
    try:
        db.session.execute(
            _INSERT,
            {
                "booking_id": booking.id,
                "revision": 1,
                "user_id": isolated["user_id"],
                "locked": False,
            },
        )
        assert refuse_ordinary_payer_change(booking.id) is None
        assert (
            lock_financial_responsibility(
                booking_id=booking.id, actor_user_id=isolated["user_id"]
            )
            == "locked"
        )
        assert (
            lock_financial_responsibility(
                booking_id=booking.id, actor_user_id=isolated["user_id"]
            )
            == "already_locked"
        )
        result = change_booking_payer(
            booking,
            target_payer="clinic",
            transport_request=None,
            institution_id=1,
            actor_user_id=isolated["user_id"],
            actor_role="institution_admin",
            actor_display_name="Fictif",
            override_reason="tentative apres verrou",
            billing_change_reason_code="payer_correction",
        )
        assert result.ok is False
        assert result.status_code == 409
        assert booking.billed_to_type == billed_before
        assert booking.amount == amount_before
    finally:
        db.session.rollback()
    assert _policy_count(booking.id) == 0
    assert _payment_count(booking.id) == 0
    current = db.session.get(type(booking), booking.id)
    assert current.billed_to_type == billed_before
    assert current.updated_at == updated_before
    assert Decimal(str(current.amount)) == Decimal("45.00")


def test_deux_connexions_se_serialisent(isolated):
    booking = _booking(isolated)
    updated_before = booking.updated_at
    engine = create_engine(os.environ["DATABASE_URL"])
    holder = engine.connect()
    waiter = engine.connect()
    try:
        held = holder.begin()
        holder.execute(
            text("SELECT id FROM booking WHERE id = :id FOR UPDATE"),
            {"id": booking.id},
        )
        holder.execute(
            _INSERT,
            {
                "booking_id": booking.id,
                "revision": 1,
                "user_id": isolated["user_id"],
                "locked": True,
            },
        )
        waiter.execute(text("SET lock_timeout = '1500ms'"))
        with pytest.raises(OperationalError):
            waiter.execute(
                text("SELECT id FROM booking WHERE id = :id FOR UPDATE"),
                {"id": booking.id},
            )
        waiter.rollback()
        held.rollback()
    finally:
        holder.close()
        waiter.close()
        engine.dispose()
    assert _policy_count(booking.id) == 0
    current = db.session.execute(
        text("SELECT updated_at, amount FROM booking WHERE id = :id"),
        {"id": booking.id},
    ).one()
    assert current.updated_at == updated_before
    assert Decimal(str(current.amount)) == Decimal("45.00")


def test_politique_active_unique_et_verrou_complet(isolated):
    booking = _booking(isolated)
    before = booking.updated_at
    payer_columns = db.session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'booking_settlement_policy' "
            "AND column_name IN "
            "('billed_to_type', 'billing_party_id', 'billing_intent')"
        )
    ).all()
    assert payer_columns == []
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        try:
            conn.execute(
                _INSERT,
                {
                    "booking_id": booking.id,
                    "revision": 1,
                    "user_id": isolated["user_id"],
                    "locked": False,
                },
            )
            with pytest.raises(IntegrityError), conn.begin_nested():
                conn.execute(
                    _INSERT,
                    {
                        "booking_id": booking.id,
                        "revision": 2,
                        "user_id": isolated["user_id"],
                        "locked": False,
                    },
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
                        "now(), :user_id, 'active')"
                    ),
                    {
                        "booking_id": booking.id,
                        "user_id": isolated["user_id"],
                    },
                )
        finally:
            conn.rollback()
    engine.dispose()
    assert _policy_count(booking.id) == 0
    updated = db.session.execute(
        text("SELECT updated_at FROM booking WHERE id = :id"),
        {"id": booking.id},
    ).scalar()
    assert updated == before


def test_preferences_n_activent_pas_les_moyens(isolated):
    from application.companies.settlement_settings import update_preferences

    before = db.session.execute(
        text(
            "SELECT count(*) FROM company_settlement_capability WHERE company_id = :id"
        ),
        {"id": isolated["company"].id},
    ).scalar()
    other_before = db.session.execute(
        text(
            "SELECT count(*) FROM company_settlement_capability WHERE company_id = :id"
        ),
        {"id": isolated["other"].id},
    ).scalar()
    try:
        with pytest.raises(PermissionError):
            update_preferences(
                company_id=isolated["company"].id,
                actor_role="driver",
                actor_user_id=isolated["user_id"],
                same_company=True,
                payload={"preferences": {"cash": True}},
            )
        with pytest.raises(PermissionError):
            update_preferences(
                company_id=isolated["company"].id,
                actor_role="company",
                actor_user_id=isolated["user_id"],
                same_company=False,
                payload={"preferences": {"cash": True}},
            )
        with pytest.raises(ValueError, match="ne rend pas le moyen"):
            update_preferences(
                company_id=isolated["company"].id,
                actor_role="company",
                actor_user_id=isolated["user_id"],
                same_company=True,
                payload={"cash_enabled": True, "preferences": {"cash": True}},
            )
        view = update_preferences(
            company_id=isolated["company"].id,
            actor_role="company",
            actor_user_id=isolated["user_id"],
            same_company=True,
            payload={
                "preferences": {
                    "tap_to_pay": True,
                    "cash": True,
                    "direct_patient_invoice": True,
                }
            },
        )
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
            {"id": isolated["company"].id},
        ).one()
        assert list(enabled) == [False, False, False]
        foreign = db.session.execute(
            text(
                "SELECT count(*) FROM company_settlement_capability "
                "WHERE company_id = :id"
            ),
            {"id": isolated["other"].id},
        ).scalar()
        assert foreign == other_before
    finally:
        db.session.rollback()
    after = db.session.execute(
        text(
            "SELECT count(*) FROM company_settlement_capability WHERE company_id = :id"
        ),
        {"id": isolated["company"].id},
    ).scalar()
    assert after == before


def test_declaration_patient_n_est_pas_une_preuve(isolated):
    from application.companies.settlement_settings import record_patient_notice

    booking = _booking(isolated)
    try:
        with pytest.raises(ValueError, match="incomplète ou précochée"):
            record_patient_notice(
                booking_id=booking.id,
                company_id=isolated["company"].id,
                actor_role="company",
                actor_user_id=isolated["user_id"],
                same_company=True,
                payload={
                    "confirmed": True,
                    "auto_checked": True,
                    "conditions_version": "2026-10-26.draft-2",
                    "amount_chf": "45.00",
                    "debtor_label": "Patient fictif",
                    "channel": "verbal",
                },
            )
        view = record_patient_notice(
            booking_id=booking.id,
            company_id=isolated["company"].id,
            actor_role="company",
            actor_user_id=isolated["user_id"],
            same_company=True,
            payload={
                "confirmed": True,
                "auto_checked": False,
                "conditions_version": "2026-10-26.draft-2",
                "amount_chf": "45.00",
                "debtor_label": "Patient fictif",
                "channel": "verbal",
                "presented_modes": ["cash"],
            },
        )
        assert view["receipt_proven"] is False
        assert view["patient_acceptance_proven"] is False
        assert _payment_count(booking.id) == 0
    finally:
        db.session.rollback()
    left = db.session.execute(
        text(
            "SELECT count(*) FROM booking_settlement_patient_notice "
            "WHERE booking_id = :id"
        ),
        {"id": booking.id},
    ).scalar()
    assert left == 0


def test_acceptation_ne_cree_pas_de_politique(isolated):
    from application.billing import settlement_acceptance
    from domain.end_of_trip_settlement import activation

    booking = _booking(isolated)
    before = _policy_count(booking.id)
    updated_before = booking.updated_at
    payer_before = str(booking.billed_to_type)
    amount_before = Decimal(str(booking.amount))
    activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = True
    try:
        reason = settlement_acceptance.note_acceptance_financial_review(
            booking, company_id=isolated["company"].id
        )
        assert reason != "skipped"
        assert reason != "refused"
        mismatch = settlement_acceptance.note_acceptance_financial_review(
            booking, company_id=isolated["other"].id
        )
        assert mismatch == "company_mismatch"
        again = settlement_acceptance.note_acceptance_financial_review(
            booking, company_id=isolated["company"].id
        )
        assert again == reason

        def _boom(*_args, **_kwargs):
            raise RuntimeError("incident de préparation")

        original = settlement_acceptance._review
        settlement_acceptance._review = _boom
        try:
            failed = settlement_acceptance.note_acceptance_financial_review(
                booking, company_id=isolated["company"].id
            )
        finally:
            settlement_acceptance._review = original
        assert failed == "financial_review_unavailable"
        assert _payment_count(booking.id) == 0
    finally:
        activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = False
        db.session.rollback()
    assert _policy_count(booking.id) == before
    current = db.session.execute(
        text(
            "SELECT amount, updated_at, billed_to_type::text AS payer "
            "FROM booking WHERE id = :id"
        ),
        {"id": booking.id},
    ).one()
    assert Decimal(str(current.amount)) == amount_before
    assert current.updated_at == updated_before
    assert current.payer == payer_before


def test_politique_confirmee_n_est_pas_ecrasee(isolated):
    from application.billing import settlement_acceptance
    from application.companies.settlement_settings import update_preferences
    from domain.end_of_trip_settlement import activation

    booking = _booking(isolated)
    activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = True
    try:
        db.session.execute(
            _INSERT,
            {
                "booking_id": booking.id,
                "revision": 1,
                "user_id": isolated["user_id"],
                "locked": False,
            },
        )
        reason = settlement_acceptance.note_acceptance_financial_review(
            booking, company_id=isolated["company"].id
        )
        assert reason == "policy_already_recorded"
        update_preferences(
            company_id=isolated["company"].id,
            actor_role="company",
            actor_user_id=isolated["user_id"],
            same_company=True,
            payload={"preferences": {"tap_to_pay": True}},
        )
        kept = db.session.execute(
            text(
                "SELECT modality, authorized_modes::text, amount_chf "
                "FROM booking_settlement_policy "
                "WHERE booking_id = :id AND status = 'active'"
            ),
            {"id": booking.id},
        ).one()
        assert kept.modality == "on_site"
        assert "cash" in kept[1]
        assert kept.amount_chf is None
        assert _payment_count(booking.id) == 0
    finally:
        activation.SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = False
        db.session.rollback()
    assert _policy_count(booking.id) == 0


def test_simulation_n_ecrit_ni_paiement_ni_signature(isolated):
    from domain.end_of_trip_settlement.prototypes import declare_cash

    booking = _booking(isolated)
    reference = booking.id
    declared = declare_cash(
        booking_id=reference,
        payment_scope="booking",
        expected_amount_chf=Decimal("45.00"),
        received_amount_chf=Decimal("45.00"),
        company_id=isolated["company"].id,
        driver_id=1,
        recorded_at=datetime(2026, 10, 26, 18, 0, tzinfo=UTC),
        cash_reference="simulation-non-persistante",
    )
    assert declared.electronic_confirmation is False
    assert _payment_count(reference) == 0
    invoices = db.session.execute(text("SELECT count(*) FROM invoices")).scalar()
    assert int(invoices) == 0
    assert db.session.get(type(booking), reference).id == reference
