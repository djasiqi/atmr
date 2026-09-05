"""G1 — plan institution == preview (monde 320 / 360). Pas d'émission PDF/QR."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from application.invoices.booking_dispute import g1_financials as g1
from application.invoices.booking_dispute.service import (
    add_carrier_evidence,
    carrier_respond,
    decide_dispute,
    ensure_open_dispute,
    submit_dispute_for_validation,
)
from application.invoices.institution_invoice_plan import build_institution_invoice_plan
from application.invoices.period_invoice_preview import build_period_invoice_preview
from ext import db
from tests.application.helpers.g1_clinic360_world import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    build_g1_clinic360_world,
)

ZURICH = ZoneInfo("Europe/Zurich")
NOW = datetime(2026, 8, 20, 12, 0, tzinfo=ZURICH)


def _plan(world):
    return build_institution_invoice_plan(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        clinic_client_id=world["clinic_client"].id,
        now=NOW,
    )


def _preview(world):
    return build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        include_line_details=True,
        now=NOW,
    )


def _assert_plan_preview_oracle(world, *, expected_total: float, marie_in: bool):
    db.session.flush()
    marie = world["marie"]
    surface = g1.institution_surface(world["all_clinic"])
    plan = _plan(world)
    preview = _preview(world)
    clinic_ht = float(plan.clinic.estimated_total) if plan.clinic else 0.0
    assert surface["institution_total"] == expected_total
    assert clinic_ht == expected_total
    assert float(preview.estimated_total) == expected_total
    assert clinic_ht == float(preview.estimated_total)
    if marie_in:
        assert int(marie.id) in surface["eligible_lines"]
    else:
        assert int(marie.id) in surface["excluded_lines"]


def test_g1_db_disputed_then_resolved_carrier(db):
    world = build_g1_clinic360_world(db)
    marie = world["marie"]
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)

    ensure_open_dispute(marie, actor_role="institution")
    carrier_respond(
        marie, stance="mission_done", actor_user_id=None, actor_role="COMPANY"
    )
    db.session.flush()
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)

    denied = decide_dispute(
        marie,
        decision="accept_carrier",
        note=None,
        actor_user_id=None,
        actor_role="COMPANY",
    )
    assert denied.status_code == 403
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)

    add_carrier_evidence(
        marie,
        kind="signed_transport_sheet",
        note="bon",
        actor_user_id=None,
        actor_role="COMPANY",
    )
    submit_dispute_for_validation(marie, actor_user_id=None, actor_role="COMPANY")
    db.session.flush()
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)

    decide_dispute(
        marie,
        decision="accept_carrier",
        note="ok",
        actor_user_id=None,
        actor_role="institution_admin",
    )
    db.session.flush()
    _assert_plan_preview_oracle(world, expected_total=360.0, marie_in=True)


def test_g1_db_institution_right_stays_320(db):
    world = build_g1_clinic360_world(db)
    marie = world["marie"]
    ensure_open_dispute(marie, actor_role="institution")
    result = carrier_respond(
        marie,
        stance="institution_right",
        exclusion_reason="created_by_error",
        actor_user_id=None,
        actor_role="COMPANY",
    )
    assert result.ok is True
    db.session.flush()
    assert marie.billed_to_type == "clinic"
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)


def test_g1_db_correction_clinic_355_and_patient_320(db):
    world = build_g1_clinic360_world(db)
    marie = world["marie"]
    ensure_open_dispute(marie, actor_role="institution")
    carrier_respond(
        marie,
        stance="needs_correction",
        proposed_amount_ht=35,
        proposed_payer_type="clinic",
        actor_user_id=None,
        actor_role="COMPANY",
    )
    add_carrier_evidence(
        marie, kind="signed_transport_sheet", actor_user_id=None, actor_role="COMPANY"
    )
    submit_dispute_for_validation(marie, actor_user_id=None, actor_role="COMPANY")
    db.session.flush()
    assert float(marie.amount) == 40.0
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)

    decide_dispute(
        marie,
        decision="accept_carrier",
        note="35 clinic",
        actor_user_id=None,
        actor_role="institution_admin",
    )
    db.session.flush()
    _assert_plan_preview_oracle(world, expected_total=355.0, marie_in=True)


def test_g1_db_correction_patient_leaves_institution_320(db):
    world = build_g1_clinic360_world(db)
    marie = world["marie"]
    ensure_open_dispute(marie, actor_role="institution")
    carrier_respond(
        marie,
        stance="needs_correction",
        proposed_amount_ht=35,
        proposed_payer_type="patient",
        actor_user_id=None,
        actor_role="COMPANY",
    )
    add_carrier_evidence(
        marie, kind="institution_written", actor_user_id=None, actor_role="COMPANY"
    )
    submit_dispute_for_validation(marie, actor_user_id=None, actor_role="COMPANY")
    db.session.flush()
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)

    decide_dispute(
        marie,
        decision="accept_carrier",
        note="patient",
        actor_user_id=None,
        actor_role="admin",
    )
    db.session.flush()
    assert marie.billed_to_type == "patient"
    _assert_plan_preview_oracle(world, expected_total=320.0, marie_in=False)
