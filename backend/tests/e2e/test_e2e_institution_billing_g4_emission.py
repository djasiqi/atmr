"""G4 — deux E2E d'émission institution. Pas de revalidation G1 / G2.

Chaîne : state → eligibility → plan → preview → facture → PDF → QR.
"""

from __future__ import annotations

import pytest

from tests.e2e.helpers.institution_billing_g4 import (
    assert_emission_chain,
    build_g4_world,
    resolve_marie_carrier,
    resolve_marie_institution,
)

pytestmark = pytest.mark.e2e


def test_g4_reintegration_resolved_carrier_emits_360(db):
    world = build_g4_world(db)
    resolve_marie_carrier(world["marie"])
    assert_emission_chain(
        world,
        expected_status="resolved_carrier",
        expected_total=360.0,
        marie_in=True,
    )


def test_g4_exclusion_resolved_institution_emits_320(db):
    world = build_g4_world(db)
    resolve_marie_institution(world["marie"])
    assert_emission_chain(
        world,
        expected_status="resolved_institution",
        expected_total=320.0,
        marie_in=False,
    )
