"""Tests unitaires — destination_billing_resolver."""

from __future__ import annotations

from dataclasses import dataclass, field

from services.billing.destination_billing_resolver import (
    build_billing_summary,
    effective_billing_for_leg,
    resolve_effective_billing_intent,
)


@dataclass
class _Leg:
    sequence_index: int = 0
    route_sequence_number: int = 1
    dropoff_location: str = "HUG"
    dropoff_establishment: str | None = None
    destination_billing_override: str | None = None
    is_return_stop: bool = False


@dataclass
class _Request:
    billing_intent: str = "institution"
    legs: list[_Leg] = field(default_factory=list)


def test_resolve_effective_billing_intent_inherits_primary():
    assert resolve_effective_billing_intent("institution", None) == "institution"


def test_resolve_effective_billing_intent_uses_override():
    assert resolve_effective_billing_intent("institution", "patient") == "patient"


def test_effective_billing_for_leg():
    req = _Request(
        billing_intent="institution",
        legs=[_Leg(destination_billing_override="patient")],
    )
    assert effective_billing_for_leg(req.legs[0], req) == "patient"


def test_build_billing_summary_multi_payer():
    req = _Request(
        billing_intent="institution",
        legs=[
            _Leg(
                dropoff_location="HUG",
                dropoff_establishment="HUG",
            ),
            _Leg(
                dropoff_location="Cabinet privé Dr X",
                dropoff_establishment="Cabinet privé Dr X",
                destination_billing_override="patient",
            ),
            _Leg(
                dropoff_location="EMS Genève",
                is_return_stop=True,
            ),
        ],
    )
    summary = build_billing_summary(req)
    assert summary["multi_payer"] is True
    assert summary["payer_count"] == 2
    assert summary["has_exceptions"] is True
    assert len(summary["exceptions"]) == 1
    assert summary["exceptions"][0]["destination_billing_override"] == "patient"


def test_primary_change_preserves_overrides():
    leg = _Leg(destination_billing_override="patient")
    req = _Request(billing_intent="institution", legs=[leg, _Leg()])
    assert effective_billing_for_leg(req.legs[0], req) == "patient"
    req.billing_intent = "insurance"
    assert effective_billing_for_leg(req.legs[0], req) == "patient"
    assert effective_billing_for_leg(req.legs[1], req) == "insurance"
