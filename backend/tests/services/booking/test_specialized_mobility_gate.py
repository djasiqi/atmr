"""Garde mobilité spécialisée (brancard / oxygène)."""

from __future__ import annotations

from services.booking.specialized_mobility_gate import (
    blocked_specialized_mobility_labels,
    reject_if_specialized_mobility_blocked,
)


def test_wheelchair_and_assistance_allowed():
    assert (
        blocked_specialized_mobility_labels(
            {"wheelchair": True, "needs_assistance": True}
        )
        == []
    )
    assert reject_if_specialized_mobility_blocked({"wheelchair": True}) is None


def test_stretcher_blocked_by_default():
    labels = blocked_specialized_mobility_labels({"stretcher": True})
    assert labels == ["brancard"]
    msg = reject_if_specialized_mobility_blocked({"stretcher": True, "oxygen": True})
    assert msg is not None
    assert "brancard" in msg
    assert "oxygène" in msg
