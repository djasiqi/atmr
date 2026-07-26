"""Tests unitaires — règles Valider / Planifier / Départ immédiat."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from services.institutions.offer_accept_rules import (
    can_validate_without_proposed_pickup,
    has_confirmed_departure,
    has_confirmed_rdv_only,
    is_departure_stale,
    validate_accept_pickup_rules,
)


def _req(**kwargs):
    base = {
        "pickup_time_confirmed": False,
        "scheduled_time": None,
        "scheduled_time_type": "departure",
        "appointment_time_confirmed": None,
        "legs": [],
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestOfferAcceptRules:
    def test_has_confirmed_departure(self):
        assert has_confirmed_departure(
            _req(
                pickup_time_confirmed=True, scheduled_time=datetime(2026, 6, 22, 19, 15)
            )
        )
        assert not has_confirmed_departure(
            _req(
                pickup_time_confirmed=True,
                scheduled_time=datetime(2026, 6, 22, 20, 0),
                scheduled_time_type="arrival",
            )
        )

    def test_rdv_only_without_departure(self):
        leg = SimpleNamespace(
            scheduled_time=datetime(2026, 6, 22, 20, 0),
            time_confirmed=True,
        )
        req = _req(
            pickup_time_confirmed=False,
            scheduled_time=datetime(2026, 6, 22, 20, 0),
            scheduled_time_type="arrival",
            appointment_time_confirmed=True,
            legs=[leg],
        )
        assert has_confirmed_rdv_only(req)
        assert not has_confirmed_departure(req)

    def test_validate_without_proposed_pickup_requires_confirmed_future_departure(self):
        ref = datetime(2026, 6, 22, 12, 0)
        future = datetime(2026, 6, 22, 18, 0)
        req = _req(
            pickup_time_confirmed=True,
            scheduled_time=future,
        )
        assert can_validate_without_proposed_pickup(req, now=ref)

        past = datetime(2026, 6, 22, 10, 0)
        stale = _req(pickup_time_confirmed=True, scheduled_time=past)
        assert not can_validate_without_proposed_pickup(stale, now=ref)

    def test_is_departure_stale(self):
        past = datetime.now() - timedelta(minutes=30)
        req = _req(pickup_time_confirmed=True, scheduled_time=past)
        assert is_departure_stale(req)

    def test_validate_accept_pickup_rules_khalid_case(self):
        """Cas Khalid : RDV seul, pas de départ — Valider interdit."""
        past_rdv = datetime.now() - timedelta(hours=2)
        req = _req(
            pickup_time_confirmed=False,
            scheduled_time=past_rdv,
            scheduled_time_type="arrival",
            appointment_time_confirmed=True,
        )
        err = validate_accept_pickup_rules(req, proposed_pickup_time=None)
        assert err is not None
        assert (
            validate_accept_pickup_rules(
                req, proposed_pickup_time=datetime.now() + timedelta(minutes=15)
            )
            is None
        )
