from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.demo.environment_guard import (
    DemoEnvironmentSnapshot,
    enforce_demo_environment_or_raise,
)
from services.demo.seed_spec import PROFILES, build_relative_transport_slots


def test_demo_guard_rejects_non_demo_database() -> None:
    snapshot = DemoEnvironmentSnapshot(
        app_env="demo",
        demo_mode=True,
        database_url="postgresql://user:pass@db:5432/atmr",
        redis_url="redis://redis-demo:6379/0",
        storage_bucket="lirie-demo-storage",
        storage_prefix="demo/",
    )
    with pytest.raises(RuntimeError, match="`_demo`"):
        enforce_demo_environment_or_raise(snapshot)


def test_demo_guard_accepts_valid_demo_stack() -> None:
    snapshot = DemoEnvironmentSnapshot(
        app_env="demo",
        demo_mode=True,
        database_url="postgresql://user:pass@db:5432/atmr_demo",
        redis_url="redis://redis-demo:6379/0",
        storage_bucket="lirie-demo-storage",
        storage_prefix="demo/",
    )
    enforce_demo_environment_or_raise(snapshot)


def test_relative_slots_cover_yesterday_today_tomorrow() -> None:
    reference_day = datetime.now(UTC).date()
    slots = build_relative_transport_slots(reference_day, PROFILES["sales"])
    dates = {slot[0].date() for slot in slots}
    assert reference_day in dates
    assert (reference_day - timedelta(days=1)) in dates
    assert (reference_day + timedelta(days=1)) in dates

