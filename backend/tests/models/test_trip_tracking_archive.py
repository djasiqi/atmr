"""Tests TripTrackingArchive — partitionnement optionnel."""

from __future__ import annotations

from unittest.mock import MagicMock

from models.trip_tracking_archive import TripTrackingArchive


def test_is_parent_partitioned_false_for_regular_table():
    session = MagicMock()
    session.execute.return_value.scalar.return_value = False
    assert TripTrackingArchive.is_parent_partitioned(session) is False


def test_is_parent_partitioned_true_for_partitioned_parent():
    session = MagicMock()
    session.execute.return_value.scalar.return_value = True
    assert TripTrackingArchive.is_parent_partitioned(session) is True


def test_ensure_partition_skipped_when_table_not_partitioned():
    session = MagicMock()
    session.execute.return_value.scalar.return_value = False
    created = TripTrackingArchive.ensure_partition_for_month(2026, 6, session)
    assert created is False
    assert session.commit.call_count == 0


def test_ensure_partition_creates_when_parent_partitioned():
    session = MagicMock()
    # 1er scalar: is_parent_partitioned → True
    # 2e scalar: partition exists → False
    session.execute.return_value.scalar.side_effect = [True, False]
    created = TripTrackingArchive.ensure_partition_for_month(2026, 6, session)
    assert created is True
    session.commit.assert_called_once()
