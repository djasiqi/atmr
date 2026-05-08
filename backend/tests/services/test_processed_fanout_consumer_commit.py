"""Sanity checks — commit manuel consumer processed fanout (kafka-python)."""

from __future__ import annotations

import pytest

kafka = pytest.importorskip("kafka")


def test_offset_and_metadata_for_manual_commit():
    from kafka.structs import OffsetAndMetadata

    m = OffsetAndMetadata(42, "", -1)
    assert m.offset == 42
