"""Gates Phase 3 : défauts earliest / no seek / enriched (sans importer FastAPI)."""

from __future__ import annotations

import os


def _compute_defaults(mode: str) -> dict[str, object]:
    is_primary = mode in ("kafka_primary", "kafka_primary_canary")
    return {
        "is_primary": is_primary,
        "auto_offset_reset": os.getenv(
            "WS_KAFKA_AUTO_OFFSET_RESET",
            "earliest" if is_primary else "latest",
        ),
        "seek_to_end": (
            os.getenv(
                "WS_KAFKA_SEEK_TO_END_ON_START",
                "false" if is_primary else "true",
            ).lower()
            == "true"
        ),
        "enriched": (
            os.getenv(
                "WS_KAFKA_ENABLE_ENRICHED",
                "true" if is_primary else "false",
            ).lower()
            == "true"
        ),
        "relay_disabled": is_primary,
    }


def test_kafka_primary_defaults(monkeypatch):
    monkeypatch.delenv("WS_KAFKA_AUTO_OFFSET_RESET", raising=False)
    monkeypatch.delenv("WS_KAFKA_SEEK_TO_END_ON_START", raising=False)
    monkeypatch.delenv("WS_KAFKA_ENABLE_ENRICHED", raising=False)
    d = _compute_defaults("kafka_primary")
    assert d["is_primary"] is True
    assert d["auto_offset_reset"] == "earliest"
    assert d["seek_to_end"] is False
    assert d["enriched"] is True
    assert d["relay_disabled"] is True


def test_legacy_defaults(monkeypatch):
    monkeypatch.delenv("WS_KAFKA_AUTO_OFFSET_RESET", raising=False)
    monkeypatch.delenv("WS_KAFKA_SEEK_TO_END_ON_START", raising=False)
    monkeypatch.delenv("WS_KAFKA_ENABLE_ENRICHED", raising=False)
    d = _compute_defaults("legacy")
    assert d["is_primary"] is False
    assert d["auto_offset_reset"] == "latest"
    assert d["seek_to_end"] is True
    assert d["enriched"] is False
    assert d["relay_disabled"] is False
