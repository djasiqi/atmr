"""Tests validation service_area (JSON V1)."""

from __future__ import annotations

import json

import pytest

from routes.company_settings import (
    SERVICE_AREA_JSON_VERSION,
    _canonicalize_service_area_token,
    _validate_service_area_value,
)


class TestCanonicalizeServiceAreaToken:
    def test_canonical_unchanged(self) -> None:
        assert _canonicalize_service_area_token("canton:GE") == "canton:GE"

    def test_canton_name_geneve(self) -> None:
        assert _canonicalize_service_area_token("canton_name:Genève") == "canton:GE"

    def test_unknown_token_returns_none(self) -> None:
        assert _canonicalize_service_area_token("invalid") is None


class TestValidateServiceAreaValue:
    def test_accepts_canonical_canton_json(self) -> None:
        raw = json.dumps(
            {"v": SERVICE_AREA_JSON_VERSION, "mode": "canton", "tokens": ["canton:GE"]},
            ensure_ascii=False,
        )
        result = _validate_service_area_value(raw)
        parsed = json.loads(result)
        assert parsed["tokens"] == ["canton:GE"]

    def test_normalizes_canton_name_token(self) -> None:
        raw = json.dumps(
            {
                "v": SERVICE_AREA_JSON_VERSION,
                "mode": "canton",
                "tokens": ["canton_name:Genève"],
            },
            ensure_ascii=False,
        )
        result = _validate_service_area_value(raw)
        parsed = json.loads(result)
        assert parsed["tokens"] == ["canton:GE"]

    def test_rejects_non_canonical_token(self) -> None:
        raw = json.dumps(
            {
                "v": SERVICE_AREA_JSON_VERSION,
                "mode": "canton",
                "tokens": ["canton:Genève"],
            },
            ensure_ascii=False,
        )
        with pytest.raises(ValueError, match="token non canonique"):
            _validate_service_area_value(raw)
