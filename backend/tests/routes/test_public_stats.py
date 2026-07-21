"""Tests extraction cantons pour les statistiques publiques."""

from __future__ import annotations

from types import SimpleNamespace

from models.enums import GeoUnitType
from routes.public_stats import (
    _canton_code_from_geo_unit,
    _cantons_from_service_area,
    _swiss_canton_name_to_code,
)


class TestCantonsFromServiceArea:
    def test_json_tokens_canton_code(self) -> None:
        raw = '{"v":1,"mode":"canton","tokens":["canton:GE"]}'
        assert _cantons_from_service_area(raw, _swiss_canton_name_to_code()) == {"GE"}

    def test_json_tokens_canton_name(self) -> None:
        raw = '{"v":1,"mode":"canton","tokens":["canton_name:Genève"]}'
        assert _cantons_from_service_area(raw, _swiss_canton_name_to_code()) == {"GE"}

    def test_legacy_plain_geneve(self) -> None:
        assert _cantons_from_service_area("Geneve", _swiss_canton_name_to_code()) == {
            "GE"
        }

    def test_legacy_plain_geneve_accent(self) -> None:
        assert _cantons_from_service_area("Genève", _swiss_canton_name_to_code()) == {
            "GE"
        }

    def test_legacy_csv_multiple(self) -> None:
        raw = "canton:GE,canton:VD"
        assert _cantons_from_service_area(raw, _swiss_canton_name_to_code()) == {
            "GE",
            "VD",
        }

    def test_empty_returns_empty_set(self) -> None:
        assert _cantons_from_service_area("", _swiss_canton_name_to_code()) == set()
        assert _cantons_from_service_area(None, _swiss_canton_name_to_code()) == set()


class TestCantonCodeFromGeoUnit:
    def test_canton_unit_returns_code(self) -> None:
        unit = SimpleNamespace(
            type=GeoUnitType.CANTON,
            code="ge",
            parent=None,
        )
        unit.lineage = lambda: [unit]
        assert _canton_code_from_geo_unit(unit) == "GE"  # type: ignore[arg-type]

    def test_commune_unit_walks_to_canton(self) -> None:
        canton = SimpleNamespace(
            type=GeoUnitType.CANTON,
            code="GE",
            parent=None,
        )
        commune = SimpleNamespace(
            type=GeoUnitType.COMMUNE,
            code="6621",
            parent=canton,
        )
        commune.lineage = lambda: [commune, canton]
        assert _canton_code_from_geo_unit(commune) == "GE"  # type: ignore[arg-type]

    def test_unknown_hierarchy_returns_none(self) -> None:
        unit = SimpleNamespace(
            type=GeoUnitType.COMMUNE,
            code="x",
            parent=None,
        )
        unit.lineage = lambda: [unit]
        assert _canton_code_from_geo_unit(unit) is None  # type: ignore[arg-type]
