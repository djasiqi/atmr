"""Tests recherche zones (canton / autocomplete service_area)."""

from __future__ import annotations

from routes.geocode import (
    _db_zone_search_terms,
    _extract_zone_type_from_geoadmin,
    _parse_canton_search_intent,
    _should_skip_geoadmin_for_canton_query,
)


class TestParseCantonSearchIntent:
    def test_canton_ge_code(self) -> None:
        assert _parse_canton_search_intent("Canton GE") == "GE"

    def test_canton_de_geneve(self) -> None:
        assert _parse_canton_search_intent("Canton de Genève") == "GE"

    def test_geneve_name(self) -> None:
        assert _parse_canton_search_intent("Genève") == "GE"

    def test_two_letter_code(self) -> None:
        assert _parse_canton_search_intent("VD") == "VD"

    def test_lausanne_not_canton(self) -> None:
        assert _parse_canton_search_intent("Lausanne") is None


class TestDbZoneSearchTerms:
    def test_strips_canton_prefix(self) -> None:
        terms = _db_zone_search_terms("Canton GE")
        assert "canton ge" in terms
        assert "ge" in terms

    def test_geneve_includes_code(self) -> None:
        terms = _db_zone_search_terms("Genève")
        assert "geneve" in terms
        assert "ge" in terms


class TestGeoadminZoneTypeExtraction:
    def test_kantone_layer_is_canton(self) -> None:
        assert _extract_zone_type_from_geoadmin({"origin": "kantone.gg25"}) == "canton"

    def test_poi_with_canton_detail_is_not_canton(self) -> None:
        assert (
            _extract_zone_type_from_geoadmin(
                {
                    "origin": "location.search",
                    "detail": "Canton, TI",
                    "label": "Lieu Mugena (TI) - Alto Malcantone",
                }
            )
            is None
        )

    def test_gg25_is_commune(self) -> None:
        assert _extract_zone_type_from_geoadmin({"origin": "gg25"}) == "commune"


class TestSkipGeoadminForCantonQuery:
    def test_skip_for_canton_ge(self) -> None:
        assert _should_skip_geoadmin_for_canton_query("Canton GE") is True

    def test_skip_for_code_only(self) -> None:
        assert _should_skip_geoadmin_for_canton_query("GE") is True

    def test_no_skip_for_city(self) -> None:
        assert _should_skip_geoadmin_for_canton_query("Lausanne") is False
