"""Tests alias canoniques (ex. HUG) pour l'autocomplete géographique."""

from __future__ import annotations

from routes.geocode import match_alias, serialize_alias_hit


class TestMatchAlias:
    def test_hug_short_query(self) -> None:
        hit = match_alias("HUG")
        assert hit is not None
        assert hit.get("short_name") == "HUG"

    def test_hug_in_sentence(self) -> None:
        hit = match_alias("trajet vers HUG demain")
        assert hit is not None

    def test_hopitaux_geneve(self) -> None:
        hit = match_alias("Hôpitaux Universitaires de Genève")
        assert hit is not None

    def test_unrelated_query(self) -> None:
        assert match_alias("Clinique de La Tour") is None


class TestSerializeAliasHit:
    def test_hug_full_label_and_coords(self) -> None:
        hit = match_alias("HUG")
        assert hit is not None
        serialized = serialize_alias_hit(hit)
        assert serialized["source"] == "alias"
        assert serialized["lat"] == 46.19226
        assert serialized["lon"] == 6.14262
        assert "Hôpitaux Universitaires de Genève (HUG)" in serialized["label"]
        assert "Rue Gabrielle-Perret-Gentil 4" in serialized["label"]
        assert serialized["main_text"] == "Hôpitaux Universitaires de Genève (HUG)"
        assert serialized["secondary_text"] == "Rue Gabrielle-Perret-Gentil 4, 1205 Genève"
        assert serialized["name"] == "Hôpitaux Universitaires de Genève (HUG)"
