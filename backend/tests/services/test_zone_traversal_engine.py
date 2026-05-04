from services.pricing import zone_traversal_engine as engine


def test_compute_zone_traversal_blocks_without_zone_set():
    result = engine.compute_zone_traversal(zone_set_key=None, route_geometry=None)
    assert result.confidence == "blocked"
    assert "zone_set_missing" in result.blocking_reasons


def test_compute_zone_traversal_blocks_without_geometry():
    result = engine.compute_zone_traversal(
        zone_set_key="zoneset_ge_v1", route_geometry=None
    )
    assert result.confidence == "blocked"
    assert "route_geometry_missing" in result.blocking_reasons


def test_compute_zone_traversal_blocks_when_geom_column_unavailable(monkeypatch):
    monkeypatch.setattr(engine, "_has_geo_unit_geom_column", lambda: False)
    result = engine.compute_zone_traversal(
        zone_set_key="zoneset_ge_v1",
        route_geometry={
            "type": "LineString",
            "coordinates": [[6.1, 46.2], [6.2, 46.3]],
        },
    )
    assert result.confidence == "blocked"
    assert "zone_geometry_unavailable" in result.blocking_reasons
