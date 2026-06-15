"""Tests extraction durée Google Directions."""

from services.geolocation.google_directions import _parse_route_metrics


def test_parse_route_metrics_sums_legs_and_traffic():
    body = {
        "routes": [
            {
                "legs": [
                    {
                        "duration": {"value": 1200},
                        "distance": {"value": 8000},
                        "duration_in_traffic": {"value": 1500},
                    },
                    {
                        "duration": {"value": 300},
                        "distance": {"value": 2000},
                    },
                ],
            }
        ]
    }

    duration, distance, traffic = _parse_route_metrics(body)

    assert duration == 1500
    assert distance == 10000
    assert traffic == 1500


def test_parse_route_metrics_empty():
    assert _parse_route_metrics({}) == (None, None, None)
