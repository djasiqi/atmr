"""Tests libellés timeline transport institution."""

from __future__ import annotations

from types import SimpleNamespace

from services.institutions.transport_timeline_service import (
    _summarize_changed_fields,
    build_timeline_label,
)


def _event(event_type: str, payload: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(event_type=event_type, payload=payload or {})


class TestSummarizeChangedFields:
    def test_group_itineraire_horaires(self):
        summary = _summarize_changed_fields(
            [
                "mission_date",
                "pickup_time_confirmed",
                "pickup_location",
                "multi_stop",
                "return_to_institution",
                "intermediate_stops",
            ]
        )
        assert "itinéraire" in summary
        assert "horaires" in summary

    def test_mobility_et_notes(self):
        summary = _summarize_changed_fields(["mobility", "notes"])
        assert summary == "mobilité et notes"


class TestBuildTimelineLabel:
    def test_field_updated_lisible(self):
        label = build_timeline_label(
            _event(
                "field_updated",
                {
                    "changed_fields": [
                        "mission_date",
                        "pickup_location",
                        "mobility",
                        "notes",
                    ],
                    "carrier_notified": True,
                },
            )
        )
        assert label.startswith("Demande modifiée (")
        assert "transporteur informé" in label
        assert "mission_date" not in label

    def test_request_converted_avec_transporteur(self):
        label = build_timeline_label(
            _event(
                "request_converted",
                {"company_name": "Emmenez Moi"},
            )
        )
        assert label == "Réservation confirmée — Emmenez Moi"

    def test_route_legs_reorganized_etapes(self):
        label = build_timeline_label(
            _event(
                "route_legs_reorganized",
                {"after_legs": [{}, {}, {}]},
            )
        )
        assert label == "Parcours modifié — 3 étapes"
