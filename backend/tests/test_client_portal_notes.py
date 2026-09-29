"""Tests unitaires pour l'assemblage des notes portail client."""

from __future__ import annotations

from shared.client_portal_notes import (
    NOTES_MEDICAL_ASSEMBLED_PORTAL_MAX_LENGTH,
    compose_client_portal_notes_medical,
)


def test_compose_arrival_and_asap_lines() -> None:
    arrival = compose_client_portal_notes_medical({"scheduled_time_type": "arrival"})
    assert arrival is not None
    assert arrival.startswith("Horaire souhaité : rendez-vous à destination.")

    asap = compose_client_portal_notes_medical({"asap": True, "scheduled_time_type": "arrival"})
    assert asap == "Horaire souhaité : dès que possible."

    assert compose_client_portal_notes_medical({"scheduled_time_type": "departure"}) is None
    assert compose_client_portal_notes_medical({}) is None
    assert compose_client_portal_notes_medical({"client_note": "  "}) is None


def test_compose_client_note_only() -> None:
    out = compose_client_portal_notes_medical({"client_note": "  Aller doux  "})
    assert out == "Aller doux"


def test_compose_occurrences_prepends() -> None:
    out = compose_client_portal_notes_medical(
        {"occurrences": 3, "client_note": "Patient PMR"}
    )
    assert out is not None
    assert out.startswith("Occurrences demandées (même trajet) : 3")
    assert out.endswith("Patient PMR")


def test_compose_truncates_client_note_preserves_meta() -> None:
    meta = "Occurrences demandées (même trajet) : 2"
    room = NOTES_MEDICAL_ASSEMBLED_PORTAL_MAX_LENGTH - len(meta) - 1
    long_note = "N" * (room + 80)
    out = compose_client_portal_notes_medical(
        {"occurrences": 2, "client_note": long_note}
    )
    assert out is not None
    assert out.startswith(meta)
    assert "NNN" in out
    assert out.endswith("…")
    assert len(out) <= NOTES_MEDICAL_ASSEMBLED_PORTAL_MAX_LENGTH
