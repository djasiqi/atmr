"""Tests unitaires : normalisation descriptions transport facture."""

from __future__ import annotations

from shared.utils.transport_description_normalize import normalize_transport_line_description


def test_trajet_dup_space_ride_only():
    assert (
        normalize_transport_line_description(
            "Trajet : Trajet Chemin des Courbes 9",
            kind="ride",
        )
        == "Trajet : Chemin des Courbes 9"
    )


def test_trajet_dup_no_space_before_chemin():
    assert (
        normalize_transport_line_description(
            "Trajet : TrajetChemin des Courbes 9",
            kind="ride",
        )
        == "Trajet : Chemin des Courbes 9"
    )


def test_trajet_dup_nbsp_entities():
    assert (
        normalize_transport_line_description(
            "Trajet&nbsp;:&nbsp;Trajet Chemin des Courbes 9",
            kind="ride",
        )
        == "Trajet : Chemin des Courbes 9"
    )


def test_trajet_dup_fullwidth_colon():
    assert (
        normalize_transport_line_description(
            "Trajet ： Trajet Rue X",
            kind="ride",
        )
        == "Trajet : Rue X"
    )


def test_livraison_dash_dup_material_only():
    assert (
        normalize_transport_line_description(
            "Livraison – Livraison Matériels divers",
            kind="material_delivery",
        )
        == "Livraison – Matériels divers"
    )


def test_livraison_colon_dup():
    assert (
        normalize_transport_line_description(
            "Livraison : Livraison Colis A",
            kind="material_delivery",
        )
        == "Livraison : Colis A"
    )


def test_idempotent_and_no_false_positive():
    s = "Trajet : Rue de la Gare 1 → Hôpital"
    assert normalize_transport_line_description(s, kind="ride") == s


def test_triple_trajet_collapses():
    assert (
        normalize_transport_line_description(
            "Trajet : Trajet : Trajet Chemin 1",
            kind="ride",
        )
        == "Trajet : Chemin 1"
    )


def test_ride_does_not_touch_livraison_dup():
    """Une ligne course ne doit pas modifier les motifs Livraison."""
    raw = "Livraison – Livraison Colis"
    assert normalize_transport_line_description(raw, kind="ride") == raw


def test_material_does_not_touch_trajet_dup():
    """Une livraison matériel ne doit pas modifier les motifs Trajet."""
    raw = "Trajet : Trajet Rue des Fleurs 1"
    assert (
        normalize_transport_line_description(raw, kind="material_delivery") == raw
    )
