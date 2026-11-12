"""FIN-CAISSE-4 — cash_prep ne s'ouvre jamais en production ni hors base jetable."""

from __future__ import annotations

from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from domain.end_of_trip_settlement.production_prep_barrier import prep_open
from domain.end_of_trip_settlement.simulation_barrier import recipe_database_allowed


def test_flag_produit_reste_eteint():
    assert END_OF_TRIP_COLLECTION_ENABLED is False


def test_prep_fermee_en_production_meme_avec_switch(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    monkeypatch.setenv("SETTLEMENT_LOCAL_RECIPE", "1")
    assert (
        prep_open(
            database="lirie_e2e_fin_caisse",
            host="postgres_test",
            switch="1",
            production=True,
        )
        is False
    )


def test_prep_fermee_sur_atmr_et_atmr_test(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    assert prep_open(database="atmr", host="postgres_test", switch="1") is False
    assert prep_open(database="atmr_test", host="postgres_test", switch="1") is False
    assert prep_open(database="atmrstg", host="remote.example", switch="1") is False


def test_bases_production_refusees_par_recette():
    assert recipe_database_allowed("atmr", host="db.prod.example", configured=True) is False
    assert recipe_database_allowed("atmr_test", host="localhost", configured=True) is False
    assert recipe_database_allowed("lirie_e2e_fin_caisse", host="postgres_test") is True


def test_prep_ouverte_seulement_jetable_hors_prod(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    assert (
        prep_open(
            database="lirie_e2e_fin_caisse",
            host="postgres_test",
            switch="1",
            production=False,
        )
        is True
    )
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "0")
    assert (
        prep_open(
            database="lirie_e2e_fin_caisse",
            host="postgres_test",
            switch="0",
            production=False,
        )
        is False
    )


def test_open_collection_pose_production_activation_false():
    """Les écritures ledger restent marquées non activées produit."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3]
        / "application"
        / "settlement"
        / "cash_ledger.py"
    ).read_text(encoding="utf-8")
    assert "production_activation=False" in source
    assert '"production_activation": False' in source
