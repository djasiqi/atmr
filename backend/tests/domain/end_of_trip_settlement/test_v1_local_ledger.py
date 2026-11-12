"""Le registre V1 local ne s'ouvre que sur la base de développement."""

from __future__ import annotations

from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from domain.end_of_trip_settlement.v1_local_ledger import v1_local_ledger_enabled


def test_flag_produit_reste_faux():
    assert END_OF_TRIP_COLLECTION_ENABLED is False


def test_seule_base_locale_atmr_est_ouverte():
    assert (
        v1_local_ledger_enabled(
            production=True, database="atmr", host="pgbouncer", switch="1"
        )
        is False
    )
    assert (
        v1_local_ledger_enabled(
            production=False, database="atmr_test", host="postgres", switch="1"
        )
        is False
    )
    assert (
        v1_local_ledger_enabled(
            production=False, database="atmr", host="db.remote", switch="1"
        )
        is False
    )
    assert (
        v1_local_ledger_enabled(
            production=False, database="atmrstg", host="pgbouncer", switch="1"
        )
        is False
    )
    assert (
        v1_local_ledger_enabled(
            production=False, database="lirie_e2e_demo", host="postgres", switch="1"
        )
        is False
    )
    assert (
        v1_local_ledger_enabled(
            production=False, database="atmr", host="pgbouncer", switch="0"
        )
        is False
    )
    assert (
        v1_local_ledger_enabled(
            production=False, database="atmr", host="pgbouncer", switch="1"
        )
        is True
    )
    assert (
        v1_local_ledger_enabled(
            production=False, database="atmr", host="127.0.0.1", switch="1"
        )
        is True
    )
