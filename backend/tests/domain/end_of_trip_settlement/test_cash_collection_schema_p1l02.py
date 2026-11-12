"""Contraintes PostgreSQL de la caisse multi-lignes. Base jetable seulement."""

import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.getenv("P1L02_DATABASE_URL"),
    reason="P1L02_DATABASE_URL absente",
)


def _engine():
    return create_engine(os.environ["P1L02_DATABASE_URL"])


def _insert(connection, **values):
    columns = ", ".join(values)
    params = {key: values[key] for key in values}
    names = ", ".join(f":{key}" for key in values)
    connection.execute(
        text(f"INSERT INTO cash_collection ({columns}) VALUES ({names})"),
        params,
    )


def _base(**overrides):
    values = {
        "booking_id": 10,
        "company_id": 1,
        "driver_id": 7,
        "debtor_label": "Essai",
        "amount_due_chf": "45.00",
        "decision_version": "v1",
        "kind": "collection",
        "client_operation_key": "op-nouvelle",
        "state": "received",
        "net_chf": "20.00",
    }
    values.update(overrides)
    return values


def test_historique_conserve_la_cle_egale_a_l_id():
    engine = _engine()
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT id, booking_id, company_id, kind, client_operation_key, "
                "net_chf, compensates_collection_id, production_activation, "
                "legally_approved FROM cash_collection WHERE id = 1"
            )
        ).one()
    assert row.id == 1
    assert row.booking_id == 10
    assert row.company_id == 1
    assert row.kind == "collection"
    assert row.client_operation_key == "1"
    assert row.net_chf == Decimal("20.00")
    assert row.compensates_collection_id is None
    assert row.production_activation is False
    assert row.legally_approved is False


def test_deux_encaissements_positifs():
    def _run(connection):
        _insert(connection, **_base(client_operation_key="enc-1", net_chf="20.00"))
        _insert(connection, **_base(client_operation_key="enc-2", net_chf="25.00"))
        count = connection.execute(
            text(
                "SELECT count(*) FROM cash_collection "
                "WHERE booking_id = 10 AND kind = 'collection'"
            )
        ).scalar_one()
        assert count >= 3

    _rollback(_run)


def _rollback(callback):
    engine = _engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            callback(connection)
        finally:
            transaction.rollback()


def test_rejeu_de_la_meme_cle_refuse():
    def _run(connection):
        _insert(connection, **_base(client_operation_key="enc-1", net_chf="20.00"))
        with pytest.raises(IntegrityError):
            _insert(
                connection, **_base(client_operation_key="enc-1", net_chf="1.00")
            )

    _rollback(_run)


def test_compensation_liee_acceptee():
    def _run(connection):
        _insert(
            connection,
            **_base(
                kind="compensation",
                client_operation_key="comp-1",
                compensates_collection_id=1,
                net_chf="-5.00",
                state="administrative",
            ),
        )
        origin = connection.execute(
            text("SELECT net_chf FROM cash_collection WHERE id = 1")
        ).scalar_one()
        assert origin == Decimal("20.00")

    _rollback(_run)


def test_compensation_sans_origine_refusee():
    def _run(connection):
        with pytest.raises(IntegrityError):
            _insert(
                connection,
                **_base(
                    kind="compensation",
                    client_operation_key="comp-orpheline",
                    net_chf="-1.00",
                ),
            )

    _rollback(_run)


def test_net_negatif_ordinaire_refuse():
    def _run(connection):
        with pytest.raises(IntegrityError):
            _insert(
                connection,
                **_base(client_operation_key="negatif", net_chf="-1.00"),
            )

    _rollback(_run)


def test_compensation_autre_entreprise_refusee():
    def _run(connection):
        with pytest.raises(IntegrityError):
            _insert(
                connection,
                **_base(
                    company_id=2,
                    kind="compensation",
                    client_operation_key="comp-autre",
                    compensates_collection_id=1,
                    net_chf="-1.00",
                ),
            )

    _rollback(_run)


def test_activation_production_refusee():
    def _run(connection):
        with pytest.raises(IntegrityError):
            _insert(
                connection,
                **_base(
                    client_operation_key="activation",
                    production_activation=True,
                ),
            )

    _rollback(_run)


def test_approbation_juridique_refusee():
    def _run(connection):
        with pytest.raises(IntegrityError):
            _insert(
                connection,
                **_base(
                    client_operation_key="juridique",
                    legally_approved=True,
                ),
            )

    _rollback(_run)


def test_unicite_booking_historique_levee():
    engine = _engine()
    with engine.connect() as connection:
        names = connection.execute(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = 'cash_collection'::regclass"
            )
        ).scalars().all()
    assert "uq_cash_collection_booking" not in names
    assert "uq_cash_collection_booking_operation" in names
    assert "ck_cash_collection_inactive" in names
    assert "ck_cash_collection_unapproved" in names
    assert "ck_cash_collection_kind_net" in names
