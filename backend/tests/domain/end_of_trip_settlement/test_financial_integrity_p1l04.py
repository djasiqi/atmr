"""Preuve d'intégrité des fondations P1, sur une base déjà migrée et jetable.

Ce module n'ajoute aucune règle. Il vérifie que la caisse, l'imputation
et les avoirs tiennent ensemble.
"""

import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.getenv("P1L04_DATABASE_URL"),
    reason="P1L04_DATABASE_URL absente",
)


def _engine():
    return create_engine(os.environ["P1L04_DATABASE_URL"])


def _rollback(callback):
    engine = _engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            callback(connection)
        finally:
            transaction.rollback()


def _invoice(connection):
    return connection.execute(
        text(
            "SELECT total_amount, amount_paid, balance_due, invoice_number, "
            "refundable_chf FROM invoices WHERE id = 50"
        )
    ).one()


def test_chaine_et_protections_historiques():
    engine = _engine()
    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        collection = connection.execute(
            text(
                "SELECT kind, client_operation_key, net_chf, "
                "production_activation, legally_approved "
                "FROM cash_collection WHERE id = 1"
            )
        ).one()
        sequence = connection.execute(
            text(
                "SELECT company_id, year, month, sequence "
                "FROM invoice_sequences WHERE id = 1"
            )
        ).one()
        payment = connection.execute(
            text(
                "SELECT cash_collection_id FROM invoice_payments "
                "WHERE reference = 'historique'"
            )
        ).scalar_one()
        names = set(
            connection.execute(
                text(
                    "SELECT conname FROM pg_constraint WHERE conname IN ("
                    "'uq_cash_collection_booking_operation', "
                    "'ck_cash_collection_inactive', "
                    "'ck_cash_collection_unapproved', "
                    "'ck_cash_collection_kind_net', "
                    "'uq_invoice_credit_note_number', "
                    "'uq_invoice_credit_note_idempotency', "
                    "'chk_invoice_balance_nonneg', "
                    "'chk_invoice_paid_nonneg', "
                    "'uq_company_year_month'"
                    ")"
                )
            ).scalars()
        )
    assert version == "bde6118d9df4"
    assert collection.kind == "collection"
    assert collection.client_operation_key == "1"
    assert collection.net_chf == Decimal("20.00")
    assert collection.production_activation is False
    assert collection.legally_approved is False
    assert payment is None
    assert (sequence.company_id, sequence.year, sequence.month, sequence.sequence) == (
        1,
        2026,
        11,
        4,
    )
    assert names == {
        "uq_cash_collection_booking_operation",
        "ck_cash_collection_inactive",
        "ck_cash_collection_unapproved",
        "ck_cash_collection_kind_net",
        "uq_invoice_credit_note_number",
        "uq_invoice_credit_note_idempotency",
        "chk_invoice_balance_nonneg",
        "chk_invoice_paid_nonneg",
        "uq_company_year_month",
    }


def test_encaissements_imputation_et_avoir_ensemble():
    def _run(connection):
        before = _invoice(connection)
        connection.execute(
            text(
                "INSERT INTO cash_collection ("
                "booking_id, company_id, driver_id, debtor_label, amount_due_chf, "
                "net_chf, decision_version, kind, client_operation_key, state"
                ") VALUES "
                "(10, 1, 7, 'Essai', 45.00, 20.00, 'v1', 'collection', 'enc-a', "
                "'received'), "
                "(10, 1, 7, 'Essai', 45.00, 25.00, 'v1', 'collection', 'enc-b', "
                "'received')"
            )
        )
        first_id = connection.execute(
            text(
                "SELECT id FROM cash_collection " "WHERE client_operation_key = 'enc-a'"
            )
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO cash_collection ("
                "booking_id, company_id, driver_id, debtor_label, amount_due_chf, "
                "net_chf, decision_version, kind, client_operation_key, state, "
                "compensates_collection_id"
                ") VALUES ("
                "10, 1, 7, 'Essai', 45.00, -5.00, 'v1', 'compensation', 'comp-a', "
                "'administrative', :origin)"
            ),
            {"origin": first_id},
        )
        origin_net = connection.execute(
            text("SELECT net_chf FROM cash_collection WHERE id = :origin"),
            {"origin": first_id},
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO invoice_payments "
                "(invoice_id, amount, method, cash_collection_id) "
                "VALUES (50, 20.00, 'cash', :origin)"
            ),
            {"origin": first_id},
        )
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note ("
                "company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key"
                ") VALUES (1, 50, 'AV-1-2026-1', 45.00, 'Prestation annulee', "
                "1, 'avoir-integralite')"
            )
        )
        payments_negatifs = connection.execute(
            text("SELECT count(*) FROM invoice_payments WHERE amount < 0")
        ).scalar_one()
        after = _invoice(connection)
        assert origin_net == Decimal("20.00")
        assert payments_negatifs == 0
        assert after.total_amount == before.total_amount
        assert after.amount_paid == before.amount_paid
        assert after.balance_due == before.balance_due
        assert after.invoice_number == before.invoice_number

    _rollback(_run)


def test_rejeu_et_compensations_invalides():
    def _run(connection):
        connection.execute(
            text("INSERT INTO booking (id) VALUES (11) ON CONFLICT DO NOTHING")
        )
        connection.execute(
            text(
                "INSERT INTO cash_collection ("
                "booking_id, company_id, driver_id, debtor_label, amount_due_chf, "
                "net_chf, decision_version, kind, client_operation_key, state"
                ") VALUES (10, 1, 7, 'Essai', 45.00, 20.00, 'v1', 'collection', "
                "'enc-a', 'received')"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO cash_collection ("
                    "booking_id, company_id, driver_id, debtor_label, "
                    "amount_due_chf, net_chf, decision_version, kind, "
                    "client_operation_key, state"
                    ") VALUES (10, 1, 7, 'Essai', 45.00, 1.00, 'v1', 'collection', "
                    "'enc-a', 'received')"
                )
            )

    _rollback(_run)


def test_compensation_sans_origine_refusee():
    def _run(connection):
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO cash_collection ("
                    "booking_id, company_id, driver_id, debtor_label, "
                    "amount_due_chf, net_chf, decision_version, kind, "
                    "client_operation_key, state"
                    ") VALUES (10, 1, 7, 'Essai', 45.00, -1.00, 'v1', "
                    "'compensation', 'orpheline', 'administrative')"
                )
            )

    _rollback(_run)


def test_compensation_autre_entreprise_ou_reservation_refusee():
    def _run(connection):
        connection.execute(text("INSERT INTO booking (id) VALUES (11)"))
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO cash_collection ("
                    "booking_id, company_id, driver_id, debtor_label, "
                    "amount_due_chf, net_chf, decision_version, kind, "
                    "client_operation_key, state, compensates_collection_id"
                    ") VALUES (10, 2, 7, 'Essai', 45.00, -1.00, 'v1', "
                    "'compensation', 'autre-ent', 'administrative', 1)"
                )
            )

    _rollback(_run)


def test_compensation_autre_reservation_refusee():
    def _run(connection):
        connection.execute(text("INSERT INTO booking (id) VALUES (11)"))
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO cash_collection ("
                    "booking_id, company_id, driver_id, debtor_label, "
                    "amount_due_chf, net_chf, decision_version, kind, "
                    "client_operation_key, state, compensates_collection_id"
                    ") VALUES (11, 1, 7, 'Essai', 45.00, -1.00, 'v1', "
                    "'compensation', 'autre-res', 'administrative', 1)"
                )
            )

    _rollback(_run)


def test_seconde_imputation_et_collection_inconnue():
    def _run(connection):
        connection.execute(
            text(
                "INSERT INTO invoice_payments "
                "(invoice_id, amount, method, cash_collection_id) "
                "VALUES (50, 20.00, 'cash', 1)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_payments "
                    "(invoice_id, amount, method, cash_collection_id) "
                    "VALUES (50, 20.00, 'cash', 1)"
                )
            )

    _rollback(_run)


def test_imputation_interentreprises_et_collection_absente():
    def _run(connection):
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_payments "
                    "(invoice_id, amount, method, cash_collection_id) "
                    "VALUES (51, 20.00, 'cash', 1)"
                )
            )

    _rollback(_run)


def test_collection_absente_refusee():
    def _run(connection):
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_payments "
                    "(invoice_id, amount, method, cash_collection_id) "
                    "VALUES (50, 10.00, 'cash', 9999)"
                )
            )

    _rollback(_run)


def test_avoir_total_et_doublons_refuses():
    def _run(connection):
        before = _invoice(connection)
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note ("
                "company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key"
                ") VALUES (1, 50, 'AV-1-2026-2', 90.00, 'Annulation totale', "
                "1, 'avoir-total')"
            )
        )
        after = _invoice(connection)
        assert after.total_amount == before.total_amount
        assert after.balance_due == before.balance_due
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_credit_note ("
                    "company_id, invoice_id, credit_number, amount_chf, reason, "
                    "created_by_user_id, idempotency_key"
                    ") VALUES (1, 50, 'AV-1-2026-3', 10.00, 'Doublon cle', "
                    "1, 'avoir-total')"
                )
            )

    _rollback(_run)


def test_numero_avoir_duplique_refuse():
    def _run(connection):
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note ("
                "company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key"
                ") VALUES (1, 50, 'AV-1-2026-9', 10.00, 'Numero', 1, 'num-a')"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_credit_note ("
                    "company_id, invoice_id, credit_number, amount_chf, reason, "
                    "created_by_user_id, idempotency_key"
                    ") VALUES (1, 50, 'AV-1-2026-9', 5.00, 'Numero', 1, 'num-b')"
                )
            )

    _rollback(_run)


def test_activation_production_refusee():
    def _run(connection):
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO cash_collection ("
                    "booking_id, company_id, driver_id, debtor_label, "
                    "amount_due_chf, decision_version, kind, client_operation_key, "
                    "production_activation"
                    ") VALUES (10, 1, 7, 'Essai', 45.00, 'v1', 'collection', "
                    "'activation', true)"
                )
            )

    _rollback(_run)
