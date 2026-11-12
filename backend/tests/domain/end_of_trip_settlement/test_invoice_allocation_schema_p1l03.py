"""Contraintes PostgreSQL de l'imputation et des avoirs. Base jetable seulement."""

import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.getenv("P1L03_DATABASE_URL"),
    reason="P1L03_DATABASE_URL absente",
)


def _engine():
    return create_engine(os.environ["P1L03_DATABASE_URL"])


def _rollback(callback):
    engine = _engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            callback(connection)
        finally:
            transaction.rollback()


def test_paiement_lie_a_une_collection():
    def _run(connection):
        connection.execute(
            text(
                "INSERT INTO invoice_payments "
                "(invoice_id, amount, method, cash_collection_id) "
                "VALUES (50, 20.00, 'cash', 1)"
            )
        )
        linked = connection.execute(
            text(
                "SELECT cash_collection_id FROM invoice_payments "
                "WHERE cash_collection_id = 1"
            )
        ).scalar_one()
        assert linked == 1

    _rollback(_run)


def test_double_imputation_refusee():
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


def test_paiement_historique_sans_collection():
    engine = _engine()
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT reference, cash_collection_id FROM invoice_payments "
                "WHERE reference = 'historique'"
            )
        ).one()
    assert row.reference == "historique"
    assert row.cash_collection_id is None


def test_collection_inexistante_refusee():
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


def test_imputation_autre_entreprise_refusee():
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


def test_avoir_partiel_sur_facture_impayee():
    def _run(connection):
        before = _invoice(connection, 50)
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note "
                "(company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key) "
                "VALUES (1, 50, 'AV-1-2026-1', 45.00, 'Prestation annulee', "
                "1, 'avoir-partiel')"
            )
        )
        after = _invoice(connection, 50)
        payments = connection.execute(
            text("SELECT count(*) FROM invoice_payments WHERE amount < 0")
        ).scalar_one()
        assert before == after
        assert payments == 0

    _rollback(_run)


def test_avoir_total_sur_facture_impayee():
    def _run(connection):
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note "
                "(company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key) "
                "VALUES (1, 50, 'AV-1-2026-2', 90.00, 'Annulation totale', "
                "1, 'avoir-total')"
            )
        )
        note = connection.execute(
            text(
                "SELECT amount_chf, status FROM invoice_credit_note "
                "WHERE idempotency_key = 'avoir-total'"
            )
        ).one()
        assert note.amount_chf == Decimal("90.00")
        assert note.status == "issued"

    _rollback(_run)


def test_avoir_sans_paiement_negatif():
    def _run(connection):
        before = connection.execute(
            text("SELECT count(*) FROM invoice_payments")
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note "
                "(company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key) "
                "VALUES (1, 50, 'AV-1-2026-3', 10.00, 'Ajustement', "
                "1, 'avoir-sans-paiement')"
            )
        )
        after = connection.execute(
            text("SELECT count(*) FROM invoice_payments")
        ).scalar_one()
        assert after == before

    _rollback(_run)


def test_idempotence_avoir_refusee():
    def _run(connection):
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note "
                "(company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key) "
                "VALUES (1, 50, 'AV-1-2026-4', 10.00, 'Premier', "
                "1, 'meme-cle')"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_credit_note "
                    "(company_id, invoice_id, credit_number, amount_chf, reason, "
                    "created_by_user_id, idempotency_key) "
                    "VALUES (1, 50, 'AV-1-2026-5', 10.00, 'Second', "
                    "1, 'meme-cle')"
                )
            )

    _rollback(_run)


def test_numero_avoir_duplique_refuse():
    def _run(connection):
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note "
                "(company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key) "
                "VALUES (1, 50, 'AV-1-2026-8', 10.00, 'Premier numero', "
                "1, 'numero-a')"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_credit_note "
                    "(company_id, invoice_id, credit_number, amount_chf, reason, "
                    "created_by_user_id, idempotency_key) "
                    "VALUES (1, 50, 'AV-1-2026-8', 5.00, 'Second numero', "
                    "1, 'numero-b')"
                )
            )

    _rollback(_run)


def test_montant_avoir_nul_refuse():
    def _run(connection):
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO invoice_credit_note "
                    "(company_id, invoice_id, credit_number, amount_chf, reason, "
                    "created_by_user_id, idempotency_key) "
                    "VALUES (1, 50, 'AV-1-2026-9', 0.00, 'Nul', 1, 'montant-nul')"
                )
            )

    _rollback(_run)


def test_sequence_factures_inchangee():
    engine = _engine()
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT company_id, year, month, sequence "
                "FROM invoice_sequences WHERE id = 1"
            )
        ).one()
    assert (row.company_id, row.year, row.month, row.sequence) == (1, 2026, 11, 4)


def test_facture_historique_compatible():
    engine = _engine()
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT invoice_number, total_amount, amount_paid, balance_due, "
                "refundable_chf FROM invoices WHERE id = 50"
            )
        ).one()
    assert row.invoice_number == "F-1-2026-1"
    assert row.total_amount == Decimal("90.00")
    assert row.amount_paid == Decimal("0.00")
    assert row.balance_due == Decimal("90.00")
    assert row.refundable_chf == Decimal("0")


def test_facture_originale_non_modifiee_par_l_avoir():
    def _run(connection):
        before = _invoice(connection, 50)
        connection.execute(
            text(
                "INSERT INTO invoice_credit_note "
                "(company_id, invoice_id, credit_number, amount_chf, reason, "
                "created_by_user_id, idempotency_key) "
                "VALUES (1, 50, 'AV-1-2026-6', 45.00, 'Sans toucher la facture', "
                "1, 'lecture-seule')"
            )
        )
        assert _invoice(connection, 50) == before

    _rollback(_run)


def test_contraintes_presentes_et_chaine():
    engine = _engine()
    with engine.connect() as connection:
        names = set(
            connection.execute(
                text(
                    "SELECT conname FROM pg_constraint WHERE conrelid IN "
                    "('invoice_payments'::regclass, 'invoice_credit_note'::regclass, "
                    "'invoices'::regclass, 'invoice_sequences'::regclass)"
                )
            ).scalars()
        )
        indexes = set(
            connection.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'invoice_payments'"
                )
            ).scalars()
        )
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        trigger = connection.execute(
            text(
                "SELECT tgname FROM pg_trigger "
                "WHERE tgname = 'trg_invoice_payment_same_company'"
            )
        ).scalar_one()
    assert "uq_invoice_credit_note_idempotency" in names
    assert "uq_invoice_credit_note_number" in names
    assert "chk_invoice_balance_nonneg" in names
    assert "chk_invoice_paid_nonneg" in names
    assert "uq_company_year_month" in names
    assert "uq_invoice_payments_cash_collection" in indexes
    assert trigger == "trg_invoice_payment_same_company"
    assert version == "bde6118d9df4"


def _invoice(connection, invoice_id):
    row = connection.execute(
        text(
            "SELECT total_amount, amount_paid, balance_due, invoice_number "
            "FROM invoices WHERE id = :invoice_id"
        ),
        {"invoice_id": invoice_id},
    ).one()
    return (
        row.total_amount,
        row.amount_paid,
        row.balance_due,
        row.invoice_number,
    )
