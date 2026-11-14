"""CONVENTION-REGLEMENT-02 — persistance sur PostgreSQL jetable.

Nécessite CR02_DATABASE_URL (ex. postgresql+psycopg://…/cr02_test).
Crée uniquement les tables du lot (sans FKs métier) pour vérifier
contraintes, idempotence et séparation transport/règlement.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.getenv("CR02_DATABASE_URL"),
    reason="CR02_DATABASE_URL absente",
)


DDL = """
CREATE TABLE IF NOT EXISTS settlement_non_payment_incident (
  id SERIAL PRIMARY KEY,
  public_id VARCHAR(36) NOT NULL UNIQUE,
  idempotency_key VARCHAR(80) NOT NULL,
  booking_id INTEGER NOT NULL,
  company_id INTEGER NOT NULL,
  driver_id INTEGER NOT NULL,
  client_id INTEGER,
  debtor_id INTEGER,
  debtor_label VARCHAR(160),
  amount_due_chf NUMERIC(10,2) NOT NULL,
  currency VARCHAR(3) NOT NULL DEFAULT 'CHF',
  transport_date VARCHAR(32) NOT NULL,
  motif VARCHAR(64) NOT NULL,
  observation TEXT NOT NULL DEFAULT '',
  status VARCHAR(32) NOT NULL DEFAULT 'a_traiter',
  payment_recorded BOOLEAN NOT NULL DEFAULT false,
  signature_recorded BOOLEAN NOT NULL DEFAULT false,
  deferred_billing_authorized BOOLEAN NOT NULL DEFAULT false,
  mission_completed BOOLEAN NOT NULL DEFAULT true,
  mission_close_allowed BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_snp_incident_company_key UNIQUE (company_id, idempotency_key),
  CONSTRAINT ck_snp_incident_status CHECK (
    status IN ('a_traiter','en_cours_examen','regularise','cloture_justifie')
  ),
  CONSTRAINT ck_snp_incident_motif CHECK (
    motif IN (
      'refus_signer_et_payer','impossibilite_regler','contestation_montant',
      'moyen_paiement_indisponible_ou_echoue','autre'
    )
  ),
  CONSTRAINT ck_snp_incident_currency CHECK (currency = 'CHF'),
  CONSTRAINT ck_snp_incident_amount CHECK (amount_due_chf >= 0),
  CONSTRAINT ck_snp_incident_not_payment CHECK (payment_recorded = false),
  CONSTRAINT ck_snp_incident_not_signature CHECK (signature_recorded = false),
  CONSTRAINT ck_snp_incident_not_deferred CHECK (deferred_billing_authorized = false),
  CONSTRAINT ck_snp_incident_mission_completed CHECK (mission_completed = true)
);

CREATE TABLE IF NOT EXISTS settlement_non_payment_event (
  id SERIAL PRIMARY KEY,
  incident_id INTEGER NOT NULL REFERENCES settlement_non_payment_incident(id) ON DELETE RESTRICT,
  idempotency_key VARCHAR(80) NOT NULL,
  actor_role VARCHAR(32) NOT NULL,
  actor_id INTEGER,
  action VARCHAR(64) NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_snp_event_incident_key UNIQUE (incident_id, idempotency_key),
  CONSTRAINT ck_snp_event_key CHECK (char_length(idempotency_key) > 0)
);

CREATE TABLE IF NOT EXISTS external_card_settlement (
  id SERIAL PRIMARY KEY,
  booking_id INTEGER NOT NULL,
  company_id INTEGER NOT NULL,
  driver_id INTEGER NOT NULL,
  amount_chf NUMERIC(10,2) NOT NULL,
  currency VARCHAR(3) NOT NULL DEFAULT 'CHF',
  channel VARCHAR(32) NOT NULL DEFAULT 'external_terminal',
  tap_to_pay BOOLEAN NOT NULL DEFAULT false,
  provider_name VARCHAR(64) NOT NULL DEFAULT 'external_bank_terminal',
  reference VARCHAR(120) NOT NULL DEFAULT '',
  receipt_note TEXT NOT NULL DEFAULT '',
  confirmed_by_driver BOOLEAN NOT NULL DEFAULT true,
  idempotency_key VARCHAR(80) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_ext_card_company_key UNIQUE (company_id, idempotency_key),
  CONSTRAINT ck_ext_card_currency CHECK (currency = 'CHF'),
  CONSTRAINT ck_ext_card_amount CHECK (amount_chf > 0),
  CONSTRAINT ck_ext_card_channel CHECK (channel = 'external_terminal'),
  CONSTRAINT ck_ext_card_not_tap CHECK (tap_to_pay = false),
  CONSTRAINT ck_ext_card_provider CHECK (provider_name = 'external_bank_terminal'),
  CONSTRAINT ck_ext_card_proof CHECK (
    char_length(reference) > 0 OR char_length(receipt_note) > 0
  )
);
"""


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(os.environ["CR02_DATABASE_URL"])
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS settlement_non_payment_event CASCADE"))
        conn.execute(
            text("DROP TABLE IF EXISTS settlement_non_payment_incident CASCADE")
        )
        conn.execute(text("DROP TABLE IF EXISTS external_card_settlement CASCADE"))
        for statement in DDL.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    yield eng
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS settlement_non_payment_event CASCADE"))
        conn.execute(
            text("DROP TABLE IF EXISTS settlement_non_payment_incident CASCADE")
        )
        conn.execute(text("DROP TABLE IF EXISTS external_card_settlement CASCADE"))


def test_incident_persiste_sans_paiement_ni_differe(engine):
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO settlement_non_payment_incident (
                  public_id, idempotency_key, booking_id, company_id, driver_id,
                  amount_due_chf, transport_date, motif, observation
                ) VALUES (
                  '11111111-1111-1111-1111-111111111111', 'k1', 100, 7, 3,
                  45.00, '2026-11-14', 'refus_signer_et_payer', 'Constat'
                ) RETURNING id, payment_recorded, deferred_billing_authorized,
                  mission_completed, status
                """
            )
        ).one()
        assert row.payment_recorded is False
        assert row.deferred_billing_authorized is False
        assert row.mission_completed is True
        assert row.status == "a_traiter"
        conn.execute(
            text(
                """
                INSERT INTO settlement_non_payment_event (
                  incident_id, idempotency_key, actor_role, actor_id, action, detail
                ) VALUES (:iid, 'open:k1', 'driver', 3, 'opened', 'refus_signer_et_payer')
                """
            ),
            {"iid": row.id},
        )
        count = conn.execute(
            text(
                "SELECT count(*) FROM settlement_non_payment_event WHERE incident_id = :iid"
            ),
            {"iid": row.id},
        ).scalar_one()
        assert count == 1


def test_idempotence_incident_par_entreprise(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO settlement_non_payment_incident (
                  public_id, idempotency_key, booking_id, company_id, driver_id,
                  amount_due_chf, transport_date, motif
                ) VALUES (
                  '22222222-2222-2222-2222-222222222222', 'same', 101, 7, 3,
                  10.00, '2026-11-14', 'impossibilite_regler'
                )
                """
            )
        )
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    """
                    INSERT INTO settlement_non_payment_incident (
                      public_id, idempotency_key, booking_id, company_id, driver_id,
                      amount_due_chf, transport_date, motif
                    ) VALUES (
                      '33333333-3333-3333-3333-333333333333', 'same', 102, 7, 3,
                      10.00, '2026-11-14', 'impossibilite_regler'
                    )
                    """
                )
            )


def test_refuse_marquer_paiement_sur_incident(engine):
    with engine.begin() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                """
                    INSERT INTO settlement_non_payment_incident (
                      public_id, idempotency_key, booking_id, company_id, driver_id,
                      amount_due_chf, transport_date, motif, payment_recorded
                    ) VALUES (
                      '44444444-4444-4444-4444-444444444444', 'pay', 103, 7, 3,
                      10.00, '2026-11-14', 'contestation_montant', true
                    )
                    """
            )
        )


def test_carte_externe_avec_reference(engine):
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO external_card_settlement (
                  booking_id, company_id, driver_id, amount_chf,
                  reference, idempotency_key
                ) VALUES (200, 7, 3, 45.00, 'TERM-9988', 'card-1')
                RETURNING tap_to_pay, channel, provider_name, amount_chf
                """
            )
        ).one()
        assert row.tap_to_pay is False
        assert row.channel == "external_terminal"
        assert row.provider_name == "external_bank_terminal"
        assert row.amount_chf == Decimal("45.00")


def test_carte_sans_preuve_refusee(engine):
    with engine.begin() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                """
                    INSERT INTO external_card_settlement (
                      booking_id, company_id, driver_id, amount_chf,
                      reference, receipt_note, idempotency_key
                    ) VALUES (201, 7, 3, 20.00, '', '', 'card-empty')
                    """
            )
        )


def test_tap_to_pay_interdit_sur_table_externe(engine):
    with engine.begin() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                """
                    INSERT INTO external_card_settlement (
                      booking_id, company_id, driver_id, amount_chf,
                      reference, idempotency_key, tap_to_pay
                    ) VALUES (202, 7, 3, 20.00, 'X', 'card-tap', true)
                    """
            )
        )
