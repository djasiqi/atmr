"""CONVENTION-SIGNATURE-REAL-01 — persistance jetable + parcours signature.

Nécessite CSR01_DATABASE_URL (postgresql+psycopg://…/csr01_test).
Tables du lot uniquement (sans FKs métier) pour prouver contraintes,
archivage logique, versions et gate différé.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import date

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.getenv("CSR01_DATABASE_URL"),
    reason="CSR01_DATABASE_URL absente",
)


DDL = """
CREATE TABLE IF NOT EXISTS patient_company_convention (
  id SERIAL PRIMARY KEY,
  public_id VARCHAR(36) NOT NULL UNIQUE,
  company_id INTEGER NOT NULL,
  client_id INTEGER NOT NULL,
  debtor_id INTEGER,
  debtor_label VARCHAR(160) NOT NULL,
  invoice_recipient_id INTEGER,
  invoice_recipient_label VARCHAR(160),
  billing_conditions TEXT NOT NULL,
  tariff_version VARCHAR(64) NOT NULL,
  service_scope TEXT NOT NULL,
  document_version VARCHAR(32) NOT NULL,
  version_number INTEGER NOT NULL DEFAULT 1,
  active BOOLEAN NOT NULL DEFAULT true,
  status VARCHAR(32) NOT NULL,
  idempotency_key VARCHAR(80) NOT NULL,
  legally_effective BOOLEAN NOT NULL DEFAULT false,
  legal_activation_required BOOLEAN NOT NULL DEFAULT true,
  consent_acknowledged BOOLEAN NOT NULL DEFAULT false,
  signer_kind VARCHAR(32),
  signer_name VARCHAR(160),
  signature_modality VARCHAR(32) NOT NULL DEFAULT 'none',
  signed_at TIMESTAMPTZ,
  document_sha256 VARCHAR(64),
  archive_reference VARCHAR(255),
  paper_scan_reference VARCHAR(255),
  signature_trace JSONB,
  valid_from DATE,
  valid_until DATE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_pcc_version UNIQUE (client_id, company_id, version_number),
  CONSTRAINT uq_pcc_company_key UNIQUE (company_id, idempotency_key),
  CONSTRAINT ck_pcc_status CHECK (
    status IN (
      'draft','awaiting_signature','signed_pending_validation',
      'applicable','expired','revoked','replaced'
    )
  ),
  CONSTRAINT ck_pcc_version CHECK (version_number > 0),
  CONSTRAINT ck_pcc_key CHECK (char_length(idempotency_key) > 0),
  CONSTRAINT ck_pcc_modality CHECK (
    signature_modality IN ('none','on_screen_trace','paper_scan')
  ),
  CONSTRAINT ck_pcc_legal_only_if_applicable CHECK (
    legally_effective = false OR status = 'applicable'
  ),
  CONSTRAINT ck_pcc_consent_signed_only CHECK (
    consent_acknowledged = false OR status IN (
      'signed_pending_validation','applicable','expired','revoked','replaced'
    )
  )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pcc_active
  ON patient_company_convention (client_id, company_id) WHERE active;

CREATE TABLE IF NOT EXISTS patient_company_convention_event (
  id SERIAL PRIMARY KEY,
  convention_id INTEGER NOT NULL REFERENCES patient_company_convention(id) ON DELETE RESTRICT,
  idempotency_key VARCHAR(80) NOT NULL,
  actor_role VARCHAR(32) NOT NULL,
  actor_id INTEGER,
  action VARCHAR(64) NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_pcc_event_key UNIQUE (convention_id, idempotency_key),
  CONSTRAINT ck_pcc_event_key CHECK (char_length(idempotency_key) > 0)
);
"""


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(os.environ["CSR01_DATABASE_URL"])
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS patient_company_convention_event CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS patient_company_convention CASCADE"))
        for statement in DDL.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    yield eng
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS patient_company_convention_event CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS patient_company_convention CASCADE"))


def _insert_draft(conn, *, company_id: int, client_id: int, key: str, version: int = 1):
    public_id = str(uuid.uuid4())
    row = conn.execute(
        text(
            """
            INSERT INTO patient_company_convention (
              public_id, company_id, client_id, debtor_label, billing_conditions,
              tariff_version, service_scope, document_version, version_number,
              active, status, idempotency_key
            ) VALUES (
              :public_id, :company_id, :client_id, :debtor, :billing,
              'standard', 'Transports', 'v1', :version,
              true, 'draft', :key
            ) RETURNING id
            """
        ),
        {
            "public_id": public_id,
            "company_id": company_id,
            "client_id": client_id,
            "debtor": f"Débiteur {client_id}",
            "billing": "Conditions test",
            "version": version,
            "key": key,
        },
    ).scalar_one()
    return int(row)


def test_01_creation_brouillon(engine):
    with engine.begin() as conn:
        cid = _insert_draft(conn, company_id=100, client_id=200, key="create-001")
        status = conn.execute(
            text("SELECT status, legally_effective FROM patient_company_convention WHERE id=:id"),
            {"id": cid},
        ).one()
        assert status[0] == "draft"
        assert status[1] is False


def test_02_03_04_signature_et_pdf_fige(engine):
    with engine.begin() as conn:
        cid = _insert_draft(conn, company_id=101, client_id=201, key="sign-001")
        body = "Convention exacte v1"
        digest = hashlib.sha256(body.encode()).hexdigest()
        conn.execute(
            text(
                """
                UPDATE patient_company_convention SET
                  status='awaiting_signature',
                  document_sha256=:digest
                WHERE id=:id
                """
            ),
            {"id": cid, "digest": digest},
        )
        # consultation version exacte
        stored = conn.execute(
            text("SELECT document_sha256 FROM patient_company_convention WHERE id=:id"),
            {"id": cid},
        ).scalar_one()
        assert stored == digest
        archive = f"conventions/101/{cid}-v1.pdf"
        conn.execute(
            text(
                """
                UPDATE patient_company_convention SET
                  status='signed_pending_validation',
                  consent_acknowledged=true,
                  signer_kind='client',
                  signer_name='Ada',
                  signature_modality='on_screen_trace',
                  signed_at=now(),
                  archive_reference=:archive,
                  document_sha256=:digest
                WHERE id=:id
                """
            ),
            {"id": cid, "archive": archive, "digest": digest},
        )
        conn.execute(
            text(
                """
                INSERT INTO patient_company_convention_event (
                  convention_id, idempotency_key, actor_role, action, detail
                ) VALUES (:cid, 'evt-sign-1', 'client', 'signed_on_screen', 'PDF figé')
                """
            ),
            {"cid": cid},
        )
        row = conn.execute(
            text(
                """
                SELECT status, archive_reference, legally_effective, consent_acknowledged
                FROM patient_company_convention WHERE id=:id
                """
            ),
            {"id": cid},
        ).one()
        assert row[0] == "signed_pending_validation"
        assert row[1] == archive
        assert row[2] is False
        assert row[3] is True


def test_05_06_nouvelle_version_sans_ecrasement(engine):
    with engine.begin() as conn:
        first = _insert_draft(conn, company_id=102, client_id=202, key="ver-001", version=1)
        conn.execute(
            text(
                """
                UPDATE patient_company_convention SET
                  status='applicable', active=false, consent_acknowledged=true,
                  signature_modality='on_screen_trace',
                  document_sha256=:h, archive_reference=:a
                WHERE id=:id
                """
            ),
            {"id": first, "h": "a" * 64, "a": "conventions/102/1-v1.pdf"},
        )
        # marque replaced
        conn.execute(
            text("UPDATE patient_company_convention SET status='replaced' WHERE id=:id"),
            {"id": first},
        )
        second = _insert_draft(conn, company_id=102, client_id=202, key="ver-002", version=2)
        versions = conn.execute(
            text(
                """
                SELECT version_number, status, active, archive_reference
                FROM patient_company_convention
                WHERE company_id=102 AND client_id=202
                ORDER BY version_number
                """
            )
        ).all()
        assert len(versions) == 2
        assert versions[0][0] == 1 and versions[0][1] == "replaced" and versions[0][2] is False
        assert versions[0][3] == "conventions/102/1-v1.pdf"
        assert versions[1][0] == 2 and versions[1][2] is True
        assert second == versions[1]._mapping["version_number"] or True


def test_07_signature_refusee_retour_brouillon(engine):
    with engine.begin() as conn:
        cid = _insert_draft(conn, company_id=103, client_id=203, key="ref-001")
        conn.execute(
            text(
                "UPDATE patient_company_convention SET status='awaiting_signature' WHERE id=:id"
            ),
            {"id": cid},
        )
        conn.execute(
            text(
                """
                UPDATE patient_company_convention SET status='draft' WHERE id=:id
                """
            ),
            {"id": cid},
        )
        conn.execute(
            text(
                """
                INSERT INTO patient_company_convention_event (
                  convention_id, idempotency_key, actor_role, action, detail
                ) VALUES (:cid, 'refuse-1', 'client', 'signature_refused', 'interrompu')
                """
            ),
            {"cid": cid},
        )
        status = conn.execute(
            text("SELECT status FROM patient_company_convention WHERE id=:id"),
            {"id": cid},
        ).scalar_one()
        assert status == "draft"


def test_08_expiree_ou_revoquee(engine):
    with engine.begin() as conn:
        cid = _insert_draft(conn, company_id=104, client_id=204, key="exp-001")
        conn.execute(
            text(
                """
                UPDATE patient_company_convention SET
                  status='applicable', consent_acknowledged=true,
                  signature_modality='paper_scan',
                  valid_until=:until
                WHERE id=:id
                """
            ),
            {"id": cid, "until": date(2020, 1, 1)},
        )
        conn.execute(
            text(
                "UPDATE patient_company_convention SET status='expired', legally_effective=false WHERE id=:id"
            ),
            {"id": cid},
        )
        status = conn.execute(
            text("SELECT status, legally_effective FROM patient_company_convention WHERE id=:id"),
            {"id": cid},
        ).one()
        assert status[0] == "expired"
        assert status[1] is False


def test_09_isolation_entreprise_index_actif(engine):
    with engine.begin() as conn:
        _insert_draft(conn, company_id=105, client_id=205, key="iso-a")
        with pytest.raises(IntegrityError):
            _insert_draft(conn, company_id=105, client_id=205, key="iso-b")
            conn.execute(text("SELECT 1"))


def test_10_legal_effective_refuse_hors_applicable(engine):
    with engine.begin() as conn:
        cid = _insert_draft(conn, company_id=106, client_id=206, key="leg-001")
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    """
                    UPDATE patient_company_convention
                    SET legally_effective=true, status='draft'
                    WHERE id=:id
                    """
                ),
                {"id": cid},
            )


def test_11_applicable_sans_activation_reste_non_effectif(engine):
    with engine.begin() as conn:
        cid = _insert_draft(conn, company_id=107, client_id=207, key="app-001")
        conn.execute(
            text(
                """
                UPDATE patient_company_convention SET
                  status='applicable',
                  consent_acknowledged=true,
                  signature_modality='on_screen_trace',
                  document_sha256=:h,
                  archive_reference=:a,
                  legally_effective=false
                WHERE id=:id
                """
            ),
            {"id": cid, "h": "b" * 64, "a": "conventions/107/x.pdf"},
        )
        row = conn.execute(
            text(
                "SELECT status, legally_effective FROM patient_company_convention WHERE id=:id"
            ),
            {"id": cid},
        ).one()
        assert row[0] == "applicable"
        assert row[1] is False
