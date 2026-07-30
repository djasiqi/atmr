"""Lot 6 perf company-space — GET /companies/<id>/invoices (liste unifiée).

Vérifie que le tri et la pagination des factures normales + partenaires sont
effectués en SQL (``UNION ALL`` + ``ORDER BY issued_at DESC NULLS LAST, id DESC``
+ ``OFFSET/LIMIT``) et non par fusion Python de deux listes déjà paginées
indépendamment :

- l'ordre combiné respecte la date d'émission décroissante, tous types confondus,
- chaque page contient exactement ``per_page`` éléments cohérents entre eux
  (pas de doublon, pas de trou),
- le départage (même ``issued_at``) est déterministe et stable d'un appel à l'autre,
- le total_count reflète bien la somme des deux types de factures.

Voir docs/perf-company-space-lot6-invoices.md.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token

from models import Booking, Client, Company, Invoice, User, UserRole
from models.booking_transfer import BookingTransfer
from models.enums import InvoiceStatus, TransferModel, TransferStatus
from models.partner_invoice import (
    PartnerInvoice,
    PartnerInvoiceStatus,
    partner_invoice_transfers,
)
from models.partnership import Partnership


def _company_headers(client, user, company_id: int) -> dict[str, str]:
    claims = {
        "role": user.role.value,
        "company_id": company_id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def lot6_invoice_client(db, sample_company):
    """Client (patient) rattaché à l'entreprise, pour les factures normales."""
    uid = uuid.uuid4().hex[:8]
    user = User()
    user.public_id = str(uuid.uuid4())
    user.username = f"lot6_patient_{uid}"
    user.email = f"lot6-patient-{uid}@test.ch"
    user.role = UserRole.CLIENT
    user.first_name = "Lot6"
    user.last_name = "Patient"
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    client_row = Client()
    client_row.user_id = user.id
    client_row.company_id = sample_company.id
    db.session.add(client_row)
    db.session.flush()
    return client_row


@pytest.fixture
def lot6_partnership(db, sample_company):
    """Partenariat entre l'entreprise de test et une entreprise partenaire."""
    uid = uuid.uuid4().hex[:8]
    partner_user = User()
    partner_user.public_id = str(uuid.uuid4())
    partner_user.username = f"lot6_partner_user_{uid}"
    partner_user.email = f"lot6-partner-{uid}@test.ch"
    partner_user.role = UserRole.company
    partner_user.set_password("password123", force_change=False)
    db.session.add(partner_user)
    db.session.flush()

    partner_company = Company()
    partner_company.name = f"Partenaire Lot6 {uid}"
    partner_company.user_id = partner_user.id
    partner_company.address = "Rue Partenaire 1, 1000 Lausanne"
    partner_company.is_approved = True
    db.session.add(partner_company)
    db.session.flush()

    partnership = Partnership()
    partnership.owner_company_id = sample_company.id
    partnership.partner_company_id = partner_company.id
    db.session.add(partnership)
    db.session.flush()
    return partnership


def _make_regular_invoice(db, company, client_row, *, issued_at, amount="100.00"):
    inv = Invoice()
    inv.company_id = company.id
    inv.client_id = client_row.id
    inv.invoice_number = f"INV-LOT6-{uuid.uuid4().hex[:8]}"
    inv.period_year = issued_at.year
    inv.period_month = issued_at.month
    inv.status = InvoiceStatus.SENT
    inv.subtotal_amount = Decimal(amount)
    inv.total_amount = Decimal(amount)
    inv.balance_due = Decimal(amount)
    inv.issued_at = issued_at
    inv.due_date = issued_at + timedelta(days=30)
    db.session.add(inv)
    db.session.flush()
    return inv


def _make_partner_invoice(db, company, partnership, *, issued_at, amount="200.00"):
    """Crée une PartnerInvoice + un BookingTransfer COMPLETED exécuté par ``company``.

    C'est la condition d'appartenance vérifiée par l'EXISTS SQL (Lot 6) :
    un transfert complété dont ``executing_company_id`` == entreprise courante.
    """
    booking = Booking()
    booking.customer_name = "Client transféré Lot6"
    booking.pickup_location = "Rue Alpha 1, Genève"
    booking.dropoff_location = "Rue Beta 2, Genève"
    booking.amount = float(amount)
    db.session.add(booking)
    db.session.flush()

    partner_invoice = PartnerInvoice()
    partner_invoice.partnership_id = partnership.id
    partner_invoice.executing_company_id = company.id
    partner_invoice.period_year = issued_at.year
    partner_invoice.period_month = issued_at.month
    partner_invoice.invoice_number = f"PINV-LOT6-{uuid.uuid4().hex[:8]}"
    partner_invoice.subtotal_amount = Decimal(amount)
    partner_invoice.total_amount = Decimal(amount)
    partner_invoice.status = PartnerInvoiceStatus.SENT
    partner_invoice.issued_at = issued_at
    partner_invoice.due_date = issued_at + timedelta(days=30)
    db.session.add(partner_invoice)
    db.session.flush()

    transfer = BookingTransfer()
    transfer.booking_id = booking.id
    transfer.partnership_id = partnership.id
    transfer.transfer_model = TransferModel.SUBCONTRACT
    transfer.owner_company_id = partnership.partner_company_id
    transfer.executing_company_id = company.id
    transfer.client_price = Decimal(amount)
    transfer.status = TransferStatus.COMPLETED
    db.session.add(transfer)
    db.session.flush()

    db.session.execute(
        partner_invoice_transfers.insert().values(
            partner_invoice_id=partner_invoice.id,
            booking_transfer_id=transfer.id,
        )
    )
    db.session.flush()
    return partner_invoice


class TestInvoicesListLot6MixedOrderAndPagination:
    """Ordre + pagination SQL d'une liste mêlant factures normales et partenaires."""

    def test_mixed_order_is_deterministic_and_paginated_in_sql(
        self,
        client,
        db,
        sample_user,
        sample_company,
        lot6_invoice_client,
        lot6_partnership,
    ):
        base = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

        # Ordre d'émission croissant : regular_old < partner_old < regular_new < partner_new
        regular_old = _make_regular_invoice(
            db, sample_company, lot6_invoice_client, issued_at=base
        )
        partner_old = _make_partner_invoice(
            db, sample_company, lot6_partnership, issued_at=base + timedelta(hours=1)
        )
        regular_new = _make_regular_invoice(
            db,
            sample_company,
            lot6_invoice_client,
            issued_at=base + timedelta(hours=2),
        )
        partner_new = _make_partner_invoice(
            db, sample_company, lot6_partnership, issued_at=base + timedelta(hours=3)
        )

        headers = _company_headers(client, sample_user, sample_company.id)

        # Page 1/2 (per_page=2) : les deux factures les plus récentes, tous types confondus
        resp_page1 = client.get(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices?page=1&per_page=2",
            headers=headers,
        )
        assert resp_page1.status_code == 200
        body1 = resp_page1.get_json()
        page1_items = body1["data"]
        assert body1["pagination"]["total"] == 4
        assert body1["pagination"]["pages"] == 2
        assert len(page1_items) == 2
        assert page1_items[0]["id"] == partner_new.id
        assert page1_items[0].get("is_partner_invoice") is True
        assert page1_items[1]["id"] == regular_new.id
        assert not page1_items[1].get("is_partner_invoice")

        # Page 2/2 : les deux factures les plus anciennes, sans doublon avec la page 1
        resp_page2 = client.get(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices?page=2&per_page=2",
            headers=headers,
        )
        assert resp_page2.status_code == 200
        body2 = resp_page2.get_json()
        page2_items = body2["data"]
        assert len(page2_items) == 2
        assert page2_items[0]["id"] == partner_old.id
        assert page2_items[0].get("is_partner_invoice") is True
        assert page2_items[1]["id"] == regular_old.id
        assert not page2_items[1].get("is_partner_invoice")

        # Pas de fusion naïve de deux listes indépendamment paginées : aucun doublon,
        # les 4 factures sont vues exactement une fois sur l'ensemble des 2 pages.
        seen_ids = {(item["id"], item.get("is_partner_invoice", False)) for item in page1_items}
        seen_ids |= {(item["id"], item.get("is_partner_invoice", False)) for item in page2_items}
        assert seen_ids == {
            (partner_new.id, True),
            (regular_new.id, False),
            (partner_old.id, True),
            (regular_old.id, False),
        }

        # Stats agrégées (SQL, Decimal en interne) : somme normales + partenaires
        stats = body1["stats"]
        assert stats["total_issued"] == pytest.approx(100.00 + 100.00 + 200.00 + 200.00)

    def test_tie_break_on_identical_issued_at_is_stable(
        self,
        client,
        db,
        sample_user,
        sample_company,
        lot6_invoice_client,
        lot6_partnership,
    ):
        """Départage déterministe (id DESC) quand deux factures partagent le même issued_at."""
        same_ts = datetime(2026, 2, 1, 8, 0, tzinfo=UTC)

        regular_inv = _make_regular_invoice(
            db, sample_company, lot6_invoice_client, issued_at=same_ts
        )
        partner_inv = _make_partner_invoice(
            db, sample_company, lot6_partnership, issued_at=same_ts
        )

        headers = _company_headers(client, sample_user, sample_company.id)
        url = f"/api/v1/invoices/companies/{sample_company.id}/invoices?page=1&per_page=10"

        first_call = client.get(url, headers=headers).get_json()["data"]
        second_call = client.get(url, headers=headers).get_json()["data"]

        ordered_keys_first = [(item["id"], item.get("is_partner_invoice", False)) for item in first_call]
        ordered_keys_second = [(item["id"], item.get("is_partner_invoice", False)) for item in second_call]

        # Même ordre à chaque appel (stabilité du départage SQL, pas un tri Python
        # instable dépendant de l'ordre d'itération).
        assert ordered_keys_first == ordered_keys_second
        assert set(ordered_keys_first) == {
            (regular_inv.id, False),
            (partner_inv.id, True),
        }
