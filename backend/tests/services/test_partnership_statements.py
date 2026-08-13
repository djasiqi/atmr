"""Couverture de ``services.partnerships.statements`` (décomptes PDF)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from flask import current_app

from ext import db
from models.booking_transfer import BookingTransfer
from models.enums import (
    BookingStatus,
    PartnershipStatus,
    TransferModel,
    TransferStatus,
)
from models.partnership import Partnership
from services.partnerships.statements import PartnershipStatementService
from tests.factories import BookingFactory, ClientFactory, CompanyFactory


def _svc(tmp_path):
    current_app.config["UPLOADS_DIR"] = str(tmp_path)
    return PartnershipStatementService(pdf_service=MagicMock())


def _partnership(owner, partner, *, active: bool = True):
    row = Partnership(
        owner_company_id=owner.id,
        partner_company_id=partner.id,
        status=PartnershipStatus.ACCEPTED,
        default_transfer_model=TransferModel.SUBCONTRACT,
        payment_terms_days=30,
        is_active=active,
    )
    db.session.add(row)
    db.session.flush()
    return row


def _booking(company, when: datetime):
    client = ClientFactory(company=company)
    return BookingFactory(
        company=company,
        client=client,
        scheduled_time=when,
        status=BookingStatus.PENDING,
        amount=45.0,
        customer_name="Client Très Long Nom Pour Tronquer Le Tableau PDF",
    )


def _transfer(*, booking, partnership, owner, executing, when: datetime, cost=None):
    aware = when.replace(tzinfo=UTC) if when.tzinfo is None else when
    transfer = BookingTransfer(
        booking_id=booking.id,
        partnership_id=partnership.id,
        transfer_model=TransferModel.SUBCONTRACT,
        status=TransferStatus.COMPLETED,
        is_validated=True,
        validated_at=aware,
        completed_at=aware,
        client_price=Decimal("45.00"),
        partner_cost=cost,
        currency="CHF",
        owner_company_id=owner.id,
        executing_company_id=executing.id,
    )
    db.session.add(transfer)
    db.session.flush()
    return transfer


def _world():
    owner = CompanyFactory(address="Rue Verte 8, Genève")
    partner = CompanyFactory()
    partnership = _partnership(owner, partner)
    when = datetime(2030, 6, 15, 10, 0, 0)
    booking = _booking(partner, when)
    transfer = _transfer(
        booking=booking,
        partnership=partnership,
        owner=owner,
        executing=partner,
        when=when,
        cost=Decimal("40.00"),
    )
    return SimpleNamespace(
        owner=owner,
        partner=partner,
        partnership=partnership,
        booking=booking,
        transfer=transfer,
        when=when,
    )


def test_init_default_pdf_service(db):
    service = PartnershipStatementService()
    assert service.pdf_service is not None


def test_period_annual_monthly_periodic_et_erreurs(db, tmp_path):
    service = _svc(tmp_path)
    annual = service._calculate_period_dates("annual", 2030, None, None, None)
    assert annual["label"] == "Année 2030"
    assert annual["start"] == datetime(2030, 1, 1, tzinfo=UTC)
    assert annual["end"] == datetime(2031, 1, 1, tzinfo=UTC)

    annual_default = service._calculate_period_dates("annual", None, None, None, None)
    assert str(datetime.now(UTC).year) in annual_default["label"]

    monthly = service._calculate_period_dates("monthly", 2030, 6, None, None)
    assert monthly["label"] == "Juin 2030"
    assert monthly["end"] == datetime(2030, 7, 1, tzinfo=UTC)

    december = service._calculate_period_dates("monthly", 2030, 12, None, None)
    assert december["label"] == "Décembre 2030"
    assert december["end"] == datetime(2031, 1, 1, tzinfo=UTC)

    monthly_default = service._calculate_period_dates(
        "monthly", None, None, None, None
    )
    assert monthly_default["start"].day == 1

    with pytest.raises(ValueError, match="mois doit être"):
        service._calculate_period_dates("monthly", 2030, 0, None, None)
    with pytest.raises(ValueError, match="mois doit être"):
        service._calculate_period_dates("monthly", 2030, 13, None, None)

    start = datetime(2030, 1, 10, tzinfo=UTC)
    end = datetime(2030, 2, 10, tzinfo=UTC)
    periodic = service._calculate_period_dates("periodic", None, None, start, end)
    assert "Du 10.01.2030" in periodic["label"]

    with pytest.raises(ValueError, match="dates de début"):
        service._calculate_period_dates("periodic", None, None, None, None)
    with pytest.raises(ValueError, match="Type de période invalide"):
        service._calculate_period_dates("weekly", None, None, None, None)


def test_consolidated_erreurs(db, tmp_path):
    service = _svc(tmp_path)
    with pytest.raises(ValueError, match="introuvable"):
        service.generate_consolidated_statement(999999, "monthly", 2030, 6)

    lonely = CompanyFactory()
    with pytest.raises(ValueError, match="Aucun partenariat"):
        service.generate_consolidated_statement(lonely.id, "monthly", 2030, 6)

    owner = CompanyFactory()
    partner = CompanyFactory()
    _partnership(owner, partner, active=False)
    with pytest.raises(ValueError, match="Aucun partenariat"):
        service.generate_consolidated_statement(owner.id, "monthly", 2030, 6)


def test_consolidated_ok_owner_et_partenaire(db, tmp_path):
    world = _world()
    idle_partner = CompanyFactory()
    _partnership(world.owner, idle_partner)
    service = _svc(tmp_path)

    url_owner = service.generate_consolidated_statement(
        world.owner.id, "monthly", 2030, 6
    )
    assert "/statements/" in url_owner
    assert url_owner.endswith(".pdf")

    url_partner = service.generate_consolidated_statement(
        world.partner.id, "annual", 2030
    )
    assert "decompte_consolide" in url_partner


def test_partnership_statement_erreurs_et_ok(db, tmp_path):
    world = _world()
    service = _svc(tmp_path)
    stranger = CompanyFactory()

    with pytest.raises(ValueError, match="Type de période"):
        service.generate_partnership_statement(
            world.partnership.id, world.owner.id, "weekly"
        )
    with pytest.raises(ValueError, match="introuvable"):
        service.generate_partnership_statement(999999, world.owner.id, "monthly", 2030, 6)
    with pytest.raises(ValueError, match="autorisé"):
        service.generate_partnership_statement(
            world.partnership.id, stranger.id, "monthly", 2030, 6
        )

    url_owner = service.generate_partnership_statement(
        world.partnership.id, world.owner.id, "monthly", 2030, 6
    )
    assert "decompte_partenaire" in url_owner

    url_partner = service.generate_partnership_statement(
        world.partnership.id,
        world.partner.id,
        "periodic",
        start_date=datetime(2030, 6, 1, tzinfo=UTC),
        end_date=datetime(2030, 7, 1, tzinfo=UTC),
    )
    assert url_partner.endswith(".pdf")


def test_partnership_statement_company_disparue(db, tmp_path, monkeypatch):
    world = _world()
    service = _svc(tmp_path)

    class _Query:
        @staticmethod
        def get(_id):
            return None

    monkeypatch.setattr(
        "services.partnerships.statements.Company",
        SimpleNamespace(query=_Query()),
    )
    with pytest.raises(ValueError, match=r"Entreprise .* introuvable"):
        service.generate_partnership_statement(
            world.partnership.id, world.owner.id, "monthly", 2030, 6
        )


def test_pdf_adresses_et_branches_consolidated(db, tmp_path):
    world = _world()
    service = _svc(tmp_path)
    start = datetime(2030, 6, 1, tzinfo=UTC)
    end = datetime(2030, 7, 1, tzinfo=UTC)

    world.owner.address = None
    world.owner.domicile_address_line1 = "Chemin 1"
    world.owner.domicile_address_line2 = "Bâtiment B"
    world.owner.domicile_zip = "1205"
    world.owner.domicile_city = "Genève"
    url = service._generate_statement_pdf(
        {
            "company": world.owner,
            "type": "consolidated",
            "period_label": "Juin 2030",
            "start_date": start,
            "end_date": end,
            "partnership_summaries": [
                {
                    "partner_company": None,
                    "count": 1,
                    "client_price": Decimal("10"),
                    "partner_cost": Decimal("8"),
                    "balance": Decimal("-2"),
                }
            ],
            "total_courses": 1,
            "total_client_price": Decimal("10"),
            "total_partner_cost": Decimal("8"),
            "net_balance": Decimal("-2"),
        },
        "consolidated",
    )
    assert url.endswith(".pdf")

    world.owner.domicile_address_line2 = None
    world.owner.domicile_zip = None
    world.owner.domicile_city = "Lausanne"
    url2 = service._generate_statement_pdf(
        {
            "company": world.owner,
            "type": "consolidated",
            "period_label": "Juin 2030",
            "start_date": start,
            "end_date": end,
            "partnership_summaries": [],
            "total_courses": 0,
            "total_client_price": Decimal("0"),
            "total_partner_cost": Decimal("0"),
            "net_balance": Decimal("0"),
        },
        "consolidated",
    )
    assert "/statements/" in url2

    world.owner.domicile_address_line1 = None
    world.owner.domicile_city = None
    url3 = service._generate_statement_pdf(
        {
            "company": world.owner,
            "type": "consolidated",
            "period_label": "Juin 2030",
            "start_date": start,
            "end_date": end,
            "partnership_summaries": [],
            "total_courses": 0,
            "total_client_price": Decimal("0"),
            "total_partner_cost": Decimal("0"),
            "net_balance": Decimal("0"),
        },
        "consolidated",
    )
    assert url3.endswith(".pdf")


def test_pdf_single_transferts_et_vides(db, tmp_path):
    world = _world()
    service = _svc(tmp_path)
    start = datetime(2030, 6, 1, tzinfo=UTC)
    end = datetime(2030, 7, 1, tzinfo=UTC)

    url_empty = service._generate_statement_pdf(
        {
            "company": world.owner,
            "partner_company": None,
            "partnership": world.partnership,
            "type": "single",
            "period_label": "Juin 2030",
            "start_date": start,
            "end_date": end,
            "transfers": [],
            "total_courses": 0,
            "total_client_price": Decimal("0"),
            "total_partner_cost": Decimal("0"),
            "net_balance": Decimal("0"),
        },
        "single",
    )
    assert "decompte_partenaire" in url_empty

    url_detail = service._generate_statement_pdf(
        {
            "company": world.owner,
            "partner_company": world.partner,
            "partnership": world.partnership,
            "type": "single",
            "period_label": "Juin 2030",
            "start_date": start,
            "end_date": end,
            "transfers": [
                world.transfer,
                SimpleNamespace(
                    booking_id=424242,
                    completed_at=None,
                    client_price=Decimal("12.00"),
                    partner_cost=None,
                ),
            ],
            "total_courses": 2,
            "total_client_price": Decimal("57.00"),
            "total_partner_cost": Decimal("40.00"),
            "net_balance": Decimal("-17.00"),
        },
        "single",
    )
    assert url_detail.endswith(".pdf")


def test_organize_partner_cost_nul(db, tmp_path):
    world = _world()
    world.transfer.partner_cost = None
    db.session.flush()
    service = _svc(tmp_path)
    data = service._organize_single_partnership_data(
        world.owner,
        world.partner,
        world.partnership,
        [world.transfer],
        datetime(2030, 6, 1, tzinfo=UTC),
        datetime(2030, 7, 1, tzinfo=UTC),
        "Juin 2030",
    )
    assert data["total_partner_cost"] == Decimal("0")
    assert data["total_client_price"] == Decimal("45.00")
