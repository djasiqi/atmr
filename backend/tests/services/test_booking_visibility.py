"""Garde-fou visibilité réservations : OR historique ≡ UNION ALL disjoint."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ext import db
from models import Booking, Company, User
from models.booking_transfer import BookingTransfer
from models.enums import (
    BookingStatus,
    DispatchOfferStatus,
    PartnershipStatus,
    TransferModel,
    TransferStatus,
    UserRole,
)
from models.partnership import Partnership
from models.service_area_pricing import DispatchOffer
from repositories.booking_repository import BookingRepository
from services.companies.booking_visibility import (
    visibility_branch_predicates,
    visible_booking_id_union,
    visible_bookings_query,
)


def _company(suffix: str) -> Company:
    user = User()
    user.username = f"vis_{suffix}"
    user.email = f"vis_{suffix}@test.ch"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.name = f"Vis {suffix}"
    company.address = "Rue Vis 1"
    company.contact_email = user.email
    company.user_id = user.id
    company.is_approved = True
    company.dispatch_enabled = False
    db.session.add(company)
    db.session.flush()
    return company


def _booking(**kwargs) -> Booking:
    booking = Booking()
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
    booking.status = BookingStatus.PENDING
    booking.amount = Decimal("45.00")
    booking.billed_to_type = "patient"
    booking.customer_name = "Vis Client"
    for key, value in kwargs.items():
        setattr(booking, key, value)
    db.session.add(booking)
    db.session.flush()
    return booking


def _partnership(owner_id: int, partner_id: int) -> Partnership:
    partnership = Partnership()
    partnership.owner_company_id = owner_id
    partnership.partner_company_id = partner_id
    partnership.default_transfer_model = TransferModel.SUBCONTRACT
    partnership.default_partner_tariff_percent = Decimal("80.00")
    partnership.default_margin_percent = Decimal("20.00")
    partnership.auto_accept_rules = False
    partnership.auto_invoice = True
    partnership.payment_terms_days = 30
    partnership.status = PartnershipStatus.ACCEPTED
    partnership.is_active = True
    db.session.add(partnership)
    db.session.flush()
    return partnership


def _transfer(*, booking_id: int, partnership_id: int, owner_id: int, exec_id: int):
    transfer = BookingTransfer()
    transfer.booking_id = booking_id
    transfer.partnership_id = partnership_id
    transfer.transfer_model = TransferModel.SUBCONTRACT
    transfer.owner_company_id = owner_id
    transfer.executing_company_id = exec_id
    transfer.client_price = Decimal("50.00")
    transfer.partner_cost = Decimal("40.00")
    transfer.platform_fee = Decimal("0.00")
    transfer.currency = "CHF"
    transfer.vat_rate = Decimal("0.00")
    transfer.vat_included = True
    transfer.status = TransferStatus.ACCEPTED
    db.session.add(transfer)
    db.session.flush()
    return transfer


def _legacy_ids(company_id: int) -> set[int]:
    return {
        int(row[0])
        for row in Booking.query.filter(
            BookingRepository._company_visibility_filter(company_id)
        )
        .with_entities(Booking.id)
        .all()
    }


def _union_ids(company_id: int) -> list[int]:
    rows = db.session.execute(visible_booking_id_union(company_id)).all()
    return [int(row[0]) for row in rows]


def _seed_disjoint_world(db):
    suffix = uuid.uuid4().hex[:8]
    viewer = _company(f"v_{suffix}")
    other = _company(f"o_{suffix}")
    partnership = _partnership(viewer.id, other.id)

    owned = _booking(company_id=viewer.id, user_id=viewer.user_id)
    # Transfert sur une course déjà propriétaire : ne doit pas dupliquer (owned gagne).
    _transfer(
        booking_id=owned.id,
        partnership_id=partnership.id,
        owner_id=viewer.id,
        exec_id=other.id,
    )

    executor = _booking(
        company_id=other.id,
        user_id=other.user_id,
        executing_company_id=viewer.id,
        status=BookingStatus.ACCEPTED,
    )

    transfer_owner = _booking(
        company_id=other.id,
        user_id=other.user_id,
        executing_company_id=other.id,
        status=BookingStatus.ACCEPTED,
    )
    _transfer(
        booking_id=transfer_owner.id,
        partnership_id=partnership.id,
        owner_id=viewer.id,
        exec_id=other.id,
    )

    open_offer = _booking(company_id=None, user_id=other.user_id)
    db.session.add(
        DispatchOffer(
            booking_id=open_offer.id,
            company_id=viewer.id,
            status=DispatchOfferStatus.PROPOSED,
            score=10,
            reason_json={"source": "visibility-guard"},
        )
    )
    db.session.commit()
    return {
        "viewer": viewer,
        "owned": owned,
        "executor": executor,
        "transfer_owner": transfer_owner,
        "open_offer": open_offer,
    }


def test_legacy_or_ids_equal_union_ids(db):
    world = _seed_disjoint_world(db)
    company_id = world["viewer"].id
    union_rows = _union_ids(company_id)
    assert set(union_rows) == _legacy_ids(company_id)
    query_ids = {
        int(row[0])
        for row in visible_bookings_query(company_id).with_entities(Booking.id).all()
    }
    assert query_ids == _legacy_ids(company_id)


def test_union_all_rows_are_distinct(db):
    """COUNT(union) == COUNT(DISTINCT id) — les 4 branches restent disjointes."""
    world = _seed_disjoint_world(db)
    union_rows = _union_ids(world["viewer"].id)
    assert union_rows
    assert len(union_rows) == len(set(union_rows))


def test_visibility_branches_pairwise_disjoint(db):
    world = _seed_disjoint_world(db)
    company_id = world["viewer"].id
    expected = {
        "owned": {world["owned"].id},
        "executor": {world["executor"].id},
        "transfer_owner": {world["transfer_owner"].id},
        "open_offer": {world["open_offer"].id},
    }
    branch_ids: dict[str, set[int]] = {}
    for name, predicate in visibility_branch_predicates(company_id).items():
        branch_ids[name] = {
            int(row[0])
            for row in Booking.query.filter(predicate).with_entities(Booking.id).all()
        }
    names = list(branch_ids)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = branch_ids[left] & branch_ids[right]
            assert not overlap, f"{left} ∩ {right} = {overlap}"
    for name, ids in expected.items():
        assert ids <= branch_ids[name], f"{name}: {ids} ⊄ {branch_ids[name]}"
