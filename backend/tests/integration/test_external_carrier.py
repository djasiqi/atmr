# ruff: noqa: I001
"""Tests transporteur externe — use cases, concurrence accept_offer, sérialisation."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from application.institutions.accept_offer import AcceptOfferInput, AcceptOfferUseCase
from application.institutions.assign_external_carrier import (
    AssignExternalCarrierInput,
    AssignExternalCarrierUseCase,
)
from application.institutions.complete_external_mission import (
    CompleteExternalMissionInput,
    CompleteExternalMissionUseCase,
)
from models import (
    Booking,
    Company,
    Institution,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
    User,
    UserRole,
)
from models.enums import CarrierSource


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"Clinique Externe {uuid.uuid4().hex[:6]}"
    inst.institution_type = "clinic"
    db.session.add(inst)
    db.session.flush()
    return inst


@pytest.fixture
def institution_user(db, institution):
    user = User()
    user.email = f"req_{uuid.uuid4().hex[:8]}@test.com"
    user.username = user.email
    user.password = "test"
    user.role = UserRole.INSTITUTION.value
    user.institution_id = institution.id
    user.institution_role = "institution_requester"
    db.session.add(user)
    db.session.flush()
    return user


@pytest.fixture
def company(db):
    user = User()
    user.email = f"co_{uuid.uuid4().hex[:8]}@test.com"
    user.username = user.email
    user.password = "test"
    user.role = UserRole.COMPANY.value
    db.session.add(user)
    db.session.flush()
    co = Company()
    co.name = "Transport LIRIE SA"
    co.user_id = user.id
    co.is_approved = True
    co.dispatch_enabled = True
    db.session.add(co)
    db.session.flush()
    return co


def _transport_request(
    db,
    institution: Institution,
    *,
    status: str = RequestStatus.SENT.value,
    carrier_source: str = CarrierSource.LIRIE.value,
) -> TransportRequest:
    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.external_reference = f"EXT-{uuid.uuid4().hex[:8]}"
    tr.pickup_location = "Clinique, Genève"
    tr.dropoff_location = "Hôpital, Genève"
    tr.mission_date = datetime.now(UTC).date()
    tr.scheduled_time = datetime.now(UTC)
    tr.pickup_time_confirmed = True
    tr.status = status
    tr.carrier_source = carrier_source
    tr.billing_intent = "patient"
    db.session.add(tr)
    db.session.flush()
    return tr


class TestAssignExternalCarrierUseCase:
    def test_assign_from_sent_closes_pending_offers(
        self, db, institution, institution_user, company
    ):
        tr = _transport_request(db, institution, status=RequestStatus.SENT.value)
        offer = RequestOffer()
        offer.transport_request_id = tr.id
        offer.company_id = company.id
        offer.status = OfferStatus.PENDING.value
        offer.mode = "broadcast"
        offer.order = 1
        db.session.add(offer)
        db.session.commit()

        result = AssignExternalCarrierUseCase().execute(
            AssignExternalCarrierInput(
                transport_request_id=tr.id,
                institution_id=institution.id,
                user_id=institution_user.id,
                name="Taxi XYZ",
                phone="079 000 00 00",
                reason="Aucun transporteur LIRIE disponible",
            )
        )
        assert result.success is True
        assert result.switched_from_lirie is True

        db.session.refresh(tr)
        db.session.refresh(offer)
        assert tr.status == RequestStatus.EXTERNAL_ASSIGNED.value
        assert tr.carrier_source == CarrierSource.EXTERNAL.value
        assert tr.external_carrier_name == "Taxi XYZ"
        assert tr.assigned_externally_at is not None
        assert tr.externalized_by_user_id == institution_user.id
        assert tr.accepted_by_company_id is None
        assert offer.status == OfferStatus.UNAVAILABLE.value

        data = tr.serialize
        assert data["status_label"] == "Transporteur externe affecté"
        assert data["carrier_source"] == "external"
        assert data["external_carrier"]["name"] == "Taxi XYZ"

    def test_assign_from_draft_without_offers(self, db, institution, institution_user):
        tr = _transport_request(db, institution, status=RequestStatus.DRAFT.value)
        db.session.commit()

        result = AssignExternalCarrierUseCase().execute(
            AssignExternalCarrierInput(
                transport_request_id=tr.id,
                institution_id=institution.id,
                user_id=institution_user.id,
                name="Ambulances du Léman",
            )
        )
        assert result.success is True
        assert result.switched_from_lirie is False
        db.session.refresh(tr)
        assert tr.status == RequestStatus.EXTERNAL_ASSIGNED.value


class TestCompleteExternalMissionUseCase:
    def test_complete_external_mission(self, db, institution, institution_user):
        tr = _transport_request(
            db,
            institution,
            status=RequestStatus.EXTERNAL_ASSIGNED.value,
            carrier_source=CarrierSource.EXTERNAL.value,
        )
        tr.external_carrier_name = "Taxi XYZ"
        tr.assigned_externally_at = datetime.now(UTC)
        tr.externalized_by_user_id = institution_user.id
        db.session.commit()

        result = CompleteExternalMissionUseCase().execute(
            CompleteExternalMissionInput(
                transport_request_id=tr.id,
                institution_id=institution.id,
                user_id=institution_user.id,
                notes="Mission effectuée sans incident",
            )
        )
        assert result.success is True
        db.session.refresh(tr)
        assert tr.status == RequestStatus.EXTERNAL_DECLARED_COMPLETED.value
        assert tr.executed_externally_at is not None
        assert tr.executed_externally_by_user_id == institution_user.id
        assert tr.external_execution_notes == "Mission effectuée sans incident"
        assert tr.serialize["status_label"] == "Déclarée réalisée par l'institution"


class TestAcceptOfferExternalGuard:
    def test_accept_offer_blocked_when_external(self, db, institution, company):
        tr = _transport_request(
            db,
            institution,
            status=RequestStatus.SENT.value,
            carrier_source=CarrierSource.EXTERNAL.value,
        )
        tr.external_carrier_name = "Taxi XYZ"
        offer = RequestOffer()
        offer.transport_request_id = tr.id
        offer.company_id = company.id
        offer.status = OfferStatus.PENDING.value
        offer.mode = "broadcast"
        offer.order = 1
        db.session.add(offer)
        db.session.commit()

        result = AcceptOfferUseCase().execute(
            AcceptOfferInput(
                offer_id=offer.id,
                company_id=company.id,
                user_id=company.user_id,
            )
        )
        assert result.success is False
        assert result.status_code == 409
        assert "transporteur externe" in (result.error or "").lower()
        db.session.refresh(tr)
        assert tr.booking_id is None
        assert Booking.query.filter(Booking.id == tr.booking_id).count() == 0


class TestExternalCarrierBillingGuard:
    def test_external_completed_has_no_booking(self, db, institution):
        tr = _transport_request(
            db,
            institution,
            status=RequestStatus.EXTERNAL_DECLARED_COMPLETED.value,
            carrier_source=CarrierSource.EXTERNAL.value,
        )
        tr.external_carrier_name = "Taxi XYZ"
        tr.executed_externally_at = datetime.now(UTC)
        db.session.commit()

        assert tr.booking_id is None
        linked = (
            db.session.query(Booking)
            .join(TransportRequest, TransportRequest.booking_id == Booking.id)
            .filter(TransportRequest.id == tr.id)
            .count()
        )
        assert linked == 0
