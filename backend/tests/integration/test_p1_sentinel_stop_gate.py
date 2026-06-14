# ruff: noqa: I001
"""Tests stop-gate P1 — fin des écritures sentinelle 00:00 (Phase 1).

Couvre les cas critiques de validation staging automatisables :
- Cas 1 : A/R institution avec return_date seul → retour null/false
- Cas 2 : serialize() expose time_confirmed comme source de vérité
- Cas 3 : minuit réel confirmé → 00:00 + time_confirmed=true
- Cas 4 : multi-stop sans héritage horaire sur les legs suivants
- Cas 5 : audit invariant DB — aucune nouvelle sentinelle sur les IDs créés

Ces tests nécessitent PostgreSQL (JSONB, date_part). Ils complètent les tests
unitaires existants sans les dupliquer.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import func

from application.institutions.accept_offer import AcceptOfferUseCase
from models import Booking, Institution, TransportRequest, TransportRequestLeg
from models.enums import RequestStatus
from shared.time_utils import now_local


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"EMS Stop-Gate P1 {uuid.uuid4().hex[:6]}"
    inst.institution_type = "ems"
    db.session.add(inst)
    db.session.flush()
    return inst


def _base_transport_request(
    db,
    institution: Institution,
    *,
    scheduled_time: datetime,
    is_round_trip: bool = False,
    return_date: date | None = None,
    return_time: datetime | None = None,
    return_time_confirmed: bool = False,
    multi_stop: bool = False,
    route_group_id: str | None = None,
) -> TransportRequest:
    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.institution = institution
    tr.external_reference = f"SG-P1-{uuid.uuid4().hex[:8]}"
    tr.pickup_location = "Clinique A, 1200 Genève"
    tr.dropoff_location = "Hôpital B, 1205 Genève"
    tr.scheduled_time = scheduled_time
    tr.mission_date = scheduled_time.date() if scheduled_time else None
    tr.status = RequestStatus.SENT.value
    tr.is_round_trip = is_round_trip
    tr.return_date = return_date
    tr.return_time = return_time
    tr.return_time_confirmed = return_time_confirmed
    tr.multi_stop = multi_stop
    tr.route_group_id = route_group_id
    tr.billing_intent = "patient"
    db.session.add(tr)
    db.session.flush()
    return tr


def _future_depart(hour: int = 8, minute: int = 0) -> datetime:
    """Horaire futur en heure locale Genève (aligné accept_offer)."""
    base = now_local() + timedelta(days=3)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _assert_no_sentinel_on_ids(db, created_ids: list[int]) -> None:
    """Aucun booking créé ne doit être sentinelle 00:00 + time_confirmed=false."""
    if not created_ids:
        return
    offenders = (
        db.session.query(Booking.id)
        .filter(
            Booking.id.in_(created_ids),
            Booking.time_confirmed.is_(False),
            Booking.scheduled_time.isnot(None),
            func.extract("hour", Booking.scheduled_time) == 0,
            func.extract("minute", Booking.scheduled_time) == 0,
        )
        .all()
    )
    assert offenders == [], (
        f"Sentinelle 00:00 détectée sur les bookings créés : "
        f"{[row[0] for row in offenders]}"
    )


# ---------------------------------------------------------------------------
# Cas 1 & 2 — A/R date seule + serialize()
# ---------------------------------------------------------------------------


class TestCas1RoundTripReturnDateOnly:
    """Institution A/R : return_date seul → booking retour null/false."""

    def test_return_booking_null_unconfirmed(
        self, db, requires_postgresql, institution, test_company, test_client
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        outbound_day = _future_depart(8, 0)
        tr = _base_transport_request(
            db,
            institution,
            scheduled_time=outbound_day,
            is_round_trip=True,
            return_date=outbound_day.date(),
            return_time=None,
            return_time_confirmed=False,
        )

        uc = AcceptOfferUseCase()
        outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        db.session.flush()

        assert outbound is not None
        assert return_booking is not None
        assert return_booking.scheduled_time is None
        assert return_booking.time_confirmed is False
        assert return_booking.is_return is True
        assert return_booking.parent_booking_id == outbound.id


class TestCas2SerializeSourceOfTruth:
    """serialize() expose time_confirmed — source de vérité pour Phase 2."""

    def test_return_booking_serialize_flags(
        self, db, requires_postgresql, institution, test_company, test_client
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        depart = _future_depart(8, 0)
        tr = _base_transport_request(
            db,
            institution,
            scheduled_time=depart,
            is_round_trip=True,
            return_date=depart.date(),
            return_time_confirmed=False,
        )

        uc = AcceptOfferUseCase()
        _outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        assert return_booking is not None

        payload = return_booking.serialize
        assert payload["time_confirmed"] is False
        assert payload["scheduled_time"] is None


# ---------------------------------------------------------------------------
# Cas 3 — Minuit réel confirmé
# ---------------------------------------------------------------------------


class TestCas3RealMidnightConfirmed:
    """Minuit réel : scheduled_time=00:00 + time_confirmed=true."""

    def test_midnight_return_confirmed(
        self, db, requires_postgresql, institution, test_company, test_client
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        outbound = _future_depart(14, 0)
        midnight = outbound.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days=1
        )
        tr = _base_transport_request(
            db,
            institution,
            scheduled_time=outbound,
            is_round_trip=True,
            return_time=midnight,
            return_time_confirmed=True,
        )

        uc = AcceptOfferUseCase()
        _outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        assert return_booking is not None
        assert return_booking.time_confirmed is True
        assert return_booking.scheduled_time is not None
        assert return_booking.scheduled_time.hour == 0
        assert return_booking.scheduled_time.minute == 0

        payload = return_booking.serialize
        assert payload["time_confirmed"] is True
        assert payload["scheduled_time"] is not None


# ---------------------------------------------------------------------------
# Cas 4 — Multi-stop sans héritage horaire
# ---------------------------------------------------------------------------


class TestCas4MultiStopNoInheritance:
    """Leg 0 confirmé ; legs suivants null/false, sans héritage de 08:00."""

    def test_subsequent_legs_null_not_inherited(
        self, db, requires_postgresql, institution, test_company, test_client
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        route_group_id = str(uuid.uuid4())
        depart_at = _future_depart(8, 0)
        tr = _base_transport_request(
            db,
            institution,
            scheduled_time=depart_at,
            multi_stop=True,
            route_group_id=route_group_id,
        )
        tr.return_to_institution = True
        tr.pickup_time_confirmed = True

        legs = [
            TransportRequestLeg(
                transport_request_id=tr.id,
                sequence_index=0,
                route_sequence_number=1,
                pickup_location="Clinique",
                dropoff_location="HUG",
                scheduled_time=None,
            ),
            TransportRequestLeg(
                transport_request_id=tr.id,
                sequence_index=1,
                route_sequence_number=2,
                pickup_location="HUG",
                dropoff_location="Grangettes",
                scheduled_time=None,
            ),
            TransportRequestLeg(
                transport_request_id=tr.id,
                sequence_index=2,
                route_sequence_number=3,
                pickup_location="Grangettes",
                dropoff_location="Clinique",
                scheduled_time=None,
            ),
        ]
        db.session.add_all(legs)
        db.session.flush()

        uc = AcceptOfferUseCase()
        primary, _return_booking = uc._create_bookings_from_legs(
            transport_request=tr,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        db.session.flush()

        created = (
            Booking.query.filter_by(route_group_id=route_group_id)
            .order_by(Booking.route_sequence_number.asc())
            .all()
        )
        assert len(created) == 3
        by_seq = {b.route_sequence_number: b for b in created}

        first = by_seq[1]
        assert first.scheduled_time is not None
        assert first.scheduled_time.hour == depart_at.hour
        assert first.scheduled_time.minute == depart_at.minute
        assert first.time_confirmed is True
        assert primary is not None
        assert primary.id == first.id

        for seq in (2, 3):
            leg_booking = by_seq[seq]
            assert leg_booking.scheduled_time is None
            assert leg_booking.time_confirmed is False
            # Pas d'héritage : l'heure du leg 0 ne doit pas apparaître sur les suivants
            assert leg_booking.scheduled_time != depart_at.replace(
                hour=8, minute=0, second=0, microsecond=0
            ) or leg_booking.scheduled_time is None


# ---------------------------------------------------------------------------
# Cas 5 — Audit invariant DB (filtrage strict par IDs créés)
# ---------------------------------------------------------------------------


class TestCas5AuditNoNewSentinel:
    """Aucun booking créé par ce test ne doit être sentinelle 00:00 + false."""

    def test_no_sentinel_on_all_created_booking_ids(
        self, db, requires_postgresql, institution, test_company, test_client
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        uc = AcceptOfferUseCase()
        created_ids: list[int] = []

        # A/R date seule
        day1 = _future_depart(8, 0)
        tr1 = _base_transport_request(
            db,
            institution,
            scheduled_time=day1,
            is_round_trip=True,
            return_date=day1.date(),
            return_time_confirmed=False,
        )
        out1, ret1 = uc._create_booking_from_request(
            transport_request=tr1,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        created_ids.extend([out1.id, ret1.id] if ret1 else [out1.id])

        # Minuit réel confirmé
        out_day = _future_depart(14, 0)
        midnight = out_day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days=1
        )
        tr2 = _base_transport_request(
            db,
            institution,
            scheduled_time=out_day,
            is_round_trip=True,
            return_time=midnight,
            return_time_confirmed=True,
        )
        out2, ret2 = uc._create_booking_from_request(
            transport_request=tr2,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        created_ids.extend([out2.id, ret2.id] if ret2 else [out2.id])

        # Multi-stop
        route_group_id = str(uuid.uuid4())
        tr3 = _base_transport_request(
            db,
            institution,
            scheduled_time=_future_depart(8, 0),
            multi_stop=True,
            route_group_id=route_group_id,
        )
        tr3.return_to_institution = True
        db.session.add_all(
            [
                TransportRequestLeg(
                    transport_request_id=tr3.id,
                    sequence_index=0,
                    route_sequence_number=1,
                    pickup_location="Clinique",
                    dropoff_location="HUG",
                ),
                TransportRequestLeg(
                    transport_request_id=tr3.id,
                    sequence_index=1,
                    route_sequence_number=2,
                    pickup_location="HUG",
                    dropoff_location="Clinique",
                ),
            ]
        )
        db.session.flush()
        uc._create_bookings_from_legs(
            transport_request=tr3,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        leg_bookings = Booking.query.filter_by(route_group_id=route_group_id).all()
        created_ids.extend(b.id for b in leg_bookings)

        db.session.flush()
        assert len(created_ids) >= 6

        _assert_no_sentinel_on_ids(db, created_ids)
