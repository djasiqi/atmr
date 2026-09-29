"""Besoins habituels du profil : persistés, sans réécrire une course déjà créée."""

from __future__ import annotations

from datetime import datetime

import pytest

from application.companies.clients.update_company_client import (
    UpdateCompanyClientUseCase,
)
from models.booking import Booking
from models.client import Client
from models.enums import BookingStatus
from tests.services.test_client_booking_contract_event import _portal_user


@pytest.mark.integration
def test_habitual_mobility_persists_without_rewriting_bookings(db) -> None:
    user, client = _portal_user(db)
    booking = Booking()
    booking.customer_name = "Jeanne Martin"
    booking.pickup_location = "Rue du Test 1"
    booking.dropoff_location = "HUG"
    booking.time_confirmed = False
    booking.scheduled_time = datetime(2026, 9, 30, 9, 0, 0)
    booking.amount = 40.0
    booking.status = BookingStatus.PENDING
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.pickup_access_notes = "3e étage\nInterphone Osmani"
    booking.wheelchair_client_has = False
    booking.needs_assistance = False
    db.session.add(booking)
    db.session.flush()

    rejected = UpdateCompanyClientUseCase().execute(
        client=client,
        data={
            "habitual_wheelchair_client_has": True,
            "habitual_wheelchair_need": True,
        },
    )
    assert rejected.ok is False
    assert client.habitual_wheelchair_client_has is False

    missing_detail = UpdateCompanyClientUseCase().execute(
        client=client,
        data={
            "habitual_wheelchair_client_has": True,
            "habitual_needs_assistance": True,
            "habitual_assistance_detail": "   ",
        },
    )
    assert missing_detail.ok is False
    assert client.habitual_needs_assistance is False

    saved = UpdateCompanyClientUseCase().execute(
        client=client,
        data={
            "habitual_wheelchair_client_has": True,
            "habitual_needs_assistance": True,
            "habitual_assistance_detail": "Aide à la marche",
            "floor": "9e",
            "door_code": "Autre",
        },
    )
    assert saved.ok is True
    db.session.flush()
    db.session.expire_all()

    reloaded_client = db.session.get(Client, client.id)
    assert reloaded_client is not None
    mobility = reloaded_client.serialize["mobility"]
    assert mobility["wheelchair_client_has"] is True
    assert mobility["wheelchair_need"] is False
    assert mobility["needs_assistance"] is True
    assert mobility["assistance_detail"] == "Aide à la marche"
    assert reloaded_client.floor == "9e"

    reloaded_booking = db.session.get(Booking, booking.id)
    assert reloaded_booking is not None
    assert reloaded_booking.pickup_access_notes == "3e étage\nInterphone Osmani"
    assert reloaded_booking.wheelchair_client_has is False
    assert reloaded_booking.needs_assistance is False
