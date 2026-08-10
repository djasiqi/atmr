"""CreateManualBooking : rattachement billing_party_id (AR + récurrence + tiers)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from application.companies.reservations.create_manual_booking import (
    CreateManualBookingUseCase,
)
from models import BillingParty, Client, ClientBillingParty, Company, User
from models.enums import BillingPartyType, UserRole


def _tomorrow_at(hour: int) -> datetime:
    """Construit un horaire futur, indépendant de la date d'exécution."""
    return (datetime.now() + timedelta(days=1)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )


def _company_and_client(db):
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"co_{suffix}"
    user.email = f"co-{suffix}@test.ch"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.name = f"Co {suffix}"
    company.address = "Rue 1"
    company.contact_email = f"c-{suffix}@test.ch"
    company.user_id = user.id
    db.session.add(company)
    db.session.flush()

    cu = User()
    cu.username = f"cli_{suffix}"
    cu.email = f"cli-{suffix}@test.ch"
    cu.role = UserRole.client
    cu.first_name = "Ada"
    cu.last_name = "Lovelace"
    cu.public_id = str(uuid.uuid4())
    cu.set_password("password123", force_change=False)
    db.session.add(cu)
    db.session.flush()

    client = Client()
    client.user_id = cu.id
    client.company_id = company.id
    client.domicile_address = "Rue Ada 1"
    client.domicile_zip = "1201"
    client.domicile_city = "Genève"
    client.default_billed_to_type = "patient"
    db.session.add(client)
    db.session.flush()
    return company, client, cu


@pytest.fixture
def company_client(db):
    return _company_and_client(db)


def _execute_manual(company, client, user, validated_data):
    with (
        patch(
            "services.platform_billing.capabilities.assert_billing_capability_allowed"
        ),
        patch(
            "services.billing.client_stay_resolver.find_active_stay_for_client",
            return_value=None,
        ),
        patch(
            "application.companies.reservations.create_manual_booking._geocode_with_nominatim",
            return_value=(46.2, 6.1),
        ),
        patch("services.geolocation.osrm._route") as mock_route,
    ):
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 4000}],
        }
        return CreateManualBookingUseCase().execute(
            company_id=company.id,
            validated_data=validated_data,
            client=client,
            user=user,
        )


def test_manual_one_way_assigns_patient_billing_party(db, company_client):
    company, client, user = company_client
    scheduled_at = _tomorrow_at(10)
    result = _execute_manual(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "A",
            "dropoff_location": "B",
            "pickup_lat": 46.2,
            "pickup_lon": 6.1,
            "dropoff_lat": 46.21,
            "dropoff_lon": 6.12,
            "scheduled_time": scheduled_at.isoformat(),
            "amount": 45.0,
            "amount_source": "manual",
        },
    )

    assert len(result.created_outbounds) == 1
    outbound = result.created_outbounds[0]
    assert outbound.billing_party_id is not None
    bp = db.session.get(BillingParty, outbound.billing_party_id)
    assert bp is not None
    assert bp.type == BillingPartyType.PATIENT
    assert bp.external_ref == f"patient_client:{client.id}"


def test_manual_round_trip_shares_billing_party(db, company_client):
    company, client, user = company_client
    scheduled_at = _tomorrow_at(10)
    result = _execute_manual(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "A",
            "dropoff_location": "B",
            "pickup_lat": 46.2,
            "pickup_lon": 6.1,
            "dropoff_lat": 46.21,
            "dropoff_lon": 6.12,
            "scheduled_time": scheduled_at.isoformat(),
            "is_round_trip": True,
            "return_date": scheduled_at.date().isoformat(),
            "return_time": "14:00",
            "amount": 90.0,
            "amount_source": "manual",
        },
    )

    assert len(result.created_outbounds) == 1
    assert len(result.created_returns) == 1
    assert (
        result.created_outbounds[0].billing_party_id
        == result.created_returns[0].billing_party_id
    )
    assert result.created_outbounds[0].billing_party_id is not None


def test_manual_recurrence_shares_billing_party(db, company_client):
    company, client, user = company_client
    scheduled_at = _tomorrow_at(10)
    result = _execute_manual(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "A",
            "dropoff_location": "B",
            "pickup_lat": 46.2,
            "pickup_lon": 6.1,
            "dropoff_lat": 46.21,
            "dropoff_lon": 6.12,
            "scheduled_time": scheduled_at.isoformat(),
            "is_recurring": True,
            "recurrence_type": "daily",
            "occurrences": 3,
            "amount": 45.0,
            "amount_source": "manual",
        },
    )

    assert len(result.created_outbounds) == 3
    ids = {b.billing_party_id for b in result.created_outbounds}
    assert len(ids) == 1
    assert None not in ids


def test_manual_uses_third_party_when_configured(db, company_client):
    company, client, user = company_client
    curator = BillingParty()
    curator.company_id = company.id
    curator.type = BillingPartyType.CURATORSHIP
    curator.display_name = "Curatelle"
    curator.billing_address = "Rue C 1\n1200 Genève"
    curator.is_active = True
    db.session.add(curator)
    db.session.flush()
    link = ClientBillingParty()
    link.client_id = client.id
    link.billing_party_id = curator.id
    link.is_default = True
    db.session.add(link)
    db.session.flush()

    scheduled_at = _tomorrow_at(10)
    result = _execute_manual(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "A",
            "dropoff_location": "B",
            "pickup_lat": 46.2,
            "pickup_lon": 6.1,
            "dropoff_lat": 46.21,
            "dropoff_lon": 6.12,
            "scheduled_time": scheduled_at.isoformat(),
            "amount": 45.0,
            "amount_source": "manual",
        },
    )

    assert result.created_outbounds[0].billing_party_id == curator.id
