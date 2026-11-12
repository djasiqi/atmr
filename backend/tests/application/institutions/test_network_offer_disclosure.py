"""Divulgation progressive des offres réseau, sans géocodage externe."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from flask_jwt_extended import create_access_token

from application.institutions.network_offer_disclosure import (
    confirmed_zone_label,
    indicative_route_hint,
    travel_estimate_for_offer,
)
from models import (
    Company,
    Institution,
    InstitutionPatient,
    OfferMode,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
    TransportRequestLeg,
    User,
    UserRole,
)
from models.enums import GeoUnitType

_STREET = "Chemin Confidentiel 9, 1247, Anières"
_DROP = "Service Secret, Hôpital Fictif, Dr Martin"
_DOCTOR = "Dr Martin Confidentiel"
_PATIENT = "Patientconfidentiel"
_LAT = "46.1234567"


def _zone(name: str, unit_type: GeoUnitType):
    return SimpleNamespace(name=name, type=unit_type, parent=None)


def test_confirmed_zone_uses_referential_name():
    commune = _zone("Anières", GeoUnitType.COMMUNE)
    canton = _zone("Genève", GeoUnitType.CANTON)
    with (
        patch(
            "application.institutions.network_offer_disclosure._local_unit",
            return_value=commune,
        ),
        patch(
            "application.institutions.network_offer_disclosure.geo_chain",
            return_value=[commune, canton],
        ),
    ):
        assert confirmed_zone_label(_STREET) == "Anières, Genève"


def test_unresolved_zone_is_not_a_truncated_street():
    with patch(
        "application.institutions.network_offer_disclosure._local_unit",
        return_value=None,
    ):
        assert confirmed_zone_label(_STREET) is None


def test_indicative_hint_uses_stored_coords_without_google():
    request = SimpleNamespace(
        legs=[],
        pickup_lat=46.2,
        pickup_lng=6.1,
        dropoff_lat=46.3,
        dropoff_lng=6.2,
    )
    with patch(
        "services.geolocation.google_directions.fetch_directions",
        side_effect=AssertionError("google"),
    ):
        hint = indicative_route_hint(request)
    assert hint["source"] == "vol_oiseau"
    assert hint["distance_km"] >= 1
    assert hint["travel_minutes"] >= 5
    assert "vol d'oiseau" in hint["distance_notice"]
    assert "routière" in hint["distance_notice"]
    dumped = json.dumps(hint)
    assert "46.2" not in dumped
    assert "6.1" not in dumped


def test_indicative_hint_without_coords_does_not_geocode():
    request = SimpleNamespace(
        legs=[],
        pickup_lat=None,
        pickup_lng=None,
        dropoff_lat=None,
        dropoff_lng=None,
    )
    with patch(
        "services.geolocation.google_places.geocode_address_google",
        side_effect=AssertionError("geocode"),
    ):
        hint = indicative_route_hint(request)
    assert hint == {
        "distance_km": None,
        "travel_minutes": None,
        "source": "unavailable",
    }


def _company(db, name: str) -> tuple[Company, User]:
    user = User()
    user.email = f"{name}_{uuid.uuid4().hex[:8]}@test.com"
    user.username = user.email
    user.password = "test"
    user.role = UserRole.COMPANY.value
    db.session.add(user)
    db.session.flush()
    company = Company()
    company.name = name
    company.user_id = user.id
    company.is_approved = True
    company.dispatch_enabled = True
    db.session.add(company)
    db.session.flush()
    return company, user


def _headers(company: Company, user: User) -> dict[str, str]:
    token = create_access_token(
        identity=str(user.public_id),
        additional_claims={
            "role": UserRole.COMPANY.value,
            "aud": "atmr-api",
            "user_id": user.id,
            "company_id": company.id,
        },
    )
    return {"Authorization": f"Bearer {token}"}


def _network_offer(db, company: Company) -> RequestOffer:
    institution = Institution()
    institution.name = "Clinique Fictive Confidentialité"
    institution.public_id = str(uuid.uuid4())
    db.session.add(institution)
    db.session.flush()

    patient = InstitutionPatient()
    patient.institution_id = institution.id
    patient.first_name = "Ada"
    patient.last_name = _PATIENT
    patient.dob = date(1948, 3, 15)
    patient.external_reference = "PAT-SECRET"
    db.session.add(patient)
    db.session.flush()

    scheduled = datetime.now(UTC) + timedelta(days=2)
    request = TransportRequest()
    request.institution_id = institution.id
    request.patient_id = patient.id
    request.external_reference = "REF-SECRET"
    request.pickup_location = _STREET
    request.dropoff_location = _DROP
    request.pickup_lat = _LAT
    request.pickup_lng = "6.1234567"
    request.dropoff_lat = "46.2234567"
    request.dropoff_lng = "6.2234567"
    request.notes = "Note médicale secrète"
    request.contact_on_site = {
        "name": "Infirmière Secrète",
        "phone": "0790000000",
    }
    request.mission_date = scheduled.date()
    request.scheduled_time = scheduled
    request.status = RequestStatus.SENT.value
    request.dispatch_target = "network"
    db.session.add(request)
    db.session.flush()

    leg = TransportRequestLeg()
    leg.transport_request_id = request.id
    leg.sequence_index = 0
    leg.route_sequence_number = 1
    leg.pickup_location = _STREET
    leg.dropoff_location = _DROP
    leg.pickup_lat = _LAT
    leg.pickup_lng = "6.1234567"
    leg.dropoff_lat = "46.2234567"
    leg.dropoff_lng = "6.2234567"
    leg.dropoff_establishment = "Hôpital Fictif"
    leg.dropoff_service = "Service Secret"
    leg.dropoff_doctor = _DOCTOR
    leg.scheduled_time = scheduled
    leg.time_confirmed = True
    db.session.add(leg)

    offer = RequestOffer(
        transport_request_id=request.id,
        company_id=company.id,
        mode=OfferMode.BROADCAST.value,
        status=OfferStatus.PENDING.value,
    )
    db.session.add(offer)
    db.session.flush()
    return offer


_FORBIDDEN = (
    _STREET,
    "Chemin Confidentiel",
    _DROP,
    _DOCTOR,
    "Hôpital Fictif",
    "Service Secret",
    _PATIENT,
    "PAT-SECRET",
    "REF-SECRET",
    "Note médicale secrète",
    "Infirmière Secrète",
    _LAT,
    "6.1234567",
)


def _assert_redacted(payload: dict) -> None:
    dumped = json.dumps(payload, default=str, ensure_ascii=False)
    for secret in _FORBIDDEN:
        assert secret not in dumped
    view = payload["transport_request"]
    assert view["disclosure"] == "network_pending"
    assert view["pickup_location"] is None
    assert view["dropoff_location"] is None
    assert view["pickup_lat"] is None
    assert view["patient"] is None
    assert payload["price_estimate"] is None
    assert view["scheduled_time"]
    leg = view["legs"][0]
    assert leg["pickup_location"] is None
    assert leg["dropoff_doctor"] is None
    assert leg["dropoff_establishment"] is None
    assert leg["scheduled_time"]


@pytest.fixture
def network_pending(db):
    company, user = _company(db, "Transporteur fictif confidentialité")
    offer = _network_offer(db, company)
    return offer, company, user


def test_pending_network_payload_hides_precise_itinerary(db, network_pending):
    offer, _company_row, _user = network_pending
    with patch(
        "services.pricing.offer_price_estimator.estimate_offer_price",
        side_effect=AssertionError("tarif"),
    ):
        payload = offer.serialize_for_company()
    _assert_redacted(payload)


def test_accepted_network_offer_keeps_precise_itinerary(db, network_pending):
    offer, _company_row, _user = network_pending
    offer.status = OfferStatus.ACCEPTED.value
    db.session.flush()
    with patch(
        "services.pricing.offer_price_estimator.estimate_offer_price",
        return_value={"amount": 10, "currency": "CHF", "source": "test"},
    ):
        payload = offer.serialize_for_company()
    view = payload["transport_request"]
    assert view["pickup_location"] == _STREET
    assert view["legs"][0]["dropoff_doctor"] == _DOCTOR
    assert _PATIENT in view["patient_name"]
    assert view.get("disclosure") is None


def test_legacy_offer_without_dispatch_target_keeps_itinerary(db, network_pending):
    offer, _company_row, _user = network_pending
    offer.transport_request.dispatch_target = None
    db.session.flush()
    with patch(
        "services.pricing.offer_price_estimator.estimate_offer_price",
        return_value={"amount": 10, "currency": "CHF", "source": "test"},
    ):
        payload = offer.serialize_for_company()
    assert payload["transport_request"]["pickup_location"] == _STREET


def test_pending_travel_estimate_does_not_call_google(db, network_pending):
    offer, _company_row, _user = network_pending
    with (
        patch(
            "services.geolocation.google_directions.fetch_directions",
            side_effect=AssertionError("directions"),
        ),
        patch(
            "services.geolocation.google_places.geocode_address_google",
            side_effect=AssertionError("geocode"),
        ),
        patch(
            "services.geolocation.maps.get_distance_duration",
            side_effect=AssertionError("matrix"),
        ),
    ):
        payload = travel_estimate_for_offer(offer)
    dumped = json.dumps(payload)
    assert "lat" not in payload
    assert "lng" not in payload
    assert _LAT not in dumped
    assert payload["source"] == "vol_oiseau"
    assert "vol d'oiseau" in payload["distance_notice"]
    assert set(payload) == {
        "travel_minutes",
        "distance_km",
        "source",
        "distance_notice",
    }


def test_foreign_company_cannot_read_offer_or_estimate(client, db, network_pending):
    offer, _owner, _owner_user = network_pending
    stranger, stranger_user = _company(db, "Entreprise étrangère")
    headers = _headers(stranger, stranger_user)
    detail = client.get(f"/api/v1/company/request-offers/{offer.id}", headers=headers)
    estimate = client.get(
        f"/api/v1/company/request-offers/{offer.id}/travel-estimate",
        headers=headers,
    )
    assert detail.status_code == 403
    assert estimate.status_code == 403
    assert _STREET not in detail.get_data(as_text=True)
    assert _LAT not in estimate.get_data(as_text=True)


def test_recipient_list_detail_and_estimate_stay_redacted(client, db, network_pending):
    offer, company, user = network_pending
    headers = _headers(company, user)
    with (
        patch(
            "services.pricing.offer_price_estimator.estimate_offer_price",
            side_effect=AssertionError("tarif"),
        ),
        patch(
            "services.geolocation.google_directions.fetch_directions",
            side_effect=AssertionError("directions"),
        ),
        patch(
            "services.institutions.route_travel_estimate_service.estimate_outbound_travel_minutes",
            side_effect=AssertionError("estimation"),
        ),
    ):
        listing = client.get("/api/v1/company/request-offers", headers=headers)
        detail = client.get(
            f"/api/v1/company/request-offers/{offer.id}",
            headers=headers,
        )
        estimate = client.get(
            f"/api/v1/company/request-offers/{offer.id}/travel-estimate",
            headers=headers,
        )
    assert listing.status_code == 200
    assert detail.status_code == 200
    assert estimate.status_code == 200
    for response in (listing, detail, estimate):
        text = json.dumps(response.get_json(), ensure_ascii=False)
        for secret in _FORBIDDEN:
            assert secret not in text
    body = detail.get_json()
    _assert_redacted(body)
    estimate_body = estimate.get_json()
    assert set(estimate_body) <= {
        "travel_minutes",
        "distance_km",
        "source",
        "distance_notice",
    }
    assert "vol d'oiseau" in estimate_body["distance_notice"]
