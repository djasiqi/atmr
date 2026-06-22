"""Tests E2E : Parité web/mobile création de réservation manuelle.

Vérifie que web et mobile produisent le même résultat métier via le même
use-case CreateManualBookingUseCase.

Cas couverts :
- Aller simple : mêmes champs Booking
- Aller-retour : idem
- Notes : notes_medical correctement persisté
- Coordonnées GPS : pickup_lat/lon, dropoff_lat/lon
- Récurrence : même nombre de courses, mêmes dates
- Alias : pickup_address/dropoff_address/is_return convertis sans perte

Requiert PostgreSQL + RUN_E2E_TESTS=1.
Exécution : RUN_E2E_TESTS=1 pytest tests/e2e/test_manual_booking_web_mobile_parity.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from models import Booking
from tests.e2e.helpers.e2e_helpers import (
    create_test_client,
    create_test_company,
)


def _company_headers(app, company):
    from flask_jwt_extended import create_access_token

    from models import User, UserRole

    user = (
        getattr(company, "user", None)
        or User.query.filter_by(id=company.user_id).first()
    )
    claims = {
        "role": UserRole.company.value,
        "company_id": company.id,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _requires_postgres(db_session):
    """Skip si la DB est SQLite."""
    url = str(db_session.engine.url)
    if "sqlite" in url.lower():
        pytest.skip("Parity E2E tests require PostgreSQL (got SQLite)")


def _base_canonical_payload(client_id: int, scheduled_time: str):
    """Payload canonique minimal (web)."""
    return {
        "client_id": client_id,
        "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
        "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
        "pickup_lat": 46.5197,
        "pickup_lon": 6.6323,
        "dropoff_lat": 46.5160,
        "dropoff_lon": 6.6328,
        "scheduled_time": scheduled_time,
    }


def _base_mobile_payload(client_id: int, scheduled_time: str):
    """Payload mobile équivalent (alias pickup_address, is_return)."""
    return {
        "client_id": client_id,
        "pickup_address": "Rue de la Gare 1, 1000 Lausanne",
        "dropoff_address": "Avenue de la Plage 10, 1000 Lausanne",
        "pickup_lat": 46.5197,
        "pickup_lon": 6.6323,
        "dropoff_lat": 46.5160,
        "dropoff_lon": 6.6328,
        "scheduled_time": scheduled_time,
    }


@patch(
    "application.companies.reservations.create_manual_booking._geocode_with_nominatim"
)
@patch("services.geolocation.osrm._route")
@pytest.mark.e2e
class TestManualBookingWebMobileParity:
    """Parité web/mobile pour création de réservation."""

    def test_case1_simple_one_way_same_fields(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Cas 1 : Aller simple — web et mobile → mêmes champs Booking."""
        _requires_postgres(db)
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 5000}],
        }
        mock_geocode.return_value = (46.5197, 6.6323)

        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.commit()
        db.session.refresh(customer)

        scheduled = (datetime.now(UTC) + timedelta(days=1)).strftime(
            "%Y-%m-%dT10:00:00"
        )
        canonical = _base_canonical_payload(customer.id, scheduled)
        mobile = _base_mobile_payload(customer.id, scheduled)

        headers = _company_headers(app, company)

        # Web
        r_web = client.post(
            "/api/v1/companies/me/reservations/manual",
            json=canonical,
            headers=headers,
        )
        assert r_web.status_code == 201, (r_web.status_code, r_web.get_json())

        web_data = r_web.get_json()
        web_reservations = web_data.get("reservations") or []
        assert len(web_reservations) >= 1
        web_booking_id = web_reservations[0].get("id")

        # Mobile (même company, nouveau client pour éviter conflit de dates)
        company2 = create_test_company(db)
        customer2 = create_test_client(db, company=company2)
        db.session.commit()
        db.session.refresh(customer2)

        mobile["client_id"] = customer2.id
        mobile["scheduled_time"] = (
            datetime.now(UTC) + timedelta(days=1, hours=1)
        ).strftime("%Y-%m-%dT11:00:00")
        headers2 = _company_headers(app, company2)

        r_mobile = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=mobile,
            headers=headers2,
        )
        assert r_mobile.status_code == 201, (r_mobile.status_code, r_mobile.get_json())

        mobile_data = r_mobile.get_json()
        mobile_summary = mobile_data.get("summary")
        assert mobile_summary is not None
        mobile_booking_id = mobile_summary.get("id")

        # Comparer les Booking en DB
        web_booking = db.session.get(Booking, web_booking_id)
        mobile_booking = db.session.get(Booking, mobile_booking_id)
        assert web_booking is not None
        assert mobile_booking is not None

        fields_to_compare = [
            "pickup_location",
            "dropoff_location",
            "pickup_lat",
            "pickup_lon",
            "dropoff_lat",
            "dropoff_lon",
            "client_id",
            "company_id",
            "status",
        ]
        for f in fields_to_compare:
            v_web = getattr(web_booking, f, None)
            v_mobile = getattr(mobile_booking, f, None)
            assert v_web == v_mobile, (
                f"Champ {f} diverge: web={v_web!r} vs mobile={v_mobile!r}"
            )

    def test_case2_round_trip_same_fields(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Cas 2 : Aller-retour — web et mobile → mêmes champs."""
        _requires_postgres(db)
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 5000}],
        }
        mock_geocode.return_value = (46.5197, 6.6323)

        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.commit()
        db.session.refresh(customer)

        scheduled = (datetime.now(UTC) + timedelta(days=1)).strftime(
            "%Y-%m-%dT10:00:00"
        )
        return_time = (datetime.now(UTC) + timedelta(days=1, hours=3)).strftime(
            "%Y-%m-%dT13:00:00"
        )

        canonical = _base_canonical_payload(customer.id, scheduled)
        canonical["is_round_trip"] = True
        canonical["return_time"] = return_time

        mobile = _base_mobile_payload(customer.id, scheduled)
        mobile["is_return"] = True
        mobile["return_time"] = return_time

        headers = _company_headers(app, company)

        r_web = client.post(
            "/api/v1/companies/me/reservations/manual",
            json=canonical,
            headers=headers,
        )
        assert r_web.status_code == 201, (r_web.status_code, r_web.get_json())
        web_data = r_web.get_json()
        assert web_data.get("return_bookings")
        assert len(web_data["return_bookings"]) >= 1

        company2 = create_test_company(db)
        customer2 = create_test_client(db, company=company2)
        db.session.commit()
        db.session.refresh(customer2)

        mobile["client_id"] = customer2.id
        mobile["scheduled_time"] = (
            datetime.now(UTC) + timedelta(days=1, hours=1)
        ).strftime("%Y-%m-%dT11:00:00")
        mobile["return_time"] = (
            datetime.now(UTC) + timedelta(days=1, hours=4)
        ).strftime("%Y-%m-%dT15:00:00")
        headers2 = _company_headers(app, company2)

        r_mobile = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=mobile,
            headers=headers2,
        )
        assert r_mobile.status_code == 201, (r_mobile.status_code, r_mobile.get_json())
        mobile_data = r_mobile.get_json()
        assert mobile_data.get("return_summary") is not None

    def test_case2b_round_trip_return_date_only_creates_return(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Cas 2b : A/R avec return_date seul (heure à définir) → retour créé côté mobile."""
        _requires_postgres(db)
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 5000}],
        }
        mock_geocode.return_value = (46.5197, 6.6323)

        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.commit()
        db.session.refresh(customer)

        scheduled = (datetime.now(UTC) + timedelta(days=1)).strftime(
            "%Y-%m-%dT11:00:00"
        )
        return_date = scheduled.split("T")[0]

        mobile = _base_mobile_payload(customer.id, scheduled)
        mobile["is_return"] = True
        mobile["return_date"] = return_date

        headers = _company_headers(app, company)
        r_mobile = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=mobile,
            headers=headers,
        )
        assert r_mobile.status_code == 201, (r_mobile.status_code, r_mobile.get_json())
        mobile_data = r_mobile.get_json()
        assert mobile_data.get("return_summary") is not None

        from models.booking import Booking

        return_id = mobile_data["return_summary"]["id"]
        return_booking = db.session.get(Booking, int(return_id))
        assert return_booking is not None
        assert return_booking.is_return is True
        assert return_booking.scheduled_time is None
        assert return_booking.time_confirmed is False

    def test_case3_notes_medical_persisted(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Cas 3 : notes_medical correctement persisté."""
        _requires_postgres(db)
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 5000}],
        }
        mock_geocode.return_value = (46.5197, 6.6323)

        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.commit()
        db.session.refresh(customer)

        scheduled = (datetime.now(UTC) + timedelta(days=1)).strftime(
            "%Y-%m-%dT10:00:00"
        )
        notes_medical = "Patient sous anticoagulants, prévoir accompagnant"

        canonical = _base_canonical_payload(customer.id, scheduled)
        canonical["notes_medical"] = notes_medical

        mobile = _base_mobile_payload(customer.id, scheduled)
        mobile["notes_medical"] = notes_medical

        headers = _company_headers(app, company)

        r_web = client.post(
            "/api/v1/companies/me/reservations/manual",
            json=canonical,
            headers=headers,
        )
        assert r_web.status_code == 201, (r_web.status_code, r_web.get_json())
        web_reservations = r_web.get_json().get("reservations") or []
        web_booking_id = web_reservations[0].get("id")
        web_booking = db.session.get(Booking, web_booking_id)
        assert web_booking.notes_medical == notes_medical

        company2 = create_test_company(db)
        customer2 = create_test_client(db, company=company2)
        db.session.commit()
        db.session.refresh(customer2)

        mobile["client_id"] = customer2.id
        mobile["scheduled_time"] = (
            datetime.now(UTC) + timedelta(days=1, hours=1)
        ).strftime("%Y-%m-%dT11:00:00")
        headers2 = _company_headers(app, company2)

        r_mobile = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=mobile,
            headers=headers2,
        )
        assert r_mobile.status_code == 201, (r_mobile.status_code, r_mobile.get_json())
        mobile_summary = r_mobile.get_json().get("summary")
        mobile_booking_id = mobile_summary.get("id")
        mobile_booking = db.session.get(Booking, mobile_booking_id)
        assert mobile_booking.notes_medical == notes_medical

    def test_case4_gps_coords_persisted(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Cas 4 : pickup_lat/lon, dropoff_lat/lon correctement persistés."""
        _requires_postgres(db)
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 5000}],
        }

        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.commit()
        db.session.refresh(customer)

        scheduled = (datetime.now(UTC) + timedelta(days=1)).strftime(
            "%Y-%m-%dT10:00:00"
        )
        lat_p, lon_p = 46.5197, 6.6323
        lat_d, lon_d = 46.5160, 6.6328

        canonical = _base_canonical_payload(customer.id, scheduled)
        mobile = _base_mobile_payload(customer.id, scheduled)

        headers = _company_headers(app, company)

        r_web = client.post(
            "/api/v1/companies/me/reservations/manual",
            json=canonical,
            headers=headers,
        )
        assert r_web.status_code == 201, (r_web.status_code, r_web.get_json())
        web_reservations = r_web.get_json().get("reservations") or []
        web_booking = db.session.get(Booking, web_reservations[0].get("id"))
        assert web_booking.pickup_lat == lat_p
        assert web_booking.pickup_lon == lon_p
        assert web_booking.dropoff_lat == lat_d
        assert web_booking.dropoff_lon == lon_d

        company2 = create_test_company(db)
        customer2 = create_test_client(db, company=company2)
        db.session.commit()
        db.session.refresh(customer2)

        mobile["client_id"] = customer2.id
        mobile["scheduled_time"] = (
            datetime.now(UTC) + timedelta(days=1, hours=1)
        ).strftime("%Y-%m-%dT11:00:00")
        headers2 = _company_headers(app, company2)

        r_mobile = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=mobile,
            headers=headers2,
        )
        assert r_mobile.status_code == 201, (r_mobile.status_code, r_mobile.get_json())
        mobile_summary = r_mobile.get_json().get("summary")
        mobile_booking = db.session.get(Booking, mobile_summary.get("id"))
        assert mobile_booking.pickup_lat == lat_p
        assert mobile_booking.pickup_lon == lon_p
        assert mobile_booking.dropoff_lat == lat_d
        assert mobile_booking.dropoff_lon == lon_d

    def test_case5_recurrence_same_count_and_dates(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Cas 5 : Récurrence — même nombre de courses, mêmes dates (web et mobile)."""
        _requires_postgres(db)
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 5000}],
        }
        mock_geocode.return_value = (46.5197, 6.6323)

        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.commit()
        db.session.refresh(customer)

        scheduled = (datetime.now(UTC) + timedelta(days=1)).strftime(
            "%Y-%m-%dT10:00:00"
        )
        canonical = _base_canonical_payload(customer.id, scheduled)
        canonical["is_recurring"] = True
        canonical["recurrence_type"] = "weekly"
        canonical["occurrences"] = 3

        headers = _company_headers(app, company)

        r_web = client.post(
            "/api/v1/companies/me/reservations/manual",
            json=canonical,
            headers=headers,
        )
        assert r_web.status_code == 201, (r_web.status_code, r_web.get_json())
        web_data = r_web.get_json()
        web_reservations = web_data.get("reservations") or []
        web_return = web_data.get("return_bookings") or []
        web_total = len(web_reservations) + len(web_return)
        assert web_total >= 3, f"Web attendu >= 3 courses, obtenu {web_total}"

        company2 = create_test_company(db)
        customer2 = create_test_client(db, company=company2)
        db.session.commit()
        db.session.refresh(customer2)

        mobile = _base_mobile_payload(customer2.id, scheduled)
        mobile["scheduled_time"] = (
            datetime.now(UTC) + timedelta(days=1, hours=1)
        ).strftime("%Y-%m-%dT11:00:00")
        mobile["is_recurring"] = True
        mobile["recurrence_type"] = "weekly"
        mobile["occurrences"] = 3

        headers2 = _company_headers(app, company2)

        r_mobile = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=mobile,
            headers=headers2,
        )
        assert r_mobile.status_code == 201, (r_mobile.status_code, r_mobile.get_json())
        mobile_summary = r_mobile.get_json().get("summary")
        assert mobile_summary is not None
        mobile_booking_id = mobile_summary.get("id")
        mobile_booking = db.session.get(Booking, mobile_booking_id)
        assert mobile_booking is not None
        mobile_count = (
            db.session.query(Booking)
            .filter(
                Booking.client_id == customer2.id,
                Booking.company_id == company2.id,
            )
            .count()
        )
        assert mobile_count >= 3, (
            f"Mobile attendu >= 3 courses récurrentes, obtenu {mobile_count}"
        )

    def test_case6_alias_conversion_no_data_loss(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Cas 6 : Alias pickup_address/dropoff_address/is_return convertis sans perte."""
        _requires_postgres(db)
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 5000}],
        }

        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.commit()
        db.session.refresh(customer)

        scheduled = (datetime.now(UTC) + timedelta(days=1)).strftime(
            "%Y-%m-%dT10:00:00"
        )
        mobile = _base_mobile_payload(customer.id, scheduled)
        mobile["is_return"] = False

        headers = _company_headers(app, company)

        r_mobile = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=mobile,
            headers=headers,
        )
        assert r_mobile.status_code == 201, (r_mobile.status_code, r_mobile.get_json())
        mobile_summary = r_mobile.get_json().get("summary")
        mobile_booking = db.session.get(Booking, mobile_summary.get("id"))
        assert mobile_booking.pickup_location == "Rue de la Gare 1, 1000 Lausanne"
        assert mobile_booking.dropoff_location == "Avenue de la Plage 10, 1000 Lausanne"

    def test_mobile_rejects_without_client_id(
        self, mock_route, mock_geocode, app, db, client
    ):
        """Mobile rejette si client_id absent (400)."""
        _requires_postgres(db)

        company = create_test_company(db)
        headers = _company_headers(app, company)

        payload = _base_mobile_payload(999, "2026-01-15T10:00:00")
        del payload["client_id"]

        r = client.post(
            "/api/v1/company_mobile/dispatch/v1/rides",
            json=payload,
            headers=headers,
        )
        assert r.status_code == 400, (r.status_code, r.get_json())
        data = r.get_json() or {}
        assert "client_id" in str(data).lower() or "requis" in str(data).lower()
