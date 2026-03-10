from datetime import UTC, datetime, timedelta

from models import (
    Booking,
    BookingStatus,
    Company,
    PricingModelType,
    PricingProfile,
    PricingProfileVersion,
    User,
    UserRole,
)


def _create_company_with_user(db, idx: int) -> Company:
    user = User(
        username=f"pricing_user_{idx}",
        email=f"pricing_user_{idx}@test.local",
        role=UserRole.COMPANY,
    )
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    company = Company(name=f"Pricing Company {idx}", user_id=user.id, dispatch_enabled=True)
    db.session.add(company)
    db.session.flush()
    return company


def test_pricing_simulate_returns_422_without_geo(client, auth_headers, db, sample_company):
    profile = PricingProfile(
        company_id=sample_company.id,
        name="Flat",
        model_type=PricingModelType.FLAT,
    )
    db.session.add(profile)
    db.session.flush()
    version = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=1,
        rules_json={"model": "flat", "base_fee": 45.0},
    )
    db.session.add(version)
    db.session.flush()

    response = client.post(
        "/api/v1/pricing/simulate",
        headers=auth_headers,
        json={
            "pricing_profile_version_id": version.id,
            "booking": {
                "pickup_at": "2026-02-24T19:30:00+01:00",
                "is_round_trip": False,
            },
        },
    )
    assert response.status_code in (422, 404)
    if response.status_code == 422:
        assert "incomplet" in str(response.get_json()).lower()


def test_pricing_simulate_forbidden_cross_company(client, auth_headers, db, sample_company):
    other_company = _create_company_with_user(db, 99)
    profile = PricingProfile(
        company_id=other_company.id,
        name="Flat Other",
        model_type=PricingModelType.FLAT,
    )
    db.session.add(profile)
    db.session.flush()
    version = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=1,
        rules_json={"model": "flat", "base_fee": 45.0},
    )
    db.session.add(version)
    db.session.flush()

    response = client.post(
        "/api/v1/pricing/simulate",
        headers=auth_headers,
        json={
            "pricing_profile_version_id": version.id,
            "booking": {
                "pickup_at": "2026-02-24T19:30:00+01:00",
                "pickup_zip": "1247",
                "is_round_trip": False,
                "distance_km": 8.4,
            },
        },
    )
    assert response.status_code in (403, 404)


def test_booking_pricing_version_freeze(db, sample_company, sample_client):
    profile = PricingProfile(
        company_id=sample_company.id,
        name="Freeze Profile",
        model_type=PricingModelType.FLAT,
    )
    db.session.add(profile)
    db.session.flush()
    version1 = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=1,
        rules_json={"model": "flat", "base_fee": 45.0},
    )
    db.session.add(version1)
    db.session.flush()
    profile.current_version_id = version1.id

    booking = Booking(
        customer_name="Freeze Test",
        pickup_location="Anières",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(hours=2),
        amount=45.0,
        status=BookingStatus.PENDING,
        user_id=sample_client.user_id,
        client_id=sample_client.id,
        company_id=sample_company.id,
        pricing_profile_id=profile.id,
        pricing_profile_version_id=version1.id,
        price_amount=45.0,
        price_breakdown_json={"total": "45.00"},
    )
    db.session.add(booking)
    db.session.flush()

    version2 = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=2,
        rules_json={"model": "flat", "base_fee": 60.0},
    )
    db.session.add(version2)
    db.session.flush()
    profile.current_version_id = version2.id
    db.session.flush()

    assert booking.pricing_profile_version_id == version1.id
    assert float(booking.price_amount) == 45.0


def test_pricing_simulate_zone_matrix_tokens(client, auth_headers, db, sample_company):
    profile = PricingProfile(
        company_id=sample_company.id,
        name="Zone Matrix",
        model_type=PricingModelType.ZONE,
    )
    db.session.add(profile)
    db.session.flush()
    version = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=1,
        rules_json={
            "model": "zone_matrix",
            "zones": [
                {"id": "z1", "code": "A", "label": "Centre", "tokens": ["commune:100"]},
                {"id": "z2", "code": "B", "label": "Rive", "tokens": ["commune:200"]},
            ],
            "matrix": {"z1": {"z2": 60.0}},
        },
    )
    db.session.add(version)
    db.session.flush()

    response = client.post(
        "/api/v1/pricing/simulate",
        headers=auth_headers,
        json={
            "pricing_profile_version_id": version.id,
            "booking": {
                "pickup_at": "2026-02-24T19:30:00+01:00",
                "pickup_admin_token": "commune:100",
                "dropoff_admin_token": "commune:200",
                "is_round_trip": False,
            },
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["amount"] == "60.00"
    assert payload["confidence"] == "exact"
    assert payload["breakdown"]["model"] == "zone_matrix"


def test_pricing_simulate_zone_count_blocks_when_exact_traversal_unavailable(
    client, auth_headers, db, sample_company
):
    profile = PricingProfile(
        company_id=sample_company.id,
        name="Zone Count",
        model_type=PricingModelType.ZONE,
    )
    db.session.add(profile)
    db.session.flush()
    version = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=1,
        rules_json={
            "v": 1,
            "model": "zone_count",
            "currency": "CHF",
            "zone_set_id": "zoneset_ge_v1",
            "components": {
                "base": {"enabled": True, "amount": 45},
                "zone_count": {
                    "enabled": True,
                    "unit_price": 5,
                    "strategy": "pickup_dropoff_diff_or_same",
                    "included_zones": 2,
                    "max_units": 10,
                },
                "distance": {"enabled": False, "per_km": 0, "included_km": 0, "rounding": "ceil_0_1"},
            },
            "extras": {},
            "caps": {"minimum": 0, "maximum": None},
        },
    )
    db.session.add(version)
    db.session.flush()

    response = client.post(
        "/api/v1/pricing/simulate",
        headers=auth_headers,
        json={
            "pricing_profile_version_id": version.id,
            "booking": {
                "pickup_at": "2026-02-24T19:30:00+01:00",
                "pickup_lat": 46.2044,
                "pickup_lng": 6.1432,
                "dropoff_lat": 46.2200,
                "dropoff_lng": 6.1200,
                "is_round_trip": False,
            },
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["amount"] is None
    assert payload["confidence"] == "blocked"
    assert isinstance(payload.get("blocking_reasons"), list)
    assert len(payload["blocking_reasons"]) > 0
