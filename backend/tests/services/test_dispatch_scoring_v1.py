from datetime import UTC, datetime, timedelta

from models import (
    Booking,
    Company,
    DispatchOffer,
    DispatchOfferStatus,
    GeoUnit,
    GeoUnitType,
    ServiceArea,
    User,
    UserRole,
)
from models.enums import BookingStatus, ServiceCoverageMode
from services.dispatch.scoring_engine import (
    compute_candidates,
    persist_offers_for_threshold,
)


def _create_company(db, idx: int) -> Company:
    user = User(
        username=f"dispatch_user_{idx}",
        email=f"dispatch_user_{idx}@test.local",
        role=UserRole.COMPANY,
    )
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    company = Company(name=f"Company {idx}", user_id=user.id, dispatch_enabled=True)
    db.session.add(company)
    db.session.flush()
    return company


def test_commune_score_beats_canton(db):
    country = GeoUnit(type=GeoUnitType.COUNTRY, code="CH", name="Suisse")
    canton = GeoUnit(type=GeoUnitType.CANTON, code="GE", name="Genève", parent=country)
    commune_anieres = GeoUnit(
        type=GeoUnitType.COMMUNE, code="6630", name="Anières", parent=canton
    )
    db.session.add_all([country, canton, commune_anieres])
    db.session.flush()

    company_commune = _create_company(db, 1)
    company_canton = _create_company(db, 2)
    db.session.add_all(
        [
            ServiceArea(
                company_id=company_commune.id,
                geo_unit_id=commune_anieres.id,
                coverage_mode=ServiceCoverageMode.A_STRICT,
                weight=0,
                is_active=True,
            ),
            ServiceArea(
                company_id=company_canton.id,
                geo_unit_id=canton.id,
                coverage_mode=ServiceCoverageMode.A_STRICT,
                weight=0,
                is_active=True,
            ),
        ]
    )
    db.session.flush()

    candidates = compute_candidates(
        pickup_geo_unit=commune_anieres,
        drop_geo_unit=commune_anieres,
    )
    assert len(candidates) >= 2
    assert candidates[0].company_id == company_commune.id
    assert candidates[0].score > candidates[1].score


def test_declined_company_not_reproposed(db, sample_client):
    country = GeoUnit(type=GeoUnitType.COUNTRY, code="CH", name="Suisse")
    canton = GeoUnit(type=GeoUnitType.CANTON, code="GE", name="Genève", parent=country)
    commune = GeoUnit(
        type=GeoUnitType.COMMUNE, code="6630", name="Anières", parent=canton
    )
    db.session.add_all([country, canton, commune])
    db.session.flush()

    company = _create_company(db, 3)
    service_area = ServiceArea(
        company_id=company.id,
        geo_unit_id=commune.id,
        coverage_mode=ServiceCoverageMode.A_STRICT,
        weight=0,
        is_active=True,
    )
    db.session.add(service_area)
    db.session.flush()

    booking = Booking(
        customer_name="Test",
        pickup_location="Anières",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(hours=2),
        amount=50.0,
        status=BookingStatus.PENDING,
        user_id=sample_client.user_id,
        client_id=sample_client.id,
        company_id=company.id,
        pickup_geo_unit_id=commune.id,
        dropoff_geo_unit_id=commune.id,
    )
    db.session.add(booking)
    db.session.flush()

    declined = DispatchOffer(
        booking_id=booking.id,
        company_id=company.id,
        status=DispatchOfferStatus.DECLINED,
        score=100,
        reason_json={"engine": "test"},
    )
    db.session.add(declined)
    db.session.flush()

    candidates = compute_candidates(pickup_geo_unit=commune, drop_geo_unit=commune)
    created = persist_offers_for_threshold(
        booking_id=booking.id, candidates=candidates, threshold=100
    )
    assert created == []


def test_c_intra_inter_canton_matches_pickup_canton_carrier(db):
    """C_INTRA_ONLY : trajet GE → VD reste proposé aux entreprises canton GE (prise en charge)."""
    country = GeoUnit(type=GeoUnitType.COUNTRY, code="CH", name="Suisse")
    canton_ge = GeoUnit(
        type=GeoUnitType.CANTON, code="GE", name="Genève", parent=country
    )
    canton_vd = GeoUnit(type=GeoUnitType.CANTON, code="VD", name="Vaud", parent=country)
    commune_lancy = GeoUnit(
        type=GeoUnitType.COMMUNE, code="1213", name="Petit-Lancy", parent=canton_ge
    )
    commune_nyon = GeoUnit(
        type=GeoUnitType.COMMUNE, code="1260", name="Nyon", parent=canton_vd
    )
    db.session.add_all([country, canton_ge, canton_vd, commune_lancy, commune_nyon])
    db.session.flush()

    company_ge = _create_company(db, 40)
    db.session.add(
        ServiceArea(
            company_id=company_ge.id,
            geo_unit_id=canton_ge.id,
            coverage_mode=ServiceCoverageMode.C_INTRA_ONLY,
            weight=0,
            is_active=True,
        )
    )
    db.session.flush()

    candidates = compute_candidates(
        pickup_geo_unit=commune_lancy,
        drop_geo_unit=commune_nyon,
    )
    ids = {c.company_id for c in candidates}
    assert company_ge.id in ids
