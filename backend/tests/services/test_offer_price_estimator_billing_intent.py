# ruff: noqa: I001
"""Tests tarification offres institution selon billing_intent effectif."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from application.institutions.accept_offer import AcceptOfferUseCase
from services.pricing.offer_price_estimator import (
    SOURCE_COMPANY_PROFILE,
    SOURCE_PREFERENTIAL,
    effective_preferential_rate,
    estimate_offer_price,
    institution_preferential_applies,
    resolve_institution_price,
)


class _DummyVersion:
    def __init__(self, version_id: int = 1):
        self.id = version_id
        self.rules_json = {"model": "flat", "base_fee": 45.0}


class _DummyProfile:
    def __init__(self, profile_id: int = 1):
        self.id = profile_id
        self.currency = "CHF"
        self.current_version = _DummyVersion()


def _patch_company_profile(amount: float = 45.0):
    """Contexte Flask + profil tarifaire actif renvoyant ``amount`` CHF."""
    return (
        patch(
            "services.pricing.offer_price_estimator.has_app_context",
            return_value=True,
        ),
        patch(
            "services.pricing.offer_price_estimator._active_profile_version",
            return_value=(_DummyProfile(), _DummyVersion()),
        ),
        patch(
            "services.pricing.offer_price_estimator._compute_distance_meters",
            return_value=5000,
        ),
        patch(
            "services.pricing.offer_price_estimator.compute_price",
            return_value=(amount, {"total": str(amount)}),
        ),
    )


class TestInstitutionPreferentialHelpers:
    def test_institution_preferential_applies_only_for_institution(self):
        assert institution_preferential_applies("institution") is True
        assert institution_preferential_applies("patient") is False
        assert institution_preferential_applies("curator") is False
        assert institution_preferential_applies("insurance") is False
        assert institution_preferential_applies(None) is False

    def test_effective_preferential_rate_neutralized_for_non_institution(self):
        assert effective_preferential_rate(35.0, "institution") == 35.0
        assert effective_preferential_rate(35.0, "patient") is None
        assert effective_preferential_rate(35.0, "curator") is None


class TestResolveInstitutionPriceBillingIntent:
    def test_institution_with_preferential_uses_preferential(self):
        result = resolve_institution_price(
            company_id=1,
            effective_billing_intent="institution",
            preferential_rate=35.0,
        )
        assert result["amount"] == 35.0
        assert result["source"] == SOURCE_PREFERENTIAL

    def test_institution_without_preferential_uses_company_profile(self):
        patches = _patch_company_profile(45.0)
        with patches[0], patches[1], patches[2], patches[3]:
            result = resolve_institution_price(
                company_id=1,
                effective_billing_intent="institution",
                preferential_rate=None,
            )
        assert result["amount"] == 45.0
        assert result["source"] == SOURCE_COMPANY_PROFILE

    def test_patient_ignores_preferential_uses_company_profile(self):
        patches = _patch_company_profile(45.0)
        with patches[0], patches[1], patches[2], patches[3]:
            result = resolve_institution_price(
                company_id=1,
                effective_billing_intent="patient",
                preferential_rate=35.0,
            )
        assert result["amount"] == 45.0
        assert result["source"] == SOURCE_COMPANY_PROFILE

    @pytest.mark.parametrize("intent", ["curator", "insurance", "other", "spc"])
    def test_third_party_payers_use_company_profile_not_preferential(self, intent: str):
        patches = _patch_company_profile(45.0)
        with patches[0], patches[1], patches[2], patches[3]:
            result = resolve_institution_price(
                company_id=1,
                effective_billing_intent=intent,
                preferential_rate=35.0,
            )
        assert result["amount"] == 45.0
        assert result["source"] == SOURCE_COMPANY_PROFILE


@dataclass
class _Leg:
    sequence_index: int
    pickup_location: str = "EMS"
    dropoff_location: str = "HUG"
    destination_billing_override: str | None = None
    pickup_lat: float | None = None
    pickup_lng: float | None = None
    dropoff_lat: float | None = None
    dropoff_lng: float | None = None


@dataclass
class _TransportRequest:
    billing_intent: str = "institution"
    scheduled_time: Any = None
    is_round_trip: bool = False
    pickup_location: str = "EMS"
    dropoff_location: str = "HUG"
    legs: list[_Leg] = field(default_factory=list)


@dataclass
class _Offer:
    company_id: int = 5
    transport_request: _TransportRequest = field(default_factory=_TransportRequest)


class TestEstimateOfferPriceMultiLeg:
    def test_multi_leg_institution_and_patient_legs(self):
        request = _TransportRequest(
            billing_intent="institution",
            legs=[
                _Leg(sequence_index=0),
                _Leg(sequence_index=1, destination_billing_override="patient"),
            ],
        )
        offer = _Offer(transport_request=request)

        with patch(
            "services.pricing.offer_price_estimator._resolve_institution_client_readonly",
            return_value=MagicMock(preferential_rate=Decimal("35.00")),
        ):
            patches = _patch_company_profile(45.0)
            with patches[0], patches[1], patches[2], patches[3]:
                with patch(
                    "services.pricing.offer_price_estimator._build_pricing_context",
                    return_value={},
                ):
                    result = estimate_offer_price(offer)

        assert result is not None
        assert result["amount"] == 80.0
        assert result["source"] == "mixed"


@dataclass
class _AcceptClient:
    id: int = 10
    preferential_rate: Decimal = Decimal("35.00")


@dataclass
class _AcceptPatient:
    first_name: str = "Jean"
    last_name: str = "Dupont"
    dob: Any = None


@dataclass
class _AcceptInstitution:
    id: int = 1
    name: str = "Clinique Test"


@dataclass
class _AcceptTransportRequest:
    id: int = 100
    billing_intent: str = "patient"
    institution: _AcceptInstitution = field(default_factory=_AcceptInstitution)
    patient: _AcceptPatient = field(default_factory=_AcceptPatient)
    pickup_location: str = "A"
    pickup_lat: float | None = 46.2
    pickup_lng: float | None = 6.1
    dropoff_location: str = "B"
    dropoff_lat: float | None = 46.3
    dropoff_lng: float | None = 6.2
    scheduled_time: Any = None
    is_round_trip: bool = False
    return_time: Any = None
    mission_type: str = "patient_transport"
    delivery_description: str | None = None
    mobility: dict | None = None
    notes: str | None = None
    billing_details: dict | None = None
    legs: list | None = None
    pickup_floor: str | None = None
    pickup_door_code: str | None = None
    dropoff_floor: str | None = None
    dropoff_door_code: str | None = None
    pickup_type: str | None = None
    dropoff_type: str | None = None
    public_id: str = "PUB-1"
    external_reference: str = "EXT-1"


class TestEstimateMatchesAcceptOfferAmount:
    @patch("application.institutions.accept_offer.resolve_institution_price")
    @patch("application.institutions.accept_offer.db")
    def test_accept_offer_passes_patient_billing_intent_to_resolve(
        self, mock_db: MagicMock, mock_resolve: MagicMock
    ):
        """L'acceptation transmet le billing_intent effectif au résolveur tarifaire."""
        mock_resolve.return_value = {
            "amount": 45.0,
            "source": SOURCE_COMPANY_PROFILE,
            "pricing_profile_id": 1,
            "pricing_profile_version_id": 1,
            "breakdown": None,
            "currency": "CHF",
        }
        tr = _AcceptTransportRequest(billing_intent="patient")
        uc = AcceptOfferUseCase()
        uc._get_or_create_institution_client = MagicMock(  # type: ignore[method-assign]
            return_value=_AcceptClient()
        )
        uc._resolve_billing_party = MagicMock()  # type: ignore[method-assign]
        uc._format_pickup_notes = MagicMock(return_value=None)  # type: ignore[method-assign]
        uc._format_dropoff_notes = MagicMock(return_value=None)  # type: ignore[method-assign]
        uc._get_mobility_flag = MagicMock(return_value=False)  # type: ignore[method-assign]
        uc._apply_clinical_dropoff_from_request = MagicMock()  # type: ignore[method-assign]
        uc._build_metadata = MagicMock(return_value={})  # type: ignore[method-assign]

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if getattr(b, "id", 0) == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, _ret = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs["effective_billing_intent"] == "patient"
        assert float(outbound.amount) == 45.0

    def test_estimate_and_resolve_same_amount_for_patient_billing(self):
        """Estimation offre et résolveur unitaire produisent le même montant (patient)."""
        patches = _patch_company_profile(45.0)
        with patches[0], patches[1], patches[2], patches[3]:
            with patch(
                "services.pricing.offer_price_estimator._build_pricing_context",
                return_value={},
            ):
                with patch(
                    "services.pricing.offer_price_estimator._resolve_institution_client_readonly",
                    return_value=MagicMock(preferential_rate=Decimal("35.00")),
                ):
                    tr = _AcceptTransportRequest(billing_intent="patient")
                    offer = _Offer(company_id=5, transport_request=tr)  # type: ignore[arg-type]
                    estimate = estimate_offer_price(offer)
                    resolved = resolve_institution_price(
                        company_id=5,
                        effective_billing_intent="patient",
                        preferential_rate=Decimal("35.00"),
                        pickup_location=tr.pickup_location,
                        dropoff_location=tr.dropoff_location,
                    )

        assert estimate is not None
        assert estimate["amount"] == 45.0
        assert estimate["source"] == SOURCE_COMPANY_PROFILE
        assert resolved["amount"] == estimate["amount"]
        assert resolved["source"] == estimate["source"]

    def test_estimate_and_resolve_same_amount_for_institution_billing(self):
        """Estimation offre et résolveur unitaire produisent le même montant (institution)."""
        tr = _AcceptTransportRequest(billing_intent="institution")
        offer = _Offer(company_id=5, transport_request=tr)  # type: ignore[arg-type]

        with patch(
            "services.pricing.offer_price_estimator._resolve_institution_client_readonly",
            return_value=MagicMock(preferential_rate=Decimal("35.00")),
        ):
            estimate = estimate_offer_price(offer)
            resolved = resolve_institution_price(
                company_id=5,
                effective_billing_intent="institution",
                preferential_rate=Decimal("35.00"),
                pickup_location=tr.pickup_location,
                dropoff_location=tr.dropoff_location,
            )

        assert estimate is not None
        assert estimate["amount"] == 35.0
        assert estimate["source"] == SOURCE_PREFERENTIAL
        assert resolved["amount"] == estimate["amount"]
        assert resolved["source"] == estimate["source"]
