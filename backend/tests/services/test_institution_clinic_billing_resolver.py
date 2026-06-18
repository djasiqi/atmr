"""Résolution clinique payeuse et éligibilité S2 portail institution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.billing.institution_billing_resolver import (
    _resolve_clinic_company_id_for_institution,
)


def test_resolve_clinic_ignores_transport_company_as_clinic():
    booking = SimpleNamespace(
        client_id=10,
        billed_to_company_id=5,
        client=None,
    )
    with patch(
        "services.billing.institution_billing_resolver.Client"
    ) as mock_client_cls:
        mock_client_cls.query.get.return_value = SimpleNamespace(
            is_institution=True,
            default_billed_to_company_id=99,
            institution_name="Clinique Test",
        )
        cid = _resolve_clinic_company_id_for_institution(
            booking=booking,
            company_id=5,
            billing_party_id=1,
        )
    assert cid == 99


def test_resolve_clinic_from_institution_name_when_no_default():
    booking = SimpleNamespace(
        client_id=10,
        billed_to_company_id=5,
        client=None,
    )
    clinic_co = SimpleNamespace(id=42)
    with patch(
        "services.billing.institution_billing_resolver.Client"
    ) as mock_client_cls, patch(
        "services.billing.institution_billing_resolver.Company"
    ) as mock_co_cls:
        mock_client_cls.query.get.return_value = SimpleNamespace(
            is_institution=True,
            default_billed_to_company_id=None,
            institution_name="Clinique Les Hauts d'Anières",
        )
        mock_co_cls.query.filter.return_value.order_by.return_value.first.return_value = (
            clinic_co
        )
        cid = _resolve_clinic_company_id_for_institution(
            booking=booking,
            company_id=5,
            billing_party_id=1,
        )
    assert cid == 42
