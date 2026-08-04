"""Module de gestion de la facturation."""

from .banking_identifiers_sync import sync_banking_identifiers
from .billing_party_linker import (
    get_or_create_billing_party_for_direct_patient,
    get_or_create_billing_party_for_legacy_bill_to_client,
    resolve_billing_party_for_portfolio_patient,
)
from .billing_profile_service import BillingProfileService
from .payment_reference_generator import (
    PaymentReferenceGenerator,
    generate_scor_reference,
    validate_scor_reference,
)

__all__ = [
    "BillingProfileService",
    "PaymentReferenceGenerator",
    "generate_scor_reference",
    "get_or_create_billing_party_for_direct_patient",
    "get_or_create_billing_party_for_legacy_bill_to_client",
    "resolve_billing_party_for_portfolio_patient",
    "sync_banking_identifiers",
    "validate_scor_reference",
]
