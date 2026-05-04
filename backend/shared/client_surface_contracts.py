"""Contrats versionnés partagés pour l'expérience client multi-surface."""

from __future__ import annotations

STATUS_DICTIONARY_VERSION = "1.0.0"
PRICING_CONTRACT_VERSION = "1.0.0"
CANONICAL_ADDRESS_CONTRACT_VERSION = "1.0.0"
PREVIEW_CONTRACT_VERSION = "1.0.0"
MEDICAL_FIELDS_CONTRACT_VERSION = "1.0.0"

PRICING_STATUS_VALUES = {
    "unavailable",
    "indicative",
    "estimated",
    "confirmed",
    "adjusted",
}

CANONICAL_ADDRESS_PRECISION_LEVELS = {
    "rooftop",
    "entrance",
    "street",
    "locality",
    "approximate",
}

CANONICAL_PRECISION_ACCEPTANCE_MATRIX = {
    "rooftop": "allow",
    "entrance": "allow",
    "street": "warn",
    "locality": "block",
    "approximate": "block",
}
