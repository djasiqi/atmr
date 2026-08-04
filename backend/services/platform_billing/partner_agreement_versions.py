"""Versions centrales du pack contrat partenaire LIRIE (source unique)."""

from __future__ import annotations

# Schéma technique du pack (gate d'intégrité / migration).
PACK_SCHEMA_VERSION = "lirie-partner-pack-v1"

# Document particular (3 pages) — nouveau modèle compact.
PARTICULAR_VERSION = "lirie-partner-particular-v1.32.1"

# Annexes canoniques (identiques pour tous les partenaires).
GENERAL_TERMS_VERSION = "lirie-partner-terms-v1.20"
DPA_VERSION = "lirie-dpa-v1.20"

RETENTION_POLICY_VERSION = "lirie-retention-v1"
SUBPROCESSORS_VERSION = "lirie-subprocessors-v2"
PENALTY_CALCULATION_VERSION = "lirie-penalty-v1"
COMMERCIAL_SNAPSHOT_SCHEMA_VERSION = "lirie-commercial-snapshot-v1"

GENERATOR_VERSION = "1.30"

# Compatibilité / migration : l'ancien TEMPLATE_VERSION pointe vers le pack.
TEMPLATE_VERSION = PACK_SCHEMA_VERSION
LEGAL_TEXT_VERSION = PARTICULAR_VERSION

# Pénalité : commissions éludées + max(2 × commissions, CHF 1'000)
PENALTY_MULTIPLIER = 2
PENALTY_MINIMUM_CHF = 1000
PENALTY_CURRENCY = "CHF"

SPECIAL_CONDITIONS_MAX_LENGTH = 4000
MANIFEST_DOCUMENT_VERSION = "lirie-delivery-manifest-v1"
