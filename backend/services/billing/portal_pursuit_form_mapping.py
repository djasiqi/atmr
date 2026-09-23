"""Mapping versionné des champs requis pour une réquisition de poursuite (CH).

Source documentaire (information publique, canton de Genève) :
https://www.ge.ch/poursuites/deposer-requisition-poursuite
Dernière consultation produit : 2026-09-24 (page mise à jour 2026-07-07).

Ce mapping n'est PAS le formulaire fédéral officiel ni une intégration EasyGov.
Il verrouille les champs minimaux identifiés avant toute transmission humaine.
"""

from __future__ import annotations

from typing import Any

# Identifiant stable de la version documentée (pas un softcode « Geneva default »).
PURSUIT_FORM_MAPPING_VERSION = "ch-ge-requisition-info-2026-07"
PURSUIT_FORM_MAPPING_SOURCE = (
    "https://www.ge.ch/poursuites/deposer-requisition-poursuite"
)

# Champs minimaux cités par l'office cantonal (débiteur domicilié à Genève) :
# - nom et adresse complète du débiteur
# - montant en CHF
# - titre ou motif de la créance
# + identité créancier nécessaire côté pratique (émetteur de la réquisition)
REQUIRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "form_field": "creditor_legal_name",
        "lirie_source": "Company.legal_name",
        "status": "mapped",
    },
    {
        "form_field": "creditor_postal_address",
        "lirie_source": "Company.domicile_* / address",
        "status": "mapped",
    },
    {
        "form_field": "debtor_full_name",
        "lirie_source": "PortalReceivable.debtor_name_snapshot / User first+last",
        "status": "mapped",
    },
    {
        "form_field": "debtor_complete_postal_address",
        "lirie_source": "PortalReceivable.debtor_domicile_address_snapshot",
        "status": "mapped",
        "note": "Domicile explicite — billing_address insuffisant",
    },
    {
        "form_field": "claim_amount_chf",
        "lirie_source": "PortalReceivable.balance_due",
        "status": "mapped",
    },
    {
        "form_field": "claim_title_or_reason",
        "lirie_source": "build_claim_reason(receivable)",
        "status": "mapped",
    },
)

# Champs souvent présents sur formulaires fédéraux / EasyGov mais non verrouillés ici.
PARTIAL_FIELDS: tuple[dict[str, str], ...] = (
    {
        "form_field": "creditor_uid_ide",
        "lirie_source": "Company.uid_ide",
        "status": "optional_partial",
    },
    {
        "form_field": "pursuit_office_jurisdiction",
        "lirie_source": "pursuit_jurisdiction (manual)",
        "status": "not_auto_resolved",
        "note": "Office compétent = domicile/siège débiteur — pas de défaut Genève",
    },
    {
        "form_field": "easygov_electronic_submission",
        "lirie_source": None,
        "status": "not_implemented",
    },
)


def pursuit_form_mapping_report() -> dict[str, Any]:
    mapped = [f for f in REQUIRED_FIELDS if f["status"] == "mapped"]
    incomplete = list(PARTIAL_FIELDS)
    overall = "PASS" if len(mapped) == len(REQUIRED_FIELDS) else "PARTIAL"
    # Les champs partiels (juridiction auto, EasyGov) empêchent un PASS « formulaire complet ».
    if incomplete:
        overall = "PARTIAL"
    return {
        "version": PURSUIT_FORM_MAPPING_VERSION,
        "source_reference": PURSUIT_FORM_MAPPING_SOURCE,
        "overall": overall,
        "required_fields": list(REQUIRED_FIELDS),
        "partial_or_future_fields": incomplete,
        "easygov_capability": "AVAILABLE",
        "easygov_integration": "NOT_IMPLEMENTED",
        "disclaimer": (
            "Mapping informatif versionné. Ce n'est pas le schéma binaire "
            "du formulaire fédéral ni une preuve de conformité EasyGov."
        ),
    }
