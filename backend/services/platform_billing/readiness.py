"""Readiness contrat / débiteur / créancier (PR1) — pas d'issuance_ready ici."""

from __future__ import annotations

from typing import Any

from models import Company
from models.billing_profile import CompanyBillingProfile
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformBillingCreditor,
)


def validate_platform_invoice_debtor(
    profile: CompanyBillingProfile | None,
    company: Company | None,
) -> tuple[bool, list[str]]:
    """Valide l'identité débiteur (transporteur facturé par LIRIE) — sans IBAN."""
    errors: list[str] = []
    if profile is None and company is None:
        return False, ["Profil et entreprise absents"]

    legal_name = (profile.legal_name if profile else None) or (
        company.name if company else None
    )
    if not legal_name:
        errors.append("Raison sociale manquante")

    street = profile.street_name if profile else None
    building = profile.building_number if profile else None
    postal = profile.postal_code if profile else None
    city = profile.city if profile else None
    country = (profile.country_code if profile else None) or "CH"

    if not profile:
        # Fallback domicile entreprise
        street = getattr(company, "domicile_address_line1", None) if company else None
        postal = getattr(company, "domicile_zip", None) if company else None
        city = getattr(company, "domicile_city", None) if company else None
        country = getattr(company, "domicile_country", None) or "CH"
        building = None

    if not street:
        errors.append("Rue manquante")
    if not building and profile is not None:
        errors.append("Numéro de rue manquant")
    if not postal:
        errors.append("NPA manquant")
    if not city:
        errors.append("Localité manquante")
    if not country:
        errors.append("Pays manquant")

    # Email recommandé (warning soft → pas bloquant identité)
    # IDE optionnel pour identité de base
    return len(errors) == 0, errors


def validate_platform_invoice_creditor(
    creditor: PlatformBillingCreditor | None,
) -> dict[str, Any]:
    """Valide le profil créancier LIRIE et la capacité QR."""
    profile_errors: list[str] = []
    qr_errors: list[str] = []
    if creditor is None or not creditor.is_active:
        return {
            "creditor_profile_ready": False,
            "creditor_qr_ready": False,
            "creditor_errors": ["Créancier LIRIE actif manquant"],
        }
    if not creditor.legal_name:
        profile_errors.append("Nom légal créancier manquant")
    if not creditor.street_name:
        profile_errors.append("Adresse créancier manquante")
    if not creditor.postal_code:
        profile_errors.append("NPA créancier manquant")
    if not creditor.city:
        profile_errors.append("Localité créancier manquante")
    if not creditor.country_code:
        profile_errors.append("Pays créancier manquant")

    iban = (creditor.qr_iban or creditor.iban or "").strip()
    if not iban:
        qr_errors.append("IBAN ou QR-IBAN créancier manquant")
    if not creditor.payment_reference_mode:
        qr_errors.append("Mode de référence paiement manquant")

    return {
        "creditor_profile_ready": len(profile_errors) == 0,
        "creditor_qr_ready": len(profile_errors) == 0 and len(qr_errors) == 0,
        "creditor_errors": profile_errors + qr_errors,
    }


def contract_calculation_ready(
    cfg: CompanyPlatformBillingConfig | None,
) -> tuple[bool, list[str]]:
    """Contrat suffisant pour calculer un relevé (QR non requis)."""
    errors: list[str] = []
    if cfg is None:
        return False, ["Aucun contrat applicable"]
    if not cfg.is_active:
        errors.append("Contrat inactif")
    if not cfg.is_billing_enabled:
        errors.append("Facturation désactivée")
    own = bool(getattr(cfg, "own_portfolio_billing_enabled", False))
    comm = bool(getattr(cfg, "lirie_commission_enabled", False))
    support = bool(getattr(cfg, "support_enabled", False))
    # Compat V1 : si flags encore false mais is_billing_enabled, calcul OK
    if not (own or comm or support or cfg.is_billing_enabled):
        errors.append("Aucun produit facturable activé")
    if (
        (
            getattr(cfg, "lirie_commission_enabled", False)
            or (cfg.is_billing_enabled and cfg.commission_rate is not None)
        )
        and cfg.commission_rate is not None
        and cfg.commission_rate < 0
    ):
        errors.append("Taux de commission invalide")
    return len(errors) == 0, errors


def build_company_readiness(
    *,
    company: Company,
    contract: CompanyPlatformBillingConfig | None,
    profile: CompanyBillingProfile | None,
    creditor: PlatformBillingCreditor | None,
) -> dict[str, Any]:
    calc_ok, calc_errors = contract_calculation_ready(contract)
    debtor_ok, debtor_errors = validate_platform_invoice_debtor(profile, company)
    cred = validate_platform_invoice_creditor(creditor)
    return {
        "contract_calculation_ready": calc_ok,
        "contract_calculation_errors": calc_errors,
        "debtor_identity_ready": debtor_ok,
        "debtor_identity_errors": debtor_errors,
        "creditor_profile_ready": cred["creditor_profile_ready"],
        "creditor_qr_ready": cred["creditor_qr_ready"],
        "creditor_errors": cred["creditor_errors"],
    }
