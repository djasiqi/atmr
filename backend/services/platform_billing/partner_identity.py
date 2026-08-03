"""Résolution déterministe de l'identité contractuelle partenaire."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.billing_profile import CompanyBillingProfile
from models.company import Company
from models.enums import LegalForm
from models.platform_billing import PlatformBillingCreditor

_LEGAL_FORM_LABELS_FR = {
    LegalForm.SOLE_PROPRIETORSHIP.value: "Indépendant",
    LegalForm.SARL.value: "Sàrl",
    LegalForm.SA.value: "SA",
    LegalForm.ASSOCIATION.value: "Association",
    LegalForm.FOUNDATION.value: "Fondation",
    LegalForm.OTHER.value: "Autre",
}

_LEGAL_FORMS = {m.value for m in LegalForm}

# Enseigne / marque — ne doit pas servir de raison sociale d'indépendant.
_OPERATOR_BRAND_ALIASES = frozenset({"lirie", "lirie.ch", "lirie sa", "lirie sàrl"})
_DEFAULT_OPERATOR_EMAIL = "info@lirie.ch"


@dataclass(frozen=True)
class ContractPartyIdentity:
    """Bloc identité d'une partie au contrat."""

    legal_name: str
    uid_ide: str | None
    street_name: str
    building_number: str | None
    postal_code: str
    city: str
    country_code: str
    legal_form: str | None
    signatory_name: str | None
    signatory_title: str | None
    identity_source: str
    company_id: int | None = None
    billing_profile_id: int | None = None
    creditor_id: int | None = None
    contractual_email: str | None = None

    def missing_fields(self, *, require_uid_ide: bool = True) -> list[str]:
        """Champs obligatoires pour générer un contrat.

        ``require_uid_ide`` : True pour le partenaire transporteur ; False pour
        l'Exploitant (indépendant sans IDE possible).
        """
        missing: list[str] = []
        if not (self.legal_name or "").strip():
            missing.append("raison_sociale")
        if require_uid_ide and not (self.uid_ide or "").strip():
            missing.append("uid_ide")
        if not (self.street_name or "").strip():
            missing.append("rue")
        if not (self.postal_code or "").strip():
            missing.append("npa")
        if not (self.city or "").strip():
            missing.append("localite")
        if not (self.legal_form or "").strip():
            missing.append("forme_juridique")
        if not (self.signatory_name or "").strip():
            missing.append("signataire")
        return missing

    def is_complete(self, *, require_uid_ide: bool = True) -> bool:
        return not self.missing_fields(require_uid_ide=require_uid_ide)

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "legal_name": self.legal_name,
            "uid_ide": self.uid_ide,
            "street_name": self.street_name,
            "building_number": self.building_number,
            "postal_code": self.postal_code,
            "city": self.city,
            "country_code": self.country_code,
            "legal_form": self.legal_form,
            "legal_form_label": legal_form_label_fr(self.legal_form),
            "signatory_name": self.signatory_name,
            "signatory_title": self.signatory_title,
            "contractual_email": self.contractual_email,
            "identity_source": self.identity_source,
            "company_id": self.company_id,
            "billing_profile_id": self.billing_profile_id,
            "creditor_id": self.creditor_id,
        }


def _partner_contractual_email(company: Company) -> str | None:
    for attr in ("billing_email", "contact_email"):
        value = (getattr(company, attr, None) or "").strip()
        if value:
            return value
    return None


def _resolve_sole_proprietor_legal_name(
    *, legal_name: str, signatory_name: str | None
) -> str:
    """Pour un indépendant, la dénomination doit être le nom de la personne physique."""
    name = (legal_name or "").strip()
    signatory = (signatory_name or "").strip()
    if signatory and (not name or name.lower() in _OPERATOR_BRAND_ALIASES):
        return signatory
    return name


def legal_form_label_fr(value: str | None) -> str | None:
    if not value:
        return None
    return _LEGAL_FORM_LABELS_FR.get(value, value)


def validate_legal_form(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    v = str(value).strip()
    if v not in _LEGAL_FORMS:
        raise ValueError(
            f"legal_form doit être parmi {sorted(_LEGAL_FORMS)}"
        )
    return v


def _profile_identity_complete(profile: CompanyBillingProfile) -> bool:
    return bool(
        (profile.legal_name or "").strip()
        and (profile.uid_ide or "").strip()
        and (profile.street_name or "").strip()
        and (profile.postal_code or "").strip()
        and (profile.city or "").strip()
    )


def _norm(s: str | None) -> str:
    return " ".join((s or "").strip().lower().split())


def detect_identity_divergence(
    company: Company, profile: CompanyBillingProfile | None
) -> list[str]:
    """Avertissements si profil et Company divergent (non bloquant)."""
    if profile is None or not _profile_identity_complete(profile):
        return []
    warnings: list[str] = []
    if _norm(profile.legal_name) != _norm(company.name):
        warnings.append("raison_sociale")
    if _norm(profile.uid_ide) != _norm(company.uid_ide):
        warnings.append("uid_ide")
    company_street = _norm(company.domicile_address_line1)
    if _norm(profile.street_name) != company_street:
        warnings.append("adresse")
    if _norm(profile.postal_code) != _norm(company.domicile_zip):
        warnings.append("npa")
    if _norm(profile.city) != _norm(company.domicile_city):
        warnings.append("localite")
    return warnings


def resolve_partner_contract_identity(
    company: Company, profile: CompanyBillingProfile | None = None
) -> ContractPartyIdentity:
    """Bloc profil complet OU bloc Company — jamais d'assemblage hybride."""
    partner_email = _partner_contractual_email(company)
    if profile is not None and _profile_identity_complete(profile):
        return ContractPartyIdentity(
            legal_name=(profile.legal_name or "").strip(),
            uid_ide=(profile.uid_ide or "").strip() or None,
            street_name=(profile.street_name or "").strip(),
            building_number=(profile.building_number or "").strip() or None,
            postal_code=(profile.postal_code or "").strip(),
            city=(profile.city or "").strip(),
            country_code=((profile.country_code or "CH").strip() or "CH").upper(),
            legal_form=company.legal_form,
            signatory_name=company.signatory_name,
            signatory_title=company.signatory_title,
            contractual_email=partner_email,
            identity_source="company_billing_profile",
            company_id=company.id,
            billing_profile_id=profile.id,
        )

    street = (company.domicile_address_line1 or "").strip()
    building = (company.domicile_address_line2 or "").strip() or None
    return ContractPartyIdentity(
        legal_name=(company.name or "").strip(),
        uid_ide=(company.uid_ide or "").strip() or None,
        street_name=street,
        building_number=building,
        postal_code=(company.domicile_zip or "").strip(),
        city=(company.domicile_city or "").strip(),
        country_code=((company.domicile_country or "CH").strip() or "CH").upper(),
        legal_form=company.legal_form,
        signatory_name=company.signatory_name,
        signatory_title=company.signatory_title,
        contractual_email=partner_email,
        identity_source="company",
        company_id=company.id,
        billing_profile_id=None,
    )


def resolve_operator_contract_identity(
    creditor: PlatformBillingCreditor | None,
) -> ContractPartyIdentity | None:
    if creditor is None:
        return None
    legal_name = (creditor.legal_name or "").strip()
    signatory_name = (creditor.signatory_name or "").strip() or None
    if creditor.legal_form == LegalForm.SOLE_PROPRIETORSHIP.value:
        legal_name = _resolve_sole_proprietor_legal_name(
            legal_name=legal_name, signatory_name=signatory_name
        )
    return ContractPartyIdentity(
        legal_name=legal_name,
        uid_ide=(creditor.uid_ide or "").strip() or None,
        street_name=(creditor.street_name or "").strip(),
        building_number=(creditor.building_number or "").strip() or None,
        postal_code=(creditor.postal_code or "").strip(),
        city=(creditor.city or "").strip(),
        country_code=((creditor.country_code or "CH").strip() or "CH").upper(),
        legal_form=creditor.legal_form,
        signatory_name=signatory_name,
        signatory_title=creditor.signatory_title,
        contractual_email=_DEFAULT_OPERATOR_EMAIL,
        identity_source="platform_billing_creditor",
        creditor_id=creditor.id,
    )


def serialize_partner_identity_payload(
    company: Company, profile: CompanyBillingProfile | None
) -> dict[str, Any]:
    identity = resolve_partner_contract_identity(company, profile)
    return {
        "identity": identity.to_snapshot_dict(),
        "is_complete": identity.is_complete(require_uid_ide=True),
        "missing_fields": identity.missing_fields(require_uid_ide=True),
        "divergence_warnings": detect_identity_divergence(company, profile),
        "company_fields": {
            "legal_form": company.legal_form,
            "signatory_name": company.signatory_name,
            "signatory_title": company.signatory_title,
            "uid_ide": company.uid_ide,
            "name": company.name,
            "domicile_address_line1": company.domicile_address_line1,
            "domicile_address_line2": company.domicile_address_line2,
            "domicile_zip": company.domicile_zip,
            "domicile_city": company.domicile_city,
            "domicile_country": company.domicile_country,
        },
    }
