# application/institutions/eligible_carriers.py
"""Contrat d'éligibilité des transporteurs pour le sélecteur institution.

`dispatch_enabled` n'entre PAS dans ce contrat.

Il contrôle le dispatch interne de flotte de l'entreprise
(`dispatch_mode == MANUAL` ⇒ `dispatch_enabled == False`).
Une entreprise manuelle approuvée (ex. Emmenez-moi) peut quand même
recevoir des missions LIRIE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import joinedload

from models import Company, Institution
from models.enums import PlatformBillingAccessState
from services.demo.soft_delete_guard import (
    company_is_demo,
    filter_companies_for_institution,
    institution_is_demo,
)

PredicateResult = Literal["PASS", "FAIL"]

# Comptes créés par pytest / load-test / factories (pas des partenaires).
_SYNTHETIC_EMAIL_SUFFIXES = (
    "@test.com",
    "@test.ch",
    "@atmr-test.ch",
    "@x.ch",
    "@example.com",
    "@localhost",
)


@dataclass(frozen=True, slots=True)
class CarrierEligibility:
    """Résultat du prédicat d'éligibilité pour une entreprise."""

    company_id: int
    approved: PredicateResult
    officially_accepted: PredicateResult
    not_suspended: PredicateResult
    owner_active: PredicateResult
    not_synthetic: PredicateResult
    institution_compatible: PredicateResult
    marketplace_billing: PredicateResult

    @property
    def eligible(self) -> bool:
        return all(
            value == "PASS"
            for value in (
                self.approved,
                self.officially_accepted,
                self.not_suspended,
                self.owner_active,
                self.not_synthetic,
                self.institution_compatible,
                self.marketplace_billing,
            )
        )


def _pass(ok: bool) -> PredicateResult:
    return "PASS" if ok else "FAIL"


def owner_email(company: Company) -> str:
    owner = getattr(company, "user", None)
    return str(getattr(owner, "email", None) or "").strip().lower()


def is_synthetic_test_carrier(company: Company) -> bool:
    """True si le compte propriétaire est un tenant de test/fixture."""
    email = owner_email(company)
    contact = str(getattr(company, "contact_email", None) or "").strip().lower()
    return any(
        email.endswith(suffix) or contact.endswith(suffix)
        for suffix in _SYNTHETIC_EMAIL_SUFFIXES
    )


def is_owner_operational(company: Company) -> bool:
    owner = getattr(company, "user", None)
    if owner is None:
        return False
    if getattr(owner, "disabled_at", None) is not None:
        return False
    if getattr(owner, "archived_at", None) is not None:
        return False
    status = str(getattr(owner, "account_status", None) or "").strip().lower()
    return status in ("", "active")


def can_receive_marketplace(company: Company) -> bool:
    state = (
        getattr(company, "platform_billing_access_state", None)
        or PlatformBillingAccessState.ACTIVE.value
    )
    if state == PlatformBillingAccessState.ACTIVE.value:
        return True
    from services.platform_billing.capabilities import (
        BillingCapability,
        is_billing_capability_allowed,
    )

    return is_billing_capability_allowed(
        company.id,
        BillingCapability.RECEIVE_MARKETPLACE_OFFERS,
    )


def explain_carrier_eligibility(
    company: Company,
    institution: Institution | None,
) -> CarrierEligibility:
    """Évalue chaque condition du contrat (diagnostic, pas d'inférence par nom)."""
    compatible = True
    if institution is not None:
        inst_demo = institution_is_demo(institution)
        co_demo = company_is_demo(company)
        compatible = co_demo if inst_demo else not co_demo

    return CarrierEligibility(
        company_id=company.id,
        approved=_pass(bool(company.is_approved)),
        officially_accepted=_pass(getattr(company, "accepted_at", None) is not None),
        not_suspended=_pass(not bool(getattr(company, "platform_suspended", False))),
        owner_active=_pass(is_owner_operational(company)),
        not_synthetic=_pass(not is_synthetic_test_carrier(company)),
        institution_compatible=_pass(compatible),
        marketplace_billing=_pass(can_receive_marketplace(company)),
    )


def companies_for_marketplace_dispatch(
    *,
    institution: Institution | None,
    excluded_ids: list[int] | None = None,
    only_company_ids: list[int] | None = None,
    require_official_acceptance: bool | None = None,
) -> list[Company]:
    """Entreprises destinataires d'offres LIRIE.

    - Liste bornée (`only_company_ids`) : transporteurs déjà choisis
      par l'institution. Pas de `dispatch_enabled`, pas d'`accepted_at`
      obligatoire (les fixtures de test restent utilisables).
    - Catalogue non borné (sélecteur, relance hors prefs, fallback) :
      approbation officielle + compte opérationnel + hors données test.
    """
    bounded = only_company_ids is not None
    if require_official_acceptance is None:
        require_official_acceptance = not bounded

    query = Company.query.options(joinedload(Company.user)).filter(
        Company.is_approved.is_(True),
        Company.platform_suspended.is_(False),
    )
    if require_official_acceptance:
        query = query.filter(Company.accepted_at.isnot(None))
    if excluded_ids:
        query = query.filter(Company.id.notin_(excluded_ids))
    if only_company_ids is not None:
        query = query.filter(Company.id.in_(only_company_ids))

    companies = query.order_by(Company.name).all()
    companies = filter_companies_for_institution(companies, institution)
    result: list[Company] = []
    for company in companies:
        if not can_receive_marketplace(company):
            continue
        if require_official_acceptance:
            if not is_owner_operational(company):
                continue
            if is_synthetic_test_carrier(company):
                continue
        result.append(company)
    return result


def list_eligible_transport_companies(
    institution: Institution | None,
) -> list[Company]:
    """Catalogue partenaire pour le sélecteur institution."""
    return companies_for_marketplace_dispatch(
        institution=institution,
        require_official_acceptance=True,
    )
