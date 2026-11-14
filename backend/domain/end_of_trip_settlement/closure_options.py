"""Options de clôture chauffeur — convention, règlement ou signalement.

Règles métier (client particulier personnellement débiteur) :

- Convention applicable → option facturation différée proposée.
- Pas de convention → espèces / carte (carte non opérationnelle = non payé).
- Aucun règlement → signalement motivé ; mission clôturable ; pas un paiement.

Ne confond pas PlatformPartnerAgreement ni les CGV avec la convention transport.
Ne confond pas destinataire de facture et débiteur.
Les transports institutionnels restent hors de cette matrice client.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

OPTION_DEFERRED = "deferred_convention"
OPTION_CASH = "cash"
OPTION_CARD = "card"
OPTION_INCIDENT = "no_payment_incident"

_CARD_OPERATIONAL = False  # Tap to Pay simulé — non opérationnel

_PRIVATE_DEBTORS = frozenset({"patient", "client", "account_holder"})
_INSTITUTION_DEBTORS = frozenset({"clinic", "institution"})
_APPLICABLE_KINDS = frozenset({"applicable_demonstration", "applicable"})


@dataclass(frozen=True, slots=True)
class ClosureContext:
    """Faits nécessaires à la présentation des options. Aucune écriture."""

    booking_id: int
    company_id: int
    driver_company_id: int
    billed_to_type: str
    mandate_kind: str
    amount_due_chf: Decimal
    payment_received: bool
    card_provider_operational: bool = _CARD_OPERATIONAL
    invoice_recipient_changed: bool = False
    debtor_id: int | None = None
    debtor_label: str | None = None


@dataclass(frozen=True, slots=True)
class ClosureOption:
    code: str
    label: str
    records_payment: bool
    authorizes_deferred_billing: bool
    opens_incident: bool
    available: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ClosureMenu:
    """Menu chauffeur. Le téléphone ne reconstruit pas ces règles."""

    applicable: bool
    options: tuple[ClosureOption, ...]
    mission_close_without_payment_allowed_via_incident: bool
    debtor_preserved_on_recipient_change: bool
    institution_uses_own_rules: bool
    reasons: tuple[str, ...]


def build_closure_menu(ctx: ClosureContext) -> ClosureMenu:
    """Construit les options pertinentes uniquement."""
    payer = (ctx.billed_to_type or "").strip().lower()
    if int(ctx.driver_company_id) != int(ctx.company_id):
        return ClosureMenu(
            applicable=False,
            options=(),
            mission_close_without_payment_allowed_via_incident=False,
            debtor_preserved_on_recipient_change=True,
            institution_uses_own_rules=False,
            reasons=("company_isolation_refused",),
        )
    if payer in _INSTITUTION_DEBTORS:
        return ClosureMenu(
            applicable=False,
            options=(),
            mission_close_without_payment_allowed_via_incident=False,
            debtor_preserved_on_recipient_change=True,
            institution_uses_own_rules=True,
            reasons=("institution_own_rules",),
        )
    if payer not in _PRIVATE_DEBTORS:
        return ClosureMenu(
            applicable=False,
            options=(),
            mission_close_without_payment_allowed_via_incident=False,
            debtor_preserved_on_recipient_change=True,
            institution_uses_own_rules=False,
            reasons=("payer_not_private_client",),
        )
    if ctx.payment_received:
        return ClosureMenu(
            applicable=True,
            options=(),
            mission_close_without_payment_allowed_via_incident=False,
            debtor_preserved_on_recipient_change=True,
            institution_uses_own_rules=False,
            reasons=("already_paid",),
        )

    convention_ok = ctx.mandate_kind in _APPLICABLE_KINDS
    options: list[ClosureOption] = []
    if convention_ok:
        options.append(
            ClosureOption(
                code=OPTION_DEFERRED,
                label="Facturation différée autorisée par convention",
                records_payment=False,
                authorizes_deferred_billing=True,
                opens_incident=False,
                available=True,
            )
        )
    else:
        options.append(
            ClosureOption(
                code=OPTION_DEFERRED,
                label="Facturation différée",
                records_payment=False,
                authorizes_deferred_billing=False,
                opens_incident=False,
                available=False,
                reason="convention_not_applicable",
            )
        )

    options.append(
        ClosureOption(
            code=OPTION_CASH,
            label="Paiement espèces",
            records_payment=True,
            authorizes_deferred_billing=False,
            opens_incident=False,
            available=True,
        )
    )

    if ctx.card_provider_operational:
        options.append(
            ClosureOption(
                code=OPTION_CARD,
                label="Paiement carte",
                records_payment=True,
                authorizes_deferred_billing=False,
                opens_incident=False,
                available=True,
            )
        )
    else:
        options.append(
            ClosureOption(
                code=OPTION_CARD,
                label="Paiement carte",
                records_payment=False,
                authorizes_deferred_billing=False,
                opens_incident=False,
                available=False,
                reason="card_provider_not_operational",
            )
        )

    options.append(
        ClosureOption(
            code=OPTION_INCIDENT,
            label="Aucun règlement — signalement",
            records_payment=False,
            authorizes_deferred_billing=False,
            opens_incident=True,
            available=True,
        )
    )

    return ClosureMenu(
        applicable=True,
        options=tuple(options),
        mission_close_without_payment_allowed_via_incident=True,
        debtor_preserved_on_recipient_change=True,
        institution_uses_own_rules=False,
        reasons=(),
    )


def preserve_debtor_on_recipient_change(
    *,
    debtor_id: int,
    debtor_type: str,
    new_recipient_id: int | None,
    new_recipient_label: str | None,
) -> dict[str, object]:
    """Changer le destinataire ne change pas le débiteur."""
    return {
        "debtor_id": debtor_id,
        "debtor_type": debtor_type,
        "invoice_recipient_id": new_recipient_id,
        "invoice_recipient_label": new_recipient_label,
        "debtor_changed": False,
        "recipient_changed": True,
    }
