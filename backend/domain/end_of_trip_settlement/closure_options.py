"""Options de clôture chauffeur — convention, règlement ou signalement.

Règles métier (client particulier personnellement débiteur) :

- Convention juridiquement applicable → facturation différée réelle autorisée.
- Convention de démonstration → option visible de préparation seulement ;
  jamais d'autorisation de facturation différée réelle.
- Pas de convention → espèces / carte terminal externe (si confirmée) /
  Tap to Pay simulé (non opérationnel).
- Aucun règlement → signalement motivé ; mission clôturable ; pas un paiement.

Ne confond pas terminal bancaire externe et Tap to Pay simulé.
Ne confond pas PlatformPartnerAgreement ni les CGV avec la convention transport.
Ne confond pas destinataire de facture et débiteur.
Les transports institutionnels restent hors de cette matrice client.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

OPTION_DEFERRED = "deferred_convention"
OPTION_CASH = "cash"
OPTION_CARD_EXTERNAL = "card_external_terminal"
OPTION_CARD_TAP = "card_tap_to_pay"
OPTION_INCIDENT = "no_payment_incident"

# Alias rétrocompat CR-01 (Tap to Pay / carte générique non op.)
OPTION_CARD = OPTION_CARD_TAP

_PRIVATE_DEBTORS = frozenset({"patient", "client", "account_holder"})
_INSTITUTION_DEBTORS = frozenset({"clinic", "institution"})
_DEMO_KINDS = frozenset({"applicable_demonstration", "specimen"})
_REAL_APPLICABLE_KINDS = frozenset({"applicable"})


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
    convention_legally_effective: bool = False
    tap_to_pay_operational: bool = False
    invoice_recipient_changed: bool = False
    debtor_id: int | None = None
    debtor_label: str | None = None
    # Rétrocompat CR-01
    card_provider_operational: bool | None = None


@dataclass(frozen=True, slots=True)
class ClosureOption:
    code: str
    label: str
    records_payment: bool
    authorizes_deferred_billing: bool
    opens_incident: bool
    available: bool
    reason: str | None = None
    channel: str | None = None


@dataclass(frozen=True, slots=True)
class ClosureMenu:
    """Menu chauffeur. Le téléphone ne reconstruit pas ces règles."""

    applicable: bool
    options: tuple[ClosureOption, ...]
    mission_close_without_payment_allowed_via_incident: bool
    debtor_preserved_on_recipient_change: bool
    institution_uses_own_rules: bool
    reasons: tuple[str, ...]
    transport_complete_without_settlement_allowed: bool = True


def _tap_operational(ctx: ClosureContext) -> bool:
    if ctx.card_provider_operational is not None:
        return bool(ctx.card_provider_operational)
    return bool(ctx.tap_to_pay_operational)


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
            transport_complete_without_settlement_allowed=False,
        )
    if payer in _INSTITUTION_DEBTORS:
        return ClosureMenu(
            applicable=False,
            options=(),
            mission_close_without_payment_allowed_via_incident=False,
            debtor_preserved_on_recipient_change=True,
            institution_uses_own_rules=True,
            reasons=("institution_own_rules",),
            transport_complete_without_settlement_allowed=False,
        )
    if payer not in _PRIVATE_DEBTORS:
        return ClosureMenu(
            applicable=False,
            options=(),
            mission_close_without_payment_allowed_via_incident=False,
            debtor_preserved_on_recipient_change=True,
            institution_uses_own_rules=False,
            reasons=("payer_not_private_client",),
            transport_complete_without_settlement_allowed=False,
        )
    if ctx.payment_received:
        return ClosureMenu(
            applicable=True,
            options=(),
            mission_close_without_payment_allowed_via_incident=False,
            debtor_preserved_on_recipient_change=True,
            institution_uses_own_rules=False,
            reasons=("already_paid",),
            transport_complete_without_settlement_allowed=False,
        )

    options: list[ClosureOption] = []
    real_deferred = (
        ctx.mandate_kind in _REAL_APPLICABLE_KINDS
        and ctx.convention_legally_effective is True
    )
    demo_kind = ctx.mandate_kind in _DEMO_KINDS

    if real_deferred:
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
    elif demo_kind:
        # Visible pour préparation / recette — jamais une autorisation réelle.
        options.append(
            ClosureOption(
                code=OPTION_DEFERRED,
                label="Facturation différée (convention de démonstration)",
                records_payment=False,
                authorizes_deferred_billing=False,
                opens_incident=False,
                available=True,
                reason="demo_convention_not_legally_effective",
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
            channel="cash",
        )
    )

    # Terminal bancaire externe à LIRIE — disponible dès confirmation chauffeur.
    options.append(
        ClosureOption(
            code=OPTION_CARD_EXTERNAL,
            label="Paiement carte (terminal bancaire externe)",
            records_payment=True,
            authorizes_deferred_billing=False,
            opens_incident=False,
            available=True,
            channel="external_terminal",
        )
    )

    # Tap to Pay simulé LIRIE — non opérationnel.
    tap_ok = _tap_operational(ctx)
    options.append(
        ClosureOption(
            code=OPTION_CARD_TAP,
            label="Paiement carte (Tap to Pay LIRIE)",
            records_payment=False,
            authorizes_deferred_billing=False,
            opens_incident=False,
            available=tap_ok,
            reason=None if tap_ok else "tap_to_pay_not_operational",
            channel="tap_to_pay",
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
        transport_complete_without_settlement_allowed=True,
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


def demo_convention_blocks_real_deferred(
    *,
    mandate_kind: str,
    convention_legally_effective: bool,
) -> bool:
    """True si une convention démo tenterait d'autoriser un différé réel."""
    if convention_legally_effective:
        return False
    return mandate_kind in _DEMO_KINDS or mandate_kind == "applicable_demonstration"
