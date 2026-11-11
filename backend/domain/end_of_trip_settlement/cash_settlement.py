"""Règles d'espèces. La décision H n'est pas approuvée.

La restitution est le scénario normal d'un montant insuffisant.
La conservation et la contestation restent administratives.
L'imputation est refusée.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class CashDecision:
    outcome: str
    net_chf: Decimal
    change_chf: Decimal
    receipt_allowed: bool
    imputed: bool
    held_is_not_void: bool
    full_receivable_created: bool
    decision_h: str


def money(value: object) -> Decimal:
    try:
        amount = Decimal(str(value))
    except Exception as exc:
        raise ValueError("montant_illisible") from exc
    if amount != amount.quantize(_CENT):
        raise ValueError("montant_illisible")
    return amount.quantize(_CENT)


def evaluate_cash(
    *,
    due: Decimal,
    tendered: Decimal,
    change: Decimal,
    change_confirmed: bool,
    disposition: str | None,
) -> CashDecision:
    """Calcule le résultat. N'écrit rien."""
    if due <= 0 or tendered < 0 or change < 0:
        raise ValueError("montant_incoherent")
    if tendered < due:
        choice = disposition or ""
        if choice == "returned":
            return CashDecision(
                outcome="returned",
                net_chf=Decimal("0.00"),
                change_chf=tendered,
                receipt_allowed=False,
                imputed=False,
                held_is_not_void=False,
                full_receivable_created=False,
                decision_h="draft",
            )
        if choice in {"held", "contested"}:
            return CashDecision(
                outcome="administrative",
                net_chf=Decimal("0.00"),
                change_chf=Decimal("0.00"),
                receipt_allowed=False,
                imputed=False,
                held_is_not_void=True,
                full_receivable_created=False,
                decision_h="draft",
            )
        if choice == "imputed":
            raise ValueError("imputation_non_approuvee")
        if choice == "partial":
            if tendered <= 0:
                raise ValueError("montant_incoherent")
            if change != 0:
                raise ValueError("monnaie_incorrecte")
            return CashDecision(
                outcome="administrative",
                net_chf=tendered,
                change_chf=Decimal("0.00"),
                # Spécimen de la somme reçue. Pas une quittance émise, pas un solde soldé.
                receipt_allowed=True,
                imputed=False,
                held_is_not_void=True,
                full_receivable_created=False,
                decision_h="draft",
            )
        raise ValueError("disposition_requise")
    expected_change = (tendered - due).quantize(_CENT)
    if change != expected_change:
        raise ValueError("monnaie_incorrecte")
    if expected_change > 0 and not change_confirmed:
        raise ValueError("monnaie_non_confirmee")
    return CashDecision(
        outcome="received",
        net_chf=due,
        change_chf=expected_change,
        receipt_allowed=True,
        imputed=False,
        held_is_not_void=False,
        full_receivable_created=False,
        decision_h="not_applicable",
    )
