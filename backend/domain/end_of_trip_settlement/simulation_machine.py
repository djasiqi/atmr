"""Transitions du dossier fictif. Aucun paiement, aucune facture.

Le prestataire est un résultat fourni au scénario. Il n'y a pas d'appel
réseau. Une clé déjà vue est rejouée. Une seconde tentative pendant
une opération ouverte est refusée sans nouvel enregistrement.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from domain.end_of_trip_settlement.prototypes import (
    TAP_CONFIRMED,
    TAP_IN_PROGRESS,
    TAP_INTERRUPTED,
    TAP_REFUSED,
    TAP_UNCERTAIN,
    open_tap_attempt,
    resolve_tap_simulation,
)

PREPARING = "preparing"
IN_PROGRESS = "operation_in_progress"
CONFIRMED = "simulated_confirmed"
REFUSED = "simulated_refused"
UNCERTAIN = "uncertain"
ADMINISTRATIVE = "administrative_simulated"
CLOSED = "closed"

_STATES = frozenset(
    {PREPARING, IN_PROGRESS, CONFIRMED, REFUSED, UNCERTAIN, ADMINISTRATIVE, CLOSED}
)
_OPEN_TAP = frozenset({IN_PROGRESS, UNCERTAIN})
_TAP_WORDS = {
    "confirmed": TAP_CONFIRMED,
    "refused": TAP_REFUSED,
    "uncertain": TAP_UNCERTAIN,
    "interrupted": TAP_INTERRUPTED,
}
_CASH_DISPOSITIONS = frozenset({"returned", "held", "contested", "administrative"})
_INVOICE_EVENTS = frozenset(
    {
        "prior_information_confirmed",
        "prior_information_missing",
        "simulated_acceptance",
        "refused",
        "unable_to_sign",
        "comprehension_difficulty",
        "unverified_representative",
        "contested_payer",
        "administrative_handoff",
    }
)
_COMPANY_CASH = frozenset({"receive", "hold", "remit", "reconcile"})
_CENT = Decimal("0.01")


class SimulationRefusal(Exception):
    """Transition interdite. L'appelant annule la transaction."""

    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


@dataclass(frozen=True, slots=True)
class MachineResult:
    state: str
    mode: str | None
    replay: bool
    event_action: str
    payload: dict[str, object]
    attempt_id: str | None


def _money(value: object) -> Decimal:
    amount = Decimal(str(value)).quantize(_CENT)
    if amount < 0:
        raise SimulationRefusal("montant_invalide", 400)
    return amount


def mock_provider_result(word: str) -> str:
    """Traduit un scénario fictif. Ne contacte aucun prestataire."""
    try:
        return _TAP_WORDS[word]
    except KeyError as exc:
        raise SimulationRefusal("resultat_simule_inconnu", 400) from exc


def _mode_available(state: str, mode: str | None, wanted: str) -> None:
    if mode in {None, wanted}:
        return
    if state == REFUSED:
        return
    raise SimulationRefusal("autre_mode_deja_choisi")


def start_tap(
    *,
    state: str,
    mode: str | None,
    attempt_id: str | None,
    amount_chf: Decimal,
    merchant_company_id: int,
    new_attempt_id: str,
) -> MachineResult:
    if state in {CONFIRMED, CLOSED}:
        raise SimulationRefusal("deja_simule")
    _mode_available(state, mode, "tap_to_pay")
    if state in _OPEN_TAP:
        raise SimulationRefusal("tentative_deja_ouverte")
    attempt = open_tap_attempt(
        current_state="ready",
        current_attempt_id=attempt_id,
        amount_chf=amount_chf,
        currency="CHF",
        merchant_company_id=merchant_company_id,
        new_attempt_id=new_attempt_id,
    )
    if attempt.may_start_another_debit or attempt.state != TAP_IN_PROGRESS:
        raise SimulationRefusal("tentative_refusee")
    return MachineResult(
        state=IN_PROGRESS,
        mode="tap_to_pay",
        replay=False,
        event_action="start_tap",
        payload={"provider": "mock", "real_psp": False},
        attempt_id=attempt.attempt_id,
    )


def resolve_tap(
    *,
    state: str,
    attempt_id: str | None,
    amount_chf: Decimal,
    merchant_company_id: int,
    simulated_result: str,
) -> MachineResult:
    if state != IN_PROGRESS or not attempt_id:
        raise SimulationRefusal("aucune_operation_en_cours")
    current = open_tap_attempt(
        current_state="ready",
        current_attempt_id=None,
        amount_chf=amount_chf,
        currency="CHF",
        merchant_company_id=merchant_company_id,
        new_attempt_id=attempt_id,
    )
    resolved = resolve_tap_simulation(current, mock_provider_result(simulated_result))
    if resolved.may_start_another_debit:
        raise SimulationRefusal("second_debit_interdit")
    mapping = {
        TAP_CONFIRMED: CONFIRMED,
        TAP_REFUSED: REFUSED,
        TAP_UNCERTAIN: UNCERTAIN,
        TAP_INTERRUPTED: UNCERTAIN,
    }
    return MachineResult(
        state=mapping[resolved.state],
        mode="tap_to_pay",
        replay=False,
        event_action="resolve_tap",
        payload={
            "provider": "mock",
            "simulated_result": simulated_result,
            "real_psp": False,
            "second_debit": False,
        },
        attempt_id=resolved.attempt_id,
    )


def recover_tap(*, state: str, attempt_id: str | None) -> MachineResult:
    """Relit l'état serveur. Ne lance pas une seconde opération."""
    if state not in _STATES:
        raise SimulationRefusal("etat_inconnu")
    if attempt_id is None and state in _OPEN_TAP:
        raise SimulationRefusal("tentative_introuvable")
    return MachineResult(
        state=state,
        mode="tap_to_pay",
        replay=True,
        event_action="recover_tap",
        payload={"second_debit": False, "provider": "mock"},
        attempt_id=attempt_id,
    )


def declare_cash(
    *,
    state: str,
    mode: str | None,
    expected_chf: Decimal,
    tendered_chf: object,
    change_chf: object,
    change_confirmed: bool,
) -> MachineResult:
    if state not in {PREPARING, REFUSED}:
        raise SimulationRefusal("especes_deja_enregistrees")
    _mode_available(state, mode, "cash")
    tendered = _money(tendered_chf)
    change = _money(change_chf)
    if tendered <= 0:
        raise SimulationRefusal("montant_invalide", 400)
    if change > 0 and not change_confirmed:
        raise SimulationRefusal("monnaie_non_confirmee", 400)
    net = (tendered - change).quantize(_CENT)
    if net != expected_chf:
        raise SimulationRefusal("net_different_du_montant", 400)
    if net <= 0:
        raise SimulationRefusal("montant_invalide", 400)
    return MachineResult(
        state=CONFIRMED,
        mode="cash",
        replay=False,
        event_action="declare_cash",
        payload={
            "tendered_chf": format(tendered, "f"),
            "change_chf": format(change, "f"),
            "net_chf": format(net, "f"),
            "change_confirmed": change_confirmed,
            "real_cash_receipt": False,
        },
        attempt_id=None,
    )


def shortfall_cash(
    *,
    state: str,
    mode: str | None,
    expected_chf: Decimal,
    tendered_chf: object,
    disposition: str,
) -> MachineResult:
    if state not in {PREPARING, REFUSED}:
        raise SimulationRefusal("especes_deja_enregistrees")
    _mode_available(state, mode, "cash")
    if disposition not in _CASH_DISPOSITIONS:
        raise SimulationRefusal("suite_especes_inconnue", 400)
    tendered = _money(tendered_chf)
    if tendered <= 0 or tendered >= expected_chf:
        raise SimulationRefusal("montant_insuffisant_attendu", 400)
    noted = (expected_chf - tendered).quantize(_CENT)
    next_state = CLOSED if disposition == "returned" else ADMINISTRATIVE
    return MachineResult(
        state=next_state,
        mode="cash",
        replay=False,
        event_action="cash_shortfall",
        payload={
            "tendered_chf": format(tendered, "f"),
            "expected_chf": format(expected_chf, "f"),
            "noted_gap_chf": format(noted, "f"),
            "disposition": disposition,
            "held_is_not_void": disposition == "held",
            "full_receivable_created": False,
            "real_cash_receipt": False,
            "decision_h": "draft",
        },
        attempt_id=None,
    )


def invoice_event(
    *,
    state: str,
    mode: str | None,
    kind: str,
    contractual_basis: str,
) -> MachineResult:
    if kind not in _INVOICE_EVENTS:
        raise SimulationRefusal("evenement_facture_inconnu", 400)
    if state in {CONFIRMED, CLOSED}:
        raise SimulationRefusal("dossier_deja_clos")
    _mode_available(state, mode, "invoice")
    accepted = kind == "simulated_acceptance" and contractual_basis == "known_draft"
    next_state = CONFIRMED if accepted else ADMINISTRATIVE
    if kind == "refused":
        next_state = REFUSED
    return MachineResult(
        state=next_state,
        mode="invoice",
        replay=False,
        event_action="invoice_event",
        payload={
            "kind": kind,
            "contractual_basis": contractual_basis,
            "document_version": "2026-10-26.v1-draft",
            "signature_recorded": False,
            "real_invoice": False,
            "patient_made_debtor": False,
        },
        attempt_id=None,
    )


def company_cash(
    *,
    state: str,
    mode: str | None,
    step: str,
    actor_role: str,
) -> MachineResult:
    if actor_role != "company":
        raise SimulationRefusal("rapprochement_reserve_entreprise", 403)
    if mode != "cash":
        raise SimulationRefusal("suivi_especes_sans_dossier")
    if step not in _COMPANY_CASH:
        raise SimulationRefusal("suivi_especes_inconnu", 400)
    if state not in {CONFIRMED, ADMINISTRATIVE, CLOSED}:
        raise SimulationRefusal("suivi_premature")
    return MachineResult(
        state=state,
        mode="cash",
        replay=False,
        event_action=f"company_cash_{step}",
        payload={
            "step": step,
            "cash_book": False,
            "real_payment": False,
        },
        attempt_id=None,
    )
