"""Registre applicatif des incidents de non-règlement.

Stockage en mémoire pour la recette isolée. Pas d'encaissement fictif,
pas de signature inventée, pas d'autorisation différée implicite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from domain.end_of_trip_settlement.settlement_incident import (
    IncidentDecision,
    SettlementIncident,
    SettlementIncidentDraft,
    apply_admin_status,
    driver_may_change_admin_status,
    driver_may_delete,
    open_incident,
    record_later_payment_note,
)


@dataclass
class SettlementIncidentRegistry:
    """File admin par entreprise. Isolation stricte."""

    _by_key: dict[str, SettlementIncident] = field(default_factory=dict)
    _by_id: dict[str, SettlementIncident] = field(default_factory=dict)

    def _scope_key(self, company_id: int, idempotency_key: str) -> str:
        return f"{company_id}:{idempotency_key.strip()}"

    def open(
        self,
        draft: SettlementIncidentDraft,
        *,
        actor_company_id: int,
    ) -> IncidentDecision:
        key = self._scope_key(draft.company_id, draft.idempotency_key)
        existing = self._by_key.get(key)
        decision = open_incident(
            draft,
            existing_for_key=existing,
            actor_company_id=actor_company_id,
        )
        if (
            decision.accepted
            and decision.incident is not None
            and not decision.duplicate
        ):
            self._by_key[key] = decision.incident
            self._by_id[decision.incident.id] = decision.incident
        return decision

    def list_for_company(self, company_id: int) -> tuple[SettlementIncident, ...]:
        return tuple(
            item
            for item in self._by_id.values()
            if int(item.company_id) == int(company_id)
        )

    def get_for_company(
        self, company_id: int, incident_id: str
    ) -> SettlementIncident | None:
        item = self._by_id.get(incident_id)
        if item is None or int(item.company_id) != int(company_id):
            return None
        return item

    def admin_set_status(
        self,
        *,
        company_id: int,
        incident_id: str,
        new_status: str,
        actor_id: int,
        justification: str,
    ) -> IncidentDecision:
        current = self.get_for_company(company_id, incident_id)
        if current is None:
            return IncidentDecision(accepted=False, incident=None, reason="introuvable")
        decision = apply_admin_status(
            current,
            new_status=new_status,
            actor_company_id=company_id,
            actor_id=actor_id,
            justification=justification,
        )
        if decision.accepted and decision.incident is not None:
            self._by_id[incident_id] = decision.incident
            self._by_key[
                self._scope_key(company_id, decision.incident.idempotency_key)
            ] = decision.incident
        return decision

    def admin_note_payment(
        self,
        *,
        company_id: int,
        incident_id: str,
        actor_id: int,
        note: str,
    ) -> IncidentDecision:
        current = self.get_for_company(company_id, incident_id)
        if current is None:
            return IncidentDecision(accepted=False, incident=None, reason="introuvable")
        decision = record_later_payment_note(
            current,
            actor_company_id=company_id,
            actor_id=actor_id,
            note=note,
        )
        if decision.accepted and decision.incident is not None:
            self._by_id[incident_id] = decision.incident
            self._by_key[
                self._scope_key(company_id, decision.incident.idempotency_key)
            ] = decision.incident
        return decision

    @staticmethod
    def driver_delete_refused(incident: SettlementIncident) -> bool:
        return not driver_may_delete(incident)

    @staticmethod
    def driver_admin_status_refused(incident: SettlementIncident) -> bool:
        return not driver_may_change_admin_status(incident)


@dataclass(frozen=True, slots=True)
class CardAttemptResult:
    """Résultat carte. Aucun prestataire électronique n'est appelé."""

    attempted: bool
    operational: bool
    payment_recorded: bool
    status: str
    reason: str


def attempt_card_payment(*, provider_operational: bool) -> CardAttemptResult:
    """Tap to Pay simulé : non opérationnel → non payé, pas d'encaissement fictif."""
    if not provider_operational:
        return CardAttemptResult(
            attempted=True,
            operational=False,
            payment_recorded=False,
            status="unpaid",
            reason="card_provider_not_operational",
        )
    # Branche réservée : aucun prestataire n'est branché dans ce lot.
    return CardAttemptResult(
        attempted=True,
        operational=True,
        payment_recorded=False,
        status="unpaid",
        reason="card_provider_not_wired",
    )


def cash_payment_recorded(*, amount_due_chf: Decimal, tendered_chf: Decimal) -> bool:
    """Espèces : paiement seulement si montant intégralement remis."""
    return tendered_chf >= amount_due_chf > 0
