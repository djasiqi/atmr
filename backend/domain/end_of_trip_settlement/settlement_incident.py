"""Signalement de non-règlement en fin de course.

Ce signalement n'est ni un paiement, ni une signature, ni une autorisation
de facturation différée. Il permet de terminer la mission et ouvre un
dossier administratif.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

MOTIF_REFUS_SIGNER_PAYER = "refus_signer_et_payer"
MOTIF_IMPOSSIBLE_REGLER = "impossibilite_regler"
MOTIF_CONTESTATION_MONTANT = "contestation_montant"
MOTIF_MOYEN_INDISPONIBLE = "moyen_paiement_indisponible_ou_echoue"
MOTIF_AUTRE = "autre"

MOTIFS = frozenset(
    {
        MOTIF_REFUS_SIGNER_PAYER,
        MOTIF_IMPOSSIBLE_REGLER,
        MOTIF_CONTESTATION_MONTANT,
        MOTIF_MOYEN_INDISPONIBLE,
        MOTIF_AUTRE,
    }
)

STATUS_A_TRAITER = "a_traiter"
STATUS_EN_COURS = "en_cours_examen"
STATUS_REGULARISE = "regularise"
STATUS_CLOTURE = "cloture_justifie"

STATUSES = frozenset(
    {
        STATUS_A_TRAITER,
        STATUS_EN_COURS,
        STATUS_REGULARISE,
        STATUS_CLOTURE,
    }
)

_ADMIN_TRANSITIONS = {
    STATUS_A_TRAITER: frozenset({STATUS_EN_COURS, STATUS_REGULARISE, STATUS_CLOTURE}),
    STATUS_EN_COURS: frozenset({STATUS_REGULARISE, STATUS_CLOTURE, STATUS_A_TRAITER}),
    STATUS_REGULARISE: frozenset({STATUS_CLOTURE}),
    STATUS_CLOTURE: frozenset(),
}


@dataclass(frozen=True, slots=True)
class SettlementIncidentDraft:
    booking_id: int
    company_id: int
    driver_id: int
    client_id: int | None
    debtor_id: int | None
    debtor_label: str | None
    amount_due_chf: Decimal
    transport_date: str
    motif: str
    observation: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class SettlementIncidentEvent:
    at: str
    actor_role: str
    actor_id: int | None
    action: str
    detail: str


@dataclass(frozen=True, slots=True)
class SettlementIncident:
    """Dossier administratif. Le chauffeur ne peut pas le supprimer."""

    id: str
    booking_id: int
    company_id: int
    driver_id: int
    client_id: int | None
    debtor_id: int | None
    debtor_label: str | None
    amount_due_chf: Decimal
    transport_date: str
    motif: str
    observation: str
    status: str
    idempotency_key: str
    payment_recorded: bool
    signature_recorded: bool
    deferred_billing_authorized: bool
    mission_close_allowed: bool
    history: tuple[SettlementIncidentEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class IncidentDecision:
    accepted: bool
    incident: SettlementIncident | None
    reason: str
    duplicate: bool = False


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def validate_draft(draft: SettlementIncidentDraft) -> str | None:
    if draft.booking_id < 1 or draft.company_id < 1 or draft.driver_id < 1:
        return "identifiants_invalides"
    if draft.motif not in MOTIFS:
        return "motif_inconnu"
    if draft.motif == MOTIF_AUTRE and not draft.observation.strip():
        return "observation_requise_pour_autre"
    if not draft.idempotency_key.strip():
        return "idempotency_requise"
    if draft.amount_due_chf < 0:
        return "montant_invalide"
    # Observation factuelle : pas d'accusation automatique codée ici.
    return None


def open_incident(
    draft: SettlementIncidentDraft,
    *,
    existing_for_key: SettlementIncident | None,
    actor_company_id: int,
) -> IncidentDecision:
    """Crée ou rejoue un signalement. Isolation entreprise stricte."""
    if int(actor_company_id) != int(draft.company_id):
        return IncidentDecision(
            accepted=False,
            incident=None,
            reason="company_isolation_refused",
        )
    err = validate_draft(draft)
    if err:
        return IncidentDecision(accepted=False, incident=None, reason=err)
    if existing_for_key is not None:
        return IncidentDecision(
            accepted=True,
            incident=existing_for_key,
            reason="idempotent_replay",
            duplicate=True,
        )
    created = SettlementIncident(
        id=f"si-{draft.booking_id}-{draft.idempotency_key.strip()}",
        booking_id=draft.booking_id,
        company_id=draft.company_id,
        driver_id=draft.driver_id,
        client_id=draft.client_id,
        debtor_id=draft.debtor_id,
        debtor_label=draft.debtor_label,
        amount_due_chf=draft.amount_due_chf,
        transport_date=draft.transport_date,
        motif=draft.motif,
        observation=draft.observation.strip(),
        status=STATUS_A_TRAITER,
        idempotency_key=draft.idempotency_key.strip(),
        payment_recorded=False,
        signature_recorded=False,
        deferred_billing_authorized=False,
        mission_close_allowed=True,
        history=(
            SettlementIncidentEvent(
                at=_now(),
                actor_role="driver",
                actor_id=draft.driver_id,
                action="opened",
                detail=draft.motif,
            ),
        ),
    )
    return IncidentDecision(
        accepted=True,
        incident=created,
        reason="incident_opened",
    )


def driver_may_delete(_: SettlementIncident) -> bool:
    return False


def driver_may_change_admin_status(_: SettlementIncident) -> bool:
    return False


def apply_admin_status(
    incident: SettlementIncident,
    *,
    new_status: str,
    actor_company_id: int,
    actor_id: int,
    justification: str,
) -> IncidentDecision:
    if int(actor_company_id) != int(incident.company_id):
        return IncidentDecision(
            accepted=False, incident=incident, reason="company_isolation_refused"
        )
    if new_status not in STATUSES:
        return IncidentDecision(
            accepted=False, incident=incident, reason="statut_inconnu"
        )
    allowed = _ADMIN_TRANSITIONS.get(incident.status, frozenset())
    if new_status not in allowed:
        return IncidentDecision(
            accepted=False, incident=incident, reason="transition_refusee"
        )
    if new_status in {STATUS_REGULARISE, STATUS_CLOTURE} and not justification.strip():
        return IncidentDecision(
            accepted=False, incident=incident, reason="justification_requise"
        )
    updated = SettlementIncident(
        id=incident.id,
        booking_id=incident.booking_id,
        company_id=incident.company_id,
        driver_id=incident.driver_id,
        client_id=incident.client_id,
        debtor_id=incident.debtor_id,
        debtor_label=incident.debtor_label,
        amount_due_chf=incident.amount_due_chf,
        transport_date=incident.transport_date,
        motif=incident.motif,
        observation=incident.observation,
        status=new_status,
        idempotency_key=incident.idempotency_key,
        payment_recorded=incident.payment_recorded,
        signature_recorded=incident.signature_recorded,
        deferred_billing_authorized=False,
        mission_close_allowed=incident.mission_close_allowed,
        history=(
            *incident.history,
            SettlementIncidentEvent(
                at=_now(),
                actor_role="company_admin",
                actor_id=actor_id,
                action=f"status:{new_status}",
                detail=justification.strip(),
            ),
        ),
    )
    return IncidentDecision(accepted=True, incident=updated, reason="status_updated")


def record_later_payment_note(
    incident: SettlementIncident,
    *,
    actor_company_id: int,
    actor_id: int,
    note: str,
) -> IncidentDecision:
    """Constate un paiement ultérieur sans inventer un encaissement chauffeur."""
    if int(actor_company_id) != int(incident.company_id):
        return IncidentDecision(
            accepted=False, incident=incident, reason="company_isolation_refused"
        )
    if not note.strip():
        return IncidentDecision(
            accepted=False, incident=incident, reason="note_requise"
        )
    updated = SettlementIncident(
        id=incident.id,
        booking_id=incident.booking_id,
        company_id=incident.company_id,
        driver_id=incident.driver_id,
        client_id=incident.client_id,
        debtor_id=incident.debtor_id,
        debtor_label=incident.debtor_label,
        amount_due_chf=incident.amount_due_chf,
        transport_date=incident.transport_date,
        motif=incident.motif,
        observation=incident.observation,
        status=incident.status,
        idempotency_key=incident.idempotency_key,
        payment_recorded=False,  # pas d'encaissement fictif chauffeur
        signature_recorded=False,
        deferred_billing_authorized=False,
        mission_close_allowed=incident.mission_close_allowed,
        history=(
            *incident.history,
            SettlementIncidentEvent(
                at=_now(),
                actor_role="company_admin",
                actor_id=actor_id,
                action="later_payment_noted",
                detail=note.strip(),
            ),
        ),
    )
    return IncidentDecision(
        accepted=True, incident=updated, reason="later_payment_noted"
    )
