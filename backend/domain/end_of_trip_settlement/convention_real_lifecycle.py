"""Cycle de vie d'une convention Client réelle (hors démonstration).

Une signature à l'écran ou un scan papier ne rendent pas automatiquement
la convention juridiquement applicable pour la facturation différée.
``legally_effective`` reste fermé tant que l'activation juridique n'est
pas autorisée explicitement.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUSES = (
    "draft",
    "awaiting_signature",
    "signed_pending_validation",
    "applicable",
    "expired",
    "revoked",
    "replaced",
)

_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"awaiting_signature", "revoked", "replaced"}),
    "awaiting_signature": frozenset(
        {"signed_pending_validation", "draft", "revoked", "replaced"}
    ),
    "signed_pending_validation": frozenset(
        {"applicable", "revoked", "awaiting_signature", "replaced"}
    ),
    "applicable": frozenset({"expired", "revoked", "replaced"}),
    "expired": frozenset({"revoked", "replaced"}),
    "revoked": frozenset(),
    "replaced": frozenset(),
}

SIGNATURE_MODALITIES = ("none", "on_screen_trace", "paper_scan")


@dataclass(frozen=True, slots=True)
class DeferredGate:
    """Décision technique de différé. Pas une certification juridique."""

    mandate_kind: str
    legally_effective: bool
    authorizes_deferred_billing: bool
    reason: str


class ConventionLifecycleRefusal(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def assert_transition(current: str, target: str) -> None:
    if current not in _TRANSITIONS:
        raise ConventionLifecycleRefusal(
            "statut_inconnu", f"Statut contractuel inconnu : {current}."
        )
    if target not in _TRANSITIONS.get(current, frozenset()):
        raise ConventionLifecycleRefusal(
            "transition_refusee",
            f"Passage de « {current} » vers « {target} » refusé.",
        )


def signature_alone_is_not_applicable(
    *,
    modality: str,
    consent_acknowledged: bool,
    document_sha256: str | None,
    archive_reference: str | None,
) -> bool:
    """True si les seules preuves de signature ne suffisent pas à l'applicabilité."""
    if modality in {"on_screen_trace", "paper_scan"}:
        return True
    if not consent_acknowledged:
        return True
    return bool(not document_sha256 or not archive_reference)


def deferred_billing_gate(
    *,
    status: str,
    legally_effective: bool,
    legal_activation_unlocked: bool,
    is_demo_row: bool = False,
) -> DeferredGate:
    """Barrière différé. Fermée tant que l'activation juridique est off."""
    if is_demo_row:
        return DeferredGate(
            mandate_kind="applicable_demonstration",
            legally_effective=False,
            authorizes_deferred_billing=False,
            reason="demo_convention_not_legally_effective",
        )
    if status in {"draft", "awaiting_signature"}:
        return DeferredGate(
            mandate_kind="prepared",
            legally_effective=False,
            authorizes_deferred_billing=False,
            reason="convention_not_signed",
        )
    if status == "signed_pending_validation":
        return DeferredGate(
            mandate_kind="prepared",
            legally_effective=False,
            authorizes_deferred_billing=False,
            reason="signature_pending_admin_validation",
        )
    if status in {"expired", "revoked", "replaced"}:
        return DeferredGate(
            mandate_kind="expired" if status == "expired" else "absent",
            legally_effective=False,
            authorizes_deferred_billing=False,
            reason=f"convention_{status}",
        )
    if status != "applicable":
        return DeferredGate(
            mandate_kind="absent",
            legally_effective=False,
            authorizes_deferred_billing=False,
            reason="convention_not_applicable",
        )
    # Statut applicable : différé réel seulement si effet juridique + activation.
    if not legal_activation_unlocked:
        return DeferredGate(
            mandate_kind="applicable",
            legally_effective=False,
            authorizes_deferred_billing=False,
            reason="legal_activation_closed",
        )
    if not legally_effective:
        return DeferredGate(
            mandate_kind="applicable",
            legally_effective=False,
            authorizes_deferred_billing=False,
            reason="legally_effective_false",
        )
    return DeferredGate(
        mandate_kind="applicable",
        legally_effective=True,
        authorizes_deferred_billing=True,
        reason="authorized",
    )


def frozen_document_must_not_change(
    *,
    previous_sha256: str,
    candidate_sha256: str,
    status: str,
) -> None:
    """Interdit de réécrire silencieusement un document déjà signé/archivé."""
    if (
        status
        in {
            "signed_pending_validation",
            "applicable",
            "expired",
            "revoked",
            "replaced",
        }
        and previous_sha256
        and previous_sha256 != candidate_sha256
    ):
        raise ConventionLifecycleRefusal(
            "document_fige",
            "Un document signé ou archivé ne peut pas être modifié sans nouvelle version.",
        )
