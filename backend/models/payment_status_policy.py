"""Transitions autorisées d'un paiement.

Un paiement confirmé reste confirmé. Le déclarer échoué ou en attente
exigerait un événement financier approuvé, qui n'existe pas encore.
Aucun remboursement n'est exécuté ici.

La base n'empêche pas encore un UPDATE SQL de payment.status. Le
déclencheur étudié n'est pas installé : une correction manuelle d'un
paiement déjà confirmé exigerait une procédure approuvée, sans variable
de session capable de contourner la règle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from models.enums import PaymentStatus

DATABASE_STATUS_GUARD_ENABLED = False

_OPEN = frozenset({"PENDING", "COMPLETED", "FAILED"})
_ALLOWED = {
    None: _OPEN,
    "PENDING": _OPEN,
    "FAILED": frozenset({"FAILED", "COMPLETED"}),
    "COMPLETED": frozenset({"COMPLETED"}),
}


class PaymentStatusTransitionError(ValueError):
    """Le statut demandé n'est pas une transition financière autorisée."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def status_name(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    name = str(raw).strip().upper()
    return name or None


def assert_payment_status_transition(current: object, target: PaymentStatus) -> None:
    previous = status_name(current)
    nxt = status_name(target)
    if nxt not in _OPEN:
        raise PaymentStatusTransitionError("statut_inconnu")
    if nxt not in _ALLOWED.get(previous, frozenset()):
        raise PaymentStatusTransitionError("transition_illegitime")


def remember_payment_status(
    payment: object,
    current: object,
    target: PaymentStatus,
    *,
    allowed: bool,
) -> None:
    """Ajoute une trace dans la transaction courante, sans la valider."""
    from ext import db
    from security.audit_log import AuditLog

    previous = status_name(current)
    nxt = status_name(target)
    if allowed and (previous is None or previous == nxt):
        return
    row = AuditLog()
    row.user_type = "system"
    row.action_type = "payment_status_applied" if allowed else "payment_status_refused"
    row.action_category = "payment"
    row.action_details = json.dumps(
        {
            "payment_id": getattr(payment, "id", None),
            "booking_id": getattr(payment, "booking_id", None),
            "from_status": previous,
            "to_status": nxt,
        },
        sort_keys=True,
    )
    row.result_status = "success" if allowed else "failure"
    row.result_message = "transition_autorisee" if allowed else "transition_illegitime"
    row.additional_metadata = "{}"
    row.created_at = datetime.now(UTC)
    db.session.add(row)
