"""Gel du bloc « Facturé à » quand une facture quitte l'état brouillon.

Toute transition ``DRAFT → autre statut`` (envoi e-mail, courrier papier, envoi en
lot, paiement direct, annulation) fige ``invoice.meta["billed_to_snapshot"]`` : le
snapshot déjà capturé lors du dernier PDF construit est marqué figé ; à défaut, il
est capturé maintenant depuis les master data. Après cela, le rendu PDF/HTML lit
exclusivement ce snapshot (voir ``services.documents.invoice_recipient``).

Un seul point d'accrochage (``Session.before_flush``) plutôt qu'un appel par route :
aucune transition ne peut l'oublier. Même patron que
``services.demo.soft_delete_guard.register_demo_soft_delete_guard``.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import event, inspect
from sqlalchemy.orm import Session

from models import Invoice

logger = logging.getLogger(__name__)

_LISTENER_REGISTERED = False
_DRAFT = "draft"


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def invoice_leaves_draft(invoice: Invoice) -> tuple[bool, str, str]:
    """(transition DRAFT → non-DRAFT ?, ancien statut, nouveau statut)."""
    history = inspect(invoice).attrs.status.history
    if not history.has_changes():
        return False, "", ""
    old = _status_value(history.deleted[0]) if history.deleted else ""
    new = (
        _status_value(history.added[0])
        if history.added
        else _status_value(invoice.status)
    )
    return (old == _DRAFT and bool(new) and new != _DRAFT), old, new


def freeze_invoice_if_leaving_draft(invoice: Invoice) -> bool:
    """Fige billed_to + débiteur/créancier QR si ``invoice`` quitte DRAFT. True si gel effectué."""
    leaves, old, new = invoice_leaves_draft(invoice)
    if not leaves:
        return False
    from services.documents.invoice_recipient import freeze_billed_to_snapshot

    try:
        freeze_billed_to_snapshot(invoice, reason=f"status:{old}->{new}")
        return True
    except Exception:  # pragma: no cover - ne bloque jamais la transition
        logger.exception(
            "[Facturé à] Gel du snapshot impossible (invoice_id=%s, %s→%s) : "
            "la facture sera rendue en legacy_live.",
            getattr(invoice, "id", None),
            old,
            new,
        )
        return False


def freeze_billed_to_for_session(session: Session) -> int:
    """Fige le bloc des factures de la session qui quittent DRAFT ; retourne le nombre."""
    frozen = 0
    for obj in list(session.dirty):
        if not isinstance(obj, Invoice):
            continue
        if freeze_invoice_if_leaving_draft(obj):
            frozen += 1
    return frozen


def _invoice_before_update(_mapper: Any, _connection: Any, target: Invoice) -> None:
    freeze_invoice_if_leaving_draft(target)


def register_billed_to_snapshot_guard() -> None:
    """Enregistre le gel : mapper ``Invoice`` (tout process qui importe le modèle)
    **et** ``Session.before_flush`` (filet pour les sessions Flask/Celery).

    Indépendant de ``create_app`` : ``models.invoice`` accroche aussi le mapper.
    """
    global _LISTENER_REGISTERED
    if _LISTENER_REGISTERED:
        return

    event.listen(Invoice, "before_update", _invoice_before_update)

    @event.listens_for(Session, "before_flush")
    def _billed_to_before_flush(
        session: Session, _flush_context: Any, _instances: Any
    ) -> None:
        frozen = freeze_billed_to_for_session(session)
        if frozen:
            logger.info("[Facturé à] snapshots figés à la sortie de DRAFT : %s", frozen)

    _LISTENER_REGISTERED = True


def billed_to_guard_is_registered() -> bool:
    return bool(_LISTENER_REGISTERED) or event.contains(
        Invoice, "before_update", _invoice_before_update
    )
