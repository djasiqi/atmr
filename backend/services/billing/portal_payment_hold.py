"""Hold de paiement PORTAL dérivé des créances (scope créancier).

Politique : ``docs/contracts/portal-payment-hold-policy.md``.
Aucun flag global mutable sur le client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Iterable, Sequence

from models.portal_receivable import (
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_PAID,
    PortalReceivable,
)
from shared.time_utils import LOCAL_TZ, now_local

HOLD_STATE_CLEAR = "clear"
HOLD_STATE_HOLD = "hold"

HOLD_EFFECT_NONE = "none"
HOLD_EFFECT_CARRIER_BLOCKED = "carrier_blocked"

ERROR_PORTAL_CLIENT_PAYMENT_HOLD = "portal_client_payment_hold"


@dataclass(frozen=True, slots=True)
class PortalPaymentHoldResult:
    state: str
    creditor_company_id: int
    overdue_receivable_ids: tuple[int, ...]
    outstanding_balance: Decimal
    oldest_due_date: datetime | None

    @property
    def is_hold(self) -> bool:
        return self.state == HOLD_STATE_HOLD

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "creditor_company_id": self.creditor_company_id,
            "overdue_receivable_ids": list(self.overdue_receivable_ids),
            "outstanding_balance": float(self.outstanding_balance),
            "oldest_due_date": (
                self.oldest_due_date.isoformat() if self.oldest_due_date else None
            ),
        }


def business_calendar_date(value: datetime | date | None) -> date | None:
    """Date calendaire Europe/Zurich pour une échéance stockée."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        else:
            dt = dt.astimezone(LOCAL_TZ)
        return dt.date()
    return value


def current_business_date(as_of: date | datetime | None = None) -> date:
    """Date métier courante (Europe/Zurich), injectable pour les tests."""
    if as_of is None:
        return now_local().date()
    if isinstance(as_of, datetime):
        resolved = business_calendar_date(as_of)
        assert resolved is not None
        return resolved
    return as_of


def _money(value: object) -> Decimal:
    return Decimal(str(value or "0"))


def receivable_contributes_to_hold(
    receivable: PortalReceivable, *, as_of_date: date
) -> bool:
    """True si la créance bloque le couple débiteur/créancier à ``as_of_date``."""
    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        return False
    if receivable.status == RECEIVABLE_DISPUTED or receivable.disputed_at is not None:
        return False
    if receivable.status == RECEIVABLE_PAID or _money(receivable.balance_due) <= 0:
        return False
    due = business_calendar_date(receivable.due_date)
    return due is not None and due < as_of_date


def _result_from_rows(
    *,
    creditor_company_id: int,
    rows: Sequence[PortalReceivable],
    as_of_date: date,
) -> PortalPaymentHoldResult:
    overdue = [
        r for r in rows if receivable_contributes_to_hold(r, as_of_date=as_of_date)
    ]
    if not overdue:
        return PortalPaymentHoldResult(
            state=HOLD_STATE_CLEAR,
            creditor_company_id=int(creditor_company_id),
            overdue_receivable_ids=(),
            outstanding_balance=Decimal("0.00"),
            oldest_due_date=None,
        )
    balance = sum((_money(r.balance_due) for r in overdue), Decimal("0.00"))
    oldest = min(overdue, key=lambda r: business_calendar_date(r.due_date) or date.max)
    return PortalPaymentHoldResult(
        state=HOLD_STATE_HOLD,
        creditor_company_id=int(creditor_company_id),
        overdue_receivable_ids=tuple(int(r.id) for r in overdue),
        outstanding_balance=balance,
        oldest_due_date=oldest.due_date,
    )


def resolve_portal_payment_hold(
    debtor_user_id: int,
    creditor_company_id: int,
    *,
    as_of_date: date | datetime | None = None,
) -> PortalPaymentHoldResult:
    """Hold dérivé pour un couple débiteur / créancier."""
    as_of = current_business_date(as_of_date)
    rows = (
        PortalReceivable.query.filter_by(
            debtor_user_id=int(debtor_user_id),
            creditor_company_id=int(creditor_company_id),
        )
        .order_by(PortalReceivable.id.asc())
        .all()
    )
    return _result_from_rows(
        creditor_company_id=int(creditor_company_id),
        rows=rows,
        as_of_date=as_of,
    )


def resolve_portal_payment_holds_for_companies(
    debtor_user_id: int,
    company_ids: Iterable[int],
    *,
    as_of_date: date | datetime | None = None,
) -> dict[int, PortalPaymentHoldResult]:
    """Batch : une requête pour tous les créanciers candidats."""
    as_of = current_business_date(as_of_date)
    ids = sorted({int(cid) for cid in company_ids if cid is not None})
    empty: dict[int, PortalPaymentHoldResult] = {
        cid: PortalPaymentHoldResult(
            state=HOLD_STATE_CLEAR,
            creditor_company_id=cid,
            overdue_receivable_ids=(),
            outstanding_balance=Decimal("0.00"),
            oldest_due_date=None,
        )
        for cid in ids
    }
    if not ids:
        return empty

    rows = (
        PortalReceivable.query.filter(
            PortalReceivable.debtor_user_id == int(debtor_user_id),
            PortalReceivable.creditor_company_id.in_(ids),
        )
        .order_by(PortalReceivable.id.asc())
        .all()
    )
    by_creditor: dict[int, list[PortalReceivable]] = {cid: [] for cid in ids}
    for row in rows:
        by_creditor.setdefault(int(row.creditor_company_id), []).append(row)

    return {
        cid: _result_from_rows(
            creditor_company_id=cid, rows=by_creditor.get(cid, []), as_of_date=as_of
        )
        for cid in ids
    }


def hold_effect_for_receivable(
    receivable: PortalReceivable, *, as_of_date: date | datetime | None = None
) -> str:
    """Effet exposé au client pour une créance donnée."""
    as_of = current_business_date(as_of_date)
    if receivable_contributes_to_hold(receivable, as_of_date=as_of):
        return HOLD_EFFECT_CARRIER_BLOCKED
    return HOLD_EFFECT_NONE


def is_receivable_overdue_for_display(
    receivable: PortalReceivable, *, as_of_date: date | datetime | None = None
) -> bool:
    """Échue au sens calendaire avec solde, hors annulée/contestée/payée."""
    return receivable_contributes_to_hold(
        receivable, as_of_date=current_business_date(as_of_date)
    )
