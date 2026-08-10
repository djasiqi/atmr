"""Gouvernance opérationnelle Company (approbation / dispatch) — admin."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from ext import db
from models.booking import Booking
from models.company import Company
from models.driver import Driver
from models.enums import BookingStatus
from security.audit_log import AuditLog
from services.control_plane.classification import (
    CompanyProjectionKind,
    classify_company_for_control_plane,
)
from services.control_plane.projector import get_projector

logger = logging.getLogger(__name__)


class AdminCompanyOpsError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error = error
        self.details = details or {}

    def to_response(self) -> tuple[dict[str, Any], int]:
        body: dict[str, Any] = {"message": self.message}
        if self.error:
            body["error"] = self.error
        if self.details:
            body["details"] = self.details
        return body, self.status_code


def _now() -> datetime:
    return datetime.now(UTC)


def _lock_transport_tenant(company_id: int) -> Company:
    company = db.session.execute(
        select(Company).where(Company.id == company_id).with_for_update()
    ).scalar_one_or_none()
    if company is None:
        raise AdminCompanyOpsError(
            "Entreprise introuvable.", status_code=404, error="company_not_found"
        )
    kind = classify_company_for_control_plane(company).kind
    if kind != CompanyProjectionKind.TRANSPORT_TENANT:
        raise AdminCompanyOpsError(
            "L'entreprise n'est pas un tenant de transport admissible.",
            status_code=422,
            error="company_not_transport_tenant",
            details={"kind": kind.value},
        )
    return company


def preview_dispatch_disable(company_id: int) -> dict[str, Any]:
    company = db.session.get(Company, company_id)
    if company is None:
        raise AdminCompanyOpsError(
            "Entreprise introuvable.", status_code=404, error="company_not_found"
        )
    active_drivers = (
        db.session.scalar(
            select(func.count())
            .select_from(Driver)
            .where(Driver.company_id == company_id, Driver.is_active.is_(True))
        )
        or 0
    )
    total_drivers = (
        db.session.scalar(
            select(func.count())
            .select_from(Driver)
            .where(Driver.company_id == company_id)
        )
        or 0
    )
    active_statuses = [
        BookingStatus.PENDING,
        BookingStatus.ACCEPTED,
        BookingStatus.ASSIGNED,
        BookingStatus.EN_ROUTE,
        BookingStatus.IN_PROGRESS,
    ]
    status_values = [s.value for s in active_statuses]
    active_bookings = (
        db.session.scalar(
            select(func.count())
            .select_from(Booking)
            .where(
                Booking.company_id == company_id,
                Booking.status.in_(status_values),
            )
        )
        or 0
    )
    return {
        "company_id": company_id,
        "company_name": company.name,
        "active_drivers_count": int(active_drivers),
        "total_drivers_count": int(total_drivers),
        "active_bookings_count": int(active_bookings),
        "dispatch_enabled": bool(company.dispatch_enabled),
        "warnings": [
            "Les nouvelles affectations / opérations dispatch seront indisponibles.",
            "Les chauffeurs et l'approbation plateforme ne sont pas modifiés.",
        ],
    }


@dataclass
class FlagUpdateResult:
    company_id: int
    field: str
    value: bool
    status: str = "updated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "company_id": self.company_id,
            self.field: self.value,
        }


def set_company_approval(
    *,
    company_id: int,
    is_approved: bool,
    reason: str,
    actor_admin_id: int,
    expected_is_approved: bool | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> FlagUpdateResult:
    if not reason or len(reason.strip()) < 5:
        raise AdminCompanyOpsError(
            "Une raison d'au moins 5 caractères est requise.",
            status_code=400,
            error="reason_required",
        )
    company = _lock_transport_tenant(company_id)
    current = bool(company.is_approved)
    if expected_is_approved is not None and bool(expected_is_approved) != current:
        raise AdminCompanyOpsError(
            "L'état d'approbation a changé concurrentement.",
            status_code=409,
            error="approval_status_changed",
            details={
                "expected_is_approved": expected_is_approved,
                "current_is_approved": current,
            },
        )
    if current == bool(is_approved):
        return FlagUpdateResult(
            company_id=int(company.id),
            field="is_approved",
            value=current,
            status="unchanged",
        )

    with db.session.begin_nested():
        company.is_approved = bool(is_approved)
        db.session.flush()
        get_projector().ensure_company_organization(company)
        audit = AuditLog()
        audit.user_id = actor_admin_id
        audit.user_type = "ADMIN"
        audit.action_type = (
            "admin_company_approved" if is_approved else "admin_company_unapproved"
        )
        audit.action_category = "security"
        audit.action_details = json.dumps(
            {
                "company_id": company.id,
                "old_is_approved": current,
                "new_is_approved": bool(is_approved),
                "reason": reason.strip(),
            },
            ensure_ascii=False,
        )
        audit.result_status = "success"
        audit.ip_address = ip_address
        audit.user_agent = user_agent
        audit.company_id = company.id
        audit.resource_type = "company"
        audit.resource_id = str(company.id)
        audit.created_at = _now()
        db.session.add(audit)

    db.session.commit()
    return FlagUpdateResult(
        company_id=int(company.id),
        field="is_approved",
        value=bool(company.is_approved),
    )


def set_company_dispatch(
    *,
    company_id: int,
    dispatch_enabled: bool,
    reason: str,
    actor_admin_id: int,
    expected_dispatch_enabled: bool | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> FlagUpdateResult:
    if not reason or len(reason.strip()) < 5:
        raise AdminCompanyOpsError(
            "Une raison d'au moins 5 caractères est requise.",
            status_code=400,
            error="reason_required",
        )
    company = _lock_transport_tenant(company_id)
    current = bool(company.dispatch_enabled)
    if (
        expected_dispatch_enabled is not None
        and bool(expected_dispatch_enabled) != current
    ):
        raise AdminCompanyOpsError(
            "L'état dispatch a changé concurrentement.",
            status_code=409,
            error="dispatch_status_changed",
            details={
                "expected_dispatch_enabled": expected_dispatch_enabled,
                "current_dispatch_enabled": current,
            },
        )
    if current == bool(dispatch_enabled):
        return FlagUpdateResult(
            company_id=int(company.id),
            field="dispatch_enabled",
            value=current,
            status="unchanged",
        )

    preview = (
        preview_dispatch_disable(int(company.id)) if not dispatch_enabled else None
    )

    with db.session.begin_nested():
        company.dispatch_enabled = bool(dispatch_enabled)
        db.session.flush()
        get_projector().ensure_company_organization(company)
        audit = AuditLog()
        audit.user_id = actor_admin_id
        audit.user_type = "ADMIN"
        audit.action_type = (
            "admin_company_dispatch_enabled"
            if dispatch_enabled
            else "admin_company_dispatch_disabled"
        )
        audit.action_category = "security"
        audit.action_details = json.dumps(
            {
                "company_id": company.id,
                "old_dispatch_enabled": current,
                "new_dispatch_enabled": bool(dispatch_enabled),
                "reason": reason.strip(),
                "preview": preview,
            },
            ensure_ascii=False,
            default=str,
        )
        audit.result_status = "success"
        audit.ip_address = ip_address
        audit.user_agent = user_agent
        audit.company_id = company.id
        audit.resource_type = "company"
        audit.resource_id = str(company.id)
        audit.created_at = _now()
        db.session.add(audit)

    db.session.commit()
    return FlagUpdateResult(
        company_id=int(company.id),
        field="dispatch_enabled",
        value=bool(company.dispatch_enabled),
    )
