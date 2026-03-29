"""Runbooks plateforme V1 — catalogue + exécutions persistées."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from ext import db
from models.company import Company
from models.platform_runbook_execution import PlatformRunbookExecution

from services.platform_exceptions import PlatformRollbackNotAllowed, PlatformRunbookConflict
from services.platform_governance_constants import (
    RUNBOOK_EXEC_COMPLETED,
    RUNBOOK_EXEC_FAILED,
    RUNBOOK_EXEC_ROLLED_BACK,
    RUNBOOK_EXEC_RUNNING,
    RUNBOOK_EXEC_VERIFYING,
    RUNBOOK_RESULT_FAILED,
    RUNBOOK_RESULT_PARTIALLY_APPLIED,
    RUNBOOK_RESULT_ROLLED_BACK,
    RUNBOOK_RESULT_SUCCESS,
)
from services.platform_tenant_governance import tenant_governance_payload

RUNBOOK_TENANT_POST_SUSPEND_VERIFY = "tenant_post_suspend_verify"

_RUNBOOKS: dict[str, dict[str, Any]] = {
    RUNBOOK_TENANT_POST_SUSPEND_VERIFY: {
        "id": RUNBOOK_TENANT_POST_SUSPEND_VERIFY,
        "version": "1.0.0",
        "title": "Vérification post-suspension tenant",
        "description": "Contrôle que l'intention suspendue est cohérente avec les agrégats observés.",
        "service_tags": ["governance", "tenant"],
    },
}


def list_runbooks() -> list[dict[str, Any]]:
    return list(_RUNBOOKS.values())


def get_runbook(runbook_id: str) -> dict[str, Any] | None:
    return _RUNBOOKS.get(runbook_id)


def preview_execution(
    runbook_id: str,
    *,
    tenant_id: int,
    correlation_id: str | None,
) -> dict[str, Any] | None:
    if runbook_id not in _RUNBOOKS:
        return None
    rb = _RUNBOOKS[runbook_id]
    steps = [
        {
            "id": "verify_reconciliation",
            "title": "Vérifier réconciliation tenant",
            "kind": "verify",
        }
    ]
    return {
        "runbook_id": rb["id"],
        "version": rb["version"],
        "tenant_id": tenant_id,
        "correlation_id": correlation_id,
        "steps": steps,
        "status": "previewed",
    }


def _execution_to_dict(row: PlatformRunbookExecution, rb: dict[str, Any]) -> dict[str, Any]:
    """Expose un statut d'exécution API canonique (running|verifying|completed|failed|rolled_back).
    Le résultat métier reste dans result_status / verification_status (ex. success = métier, pas statut machine)."""
    result_json = row.result_json or {}
    base_lifecycle = ["previewed", "running", "verifying", "completed"]
    out: dict[str, Any] = {
        "id": row.id,
        "runbook_id": row.runbook_id,
        "version": rb["version"],
        "tenant_id": row.tenant_id,
        "correlation_id": row.correlation_id,
        "lifecycle": list(base_lifecycle),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.finished_at.isoformat() if row.finished_at else None,
        "verification_status": result_json.get("verification_status"),
        "result_status": result_json.get("result_status"),
        "human_reason": result_json.get("human_reason"),
        "tenant_snapshot": result_json.get("tenant_snapshot"),
    }
    st = row.status
    if st == RUNBOOK_EXEC_ROLLED_BACK:
        out["status"] = "rolled_back"
        out["lifecycle"] = base_lifecycle + ["rolled_back"]
        out["rolled_back_at"] = result_json.get("rolled_back_at")
    elif st == RUNBOOK_EXEC_FAILED:
        out["status"] = "failed"
    elif st == RUNBOOK_EXEC_COMPLETED:
        out["status"] = "completed"
    elif st == RUNBOOK_EXEC_RUNNING:
        out["status"] = "running"
    elif st == RUNBOOK_EXEC_VERIFYING:
        out["status"] = "verifying"
    else:
        out["status"] = st
    return out


def execute_runbook(
    runbook_id: str,
    *,
    tenant_id: int,
    correlation_id: str | None,
    triggered_by_user_id: int | None = None,
) -> dict[str, Any] | None:
    if runbook_id not in _RUNBOOKS:
        return None
    if runbook_id != RUNBOOK_TENANT_POST_SUSPEND_VERIFY:
        return None
    rb = _RUNBOOKS[runbook_id]
    company = db.session.get(Company, tenant_id)
    if not company:
        return None

    exec_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    row = PlatformRunbookExecution(
        id=exec_id,
        runbook_id=runbook_id,
        status=RUNBOOK_EXEC_RUNNING,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        started_at=now,
        triggered_by_user_id=triggered_by_user_id,
        preview_snapshot_json=None,
        metadata_json={},
    )
    db.session.add(row)
    db.session.flush()

    active = db.session.scalar(
        select(func.count())
        .select_from(PlatformRunbookExecution)
        .where(
            PlatformRunbookExecution.tenant_id == tenant_id,
            PlatformRunbookExecution.runbook_id == runbook_id,
            PlatformRunbookExecution.status.in_(
                (RUNBOOK_EXEC_RUNNING, RUNBOOK_EXEC_VERIFYING)
            ),
        )
    ) or 0
    if active > 1:
        db.session.rollback()
        raise PlatformRunbookConflict()

    row.status = RUNBOOK_EXEC_VERIFYING
    db.session.flush()

    payload = tenant_governance_payload(company)
    desired = payload.get("desired_state") or {}
    recon = payload.get("reconciliation_status")
    suspended = desired.get("operational") == "suspended"

    if not suspended:
        verification_status = "failed"
        result_status = RUNBOOK_RESULT_FAILED
        human = "Tenant non marqué suspendu (desired)."
        row.status = RUNBOOK_EXEC_FAILED
    elif recon == "drift":
        verification_status = "partial"
        result_status = RUNBOOK_RESULT_PARTIALLY_APPLIED
        human = "Suspension demandée mais activité résiduelle (drift)."
        row.status = RUNBOOK_EXEC_COMPLETED
    else:
        verification_status = "passed"
        result_status = RUNBOOK_RESULT_SUCCESS
        human = "Suspension alignée avec les agrégats observés."
        row.status = RUNBOOK_EXEC_COMPLETED

    row.finished_at = datetime.now(UTC)
    row.result_json = {
        "verification_status": verification_status,
        "result_status": result_status,
        "human_reason": human,
        "tenant_snapshot": payload,
    }
    db.session.add(row)
    db.session.commit()
    fresh = db.session.get(PlatformRunbookExecution, exec_id)
    if not fresh:
        return None
    return _execution_to_dict(fresh, rb)


def rollback_execution(execution_id: str) -> dict[str, Any] | None:
    row = db.session.get(PlatformRunbookExecution, execution_id)
    if not row:
        return None
    if row.status not in (RUNBOOK_EXEC_COMPLETED, RUNBOOK_EXEC_FAILED):
        raise PlatformRollbackNotAllowed()
    now = datetime.now(UTC).isoformat()
    row.status = RUNBOOK_EXEC_ROLLED_BACK
    row.updated_at = datetime.now(UTC)
    rj = dict(row.result_json or {})
    rj["rolled_back_at"] = now
    rj["result_status"] = RUNBOOK_RESULT_ROLLED_BACK
    row.result_json = rj
    db.session.add(row)
    db.session.commit()
    rb = _RUNBOOKS.get(row.runbook_id) or {"version": "1.0.0"}
    fresh = db.session.get(PlatformRunbookExecution, execution_id)
    if not fresh:
        return None
    return _execution_to_dict(fresh, rb)


def get_execution(execution_id: str) -> dict[str, Any] | None:
    row = db.session.get(PlatformRunbookExecution, execution_id)
    if not row:
        return None
    rb = _RUNBOOKS.get(row.runbook_id) or {"version": "1.0.0"}
    return _execution_to_dict(row, rb)
