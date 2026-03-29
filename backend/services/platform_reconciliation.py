"""Drift / réconciliation — agrégat pour endpoint dédié (§18 MVP)."""

from __future__ import annotations

from typing import Any

from ext import db
from models.company import Company

from services.platform_tenant_governance import tenant_governance_payload


def drift_summary_for_tenant(tenant_id: int) -> dict[str, Any] | None:
    company = db.session.get(Company, tenant_id)
    if not company:
        return None
    gov = tenant_governance_payload(company)
    return {
        "tenant_id": tenant_id,
        "reconciliation_status": gov.get("reconciliation_status"),
        "desired_state": gov.get("desired_state"),
        "observed_state": gov.get("observed_state"),
        "effective_state": gov.get("effective_state"),
        "effective_badges": gov.get("effective_badges"),
        "remediation_suggested": (
            ["tenant_post_suspend_verify"]
            if gov.get("reconciliation_status") == "drift"
            and gov.get("desired_state", {}).get("operational") == "suspended"
            else []
        ),
    }
