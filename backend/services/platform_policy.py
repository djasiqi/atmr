"""Évaluation policy plateforme — ordre DEC-001 (explicit_deny → … → allow)."""

from __future__ import annotations

from typing import Any

from services.platform_authz import admin_has_permission, user_has_platform_permission

REASON_EMERGENCY_NEEDS_INCIDENT = "emergency_override_requires_incident"
REASON_EXPLICIT_DENY = "explicit_deny"
REASON_SCOPE_MISMATCH = "blocked_by_scope"
REASON_MISSING_PERMISSION = "blocked_by_policy"
REASON_APPROVAL_REQUIRED = "require_approval"


def evaluate_policy(
    *,
    action_type: str,
    scope_type: str,
    scope_id: str | None,
    explicit_deny: bool = False,
    emergency_override: bool = False,
    incident_id: str | None = None,
    require_incident_for_override: bool = True,
    requested_permission: str | None = None,
    is_admin: bool = True,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Retourne une enveloppe stable (§25 / DEC-005) pour policies/evaluate et mutations.

    L'ordre est documenté dans action_details.evaluation_order.
    """
    evaluation_order = [
        "explicit_deny",
        "scope_check",
        "policy_check",
        "approval_requirement",
        "emergency_override",
        "allow",
    ]

    if explicit_deny:
        return _deny(
            reason_code=REASON_EXPLICIT_DENY,
            human_reason="Refus explicite (policy ou opérateur).",
            evaluation_order=evaluation_order,
            stopped_at="explicit_deny",
        )

    # Scope : V1 — seul scope tenant avec id numérique (suspend, runbooks)
    if action_type in (
        "governance.tenant.suspend",
        "governance.runbook.execute",
        "governance.runbook.rollback",
    ):
        if scope_type != "tenant" or not scope_id or not str(scope_id).isdigit():
            return _deny(
                reason_code=REASON_SCOPE_MISMATCH,
                human_reason="Action réservée au scope tenant avec identifiant valide.",
                evaluation_order=evaluation_order,
                stopped_at="scope_check",
            )

    if requested_permission:
        if user_id is not None:
            perm_ok = user_has_platform_permission(user_id, requested_permission)
        else:
            perm_ok = bool(is_admin and admin_has_permission(requested_permission))
        if not perm_ok:
            return _deny(
                reason_code=REASON_MISSING_PERMISSION,
                human_reason="Permission plateforme insuffisante pour cette action.",
                evaluation_order=evaluation_order,
                stopped_at="policy_check",
            )

    # Placeholder approbation prod : non activée en V1 sauf flag futur
    # if needs_approval:
    #     return _require_approval(...)

    if emergency_override:
        if require_incident_for_override and not (incident_id and str(incident_id).strip()):
            return _deny(
                reason_code=REASON_EMERGENCY_NEEDS_INCIDENT,
                human_reason="EmergencyOverride exige un incident_id (ou ticket) valide.",
                evaluation_order=evaluation_order,
                stopped_at="emergency_override",
            )

    return {
        "decision": "allow",
        "reason_code": None,
        "human_reason": None,
        "policy_ref": "platform_v1_default",
        "missing_scope": None,
        "required_approval": None,
        "retryable": False,
        "rollback_available": False,
        "suggested_next_actions": [],
        "decision_basis": "permission",
        "evaluation_order": evaluation_order,
        "stopped_at": "allow",
    }


def _deny(
    *,
    reason_code: str,
    human_reason: str,
    evaluation_order: list[str],
    stopped_at: str,
) -> dict[str, Any]:
    return {
        "decision": "deny",
        "reason_code": reason_code,
        "human_reason": human_reason,
        "policy_ref": "platform_v1_default",
        "missing_scope": None,
        "required_approval": None,
        "retryable": False,
        "rollback_available": False,
        "suggested_next_actions": [],
        "decision_basis": "policy_evaluation",
        "evaluation_order": evaluation_order,
        "stopped_at": stopped_at,
    }
