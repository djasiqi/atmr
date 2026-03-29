"""Constantes gouvernance plateforme V1 — enums et codes stables (alignés plan audit)."""

from __future__ import annotations

# --- RunbookExecution.status (cycle machine) ---
RUNBOOK_EXEC_PREVIEWED = "previewed"
RUNBOOK_EXEC_RUNNING = "running"
RUNBOOK_EXEC_VERIFYING = "verifying"
RUNBOOK_EXEC_COMPLETED = "completed"
RUNBOOK_EXEC_FAILED = "failed"
RUNBOOK_EXEC_ROLLED_BACK = "rolled_back"

RUNBOOK_EXECUTION_STATUSES = frozenset(
    {
        RUNBOOK_EXEC_PREVIEWED,
        RUNBOOK_EXEC_RUNNING,
        RUNBOOK_EXEC_VERIFYING,
        RUNBOOK_EXEC_COMPLETED,
        RUNBOOK_EXEC_FAILED,
        RUNBOOK_EXEC_ROLLED_BACK,
    }
)

# --- Résultat métier (API / result_json) ---
RUNBOOK_RESULT_SUCCESS = "success"
RUNBOOK_RESULT_FAILED = "failed"
RUNBOOK_RESULT_PARTIALLY_APPLIED = "partially_applied"
RUNBOOK_RESULT_ROLLED_BACK = "rolled_back"

# --- ChangeRequest.status ---
CHANGE_REQUEST_EXECUTING = "executing"
CHANGE_REQUEST_COMPLETED = "completed"
CHANGE_REQUEST_FAILED = "failed"

CHANGE_REQUEST_STATUSES = frozenset(
    {
        CHANGE_REQUEST_EXECUTING,
        CHANGE_REQUEST_COMPLETED,
        CHANGE_REQUEST_FAILED,
    }
)

# --- Gates / erreurs ---
ERROR_TENANT_PLATFORM_SUSPENDED = "tenant_platform_suspended"
ERROR_TENANT_ALREADY_SUSPENDED = "tenant_already_suspended"
ERROR_ALREADY_IN_PROGRESS = "already_in_progress"
ERROR_RUNBOOK_CONFLICT = "runbook_conflict"
ERROR_ROLLBACK_NOT_ALLOWED = "rollback_not_allowed"
