"""Routes Admin Ops / Platform — status/runtime + gouvernance V1 (tenant, policy, runbooks)."""
# ruff: noqa: I001

from __future__ import annotations

import logging
from typing import Any

from flask import current_app, request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from ext import db, limiter, role_required
from feature_flags import get_feature_flags_status
from models import UserRole
from models.company import Company
from security.audit_log import AuditLogger
from security.ip_whitelist import ip_whitelist_required
from services.platform_authz import (
    PERM_GOVERNANCE_TENANT_SUSPEND,
    PERM_OPERATE_RUNBOOKS,
    user_effective_platform_permissions,
)
from services.platform_audit_events import (
    list_audit_events,
    replay_timeline_by_correlation_id,
)
from services.platform_change_requests import (
    complete_change_request,
    create_change_request,
    get_change_request,
    list_change_requests,
)
from services.platform_exceptions import (
    PlatformRollbackNotAllowed,
    PlatformRunbookConflict,
)
from services.platform_policy import evaluate_policy
from services.platform_reconciliation import drift_summary_for_tenant
from services.platform_region_topology import current_region_topology
from services.platform_runbooks import (
    execute_runbook,
    get_execution,
    get_runbook,
    list_runbooks,
    preview_execution,
    rollback_execution,
)
from services.platform_runtime import build_platform_runtime_payload
from services.platform_search import search_investigation
from services.platform_services_catalog import list_platform_services
from services.platform_status_aggregator import build_platform_status_payload
from services.platform_tenant_governance import (
    apply_suspend,
    get_company_or_404,
    suspend_preview,
    tenant_governance_payload,
)
from shared.infrastructure.adapters.auth_adapter import get_current_user_via_use_case

logger = logging.getLogger(__name__)
MIN_JUSTIFICATION_LENGTH = 3

platform_ops_ns = Namespace(
    "platform",
    description="Admin Ops / Platform (status, runtime, policy, tenants, runbooks)",
)


def _correlation_id_from_request(data: dict[str, Any] | None) -> str | None:
    if request.headers.get("X-Correlation-Id"):
        return request.headers.get("X-Correlation-Id")
    if data and data.get("correlation_id"):
        return str(data["correlation_id"])
    return None


@platform_ops_ns.route("/status")
class PlatformStatusResource(Resource):
    """GET /api/v1/platform/status — agrégat prod/demo + liens observabilité."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")  # ~30s polling max théorique
    def get(self):
        try:
            payload = build_platform_status_payload(current_app.config)
        except Exception as e:
            logger.exception("[platform/status] agrégation: %s", e)
            return {"error": "aggregation_failed", "message": str(e)}, 500

        try:
            current_user = get_current_user_via_use_case()
            AuditLogger.log_action(
                action_type="platform_status_read",
                action_category="platform_ops",
                user_id=current_user.id if current_user else None,
                user_type=current_user.role.value
                if current_user and current_user.role
                else "admin",
                result_status="success",
                action_details={},
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )
        except Exception as audit_error:
            logger.warning("[platform/status] audit: %s", audit_error)

        return payload, 200


@platform_ops_ns.route("/runtime")
class PlatformRuntimeResource(Resource):
    """GET /api/v1/platform/runtime — exploitation enrichie (Phase 2, hors hot path status)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def get(self):
        try:
            payload = build_platform_runtime_payload()
        except Exception as e:
            logger.exception("[platform/runtime] agrégation: %s", e)
            return {"error": "aggregation_failed", "message": str(e)}, 500
        return payload, 200


@platform_ops_ns.route("/topology/regions")
class PlatformRegionTopologyResource(Resource):
    """GET /api/v1/platform/topology/regions — état mono/multi-région."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self):
        return current_region_topology(), 200


@platform_ops_ns.route("/me")
class PlatformMeResource(Resource):
    """GET /api/v1/platform/me — identité JWT + rôle plateforme (V1)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self):
        user = get_current_user_via_use_case()
        if not user:
            return {"error": "unauthorized", "message": "Utilisateur introuvable."}, 401
        perms = sorted(user_effective_platform_permissions(user.id))
        return {
            "user_id": user.id,
            "public_id": user.public_id,
            "role": user.role.value if user.role else None,
            "email": user.email,
            "platform": {
                "scopes_hint": ["global", "tenant"],
                "bundles_effective": ["observe_core", "operate_tenant_controls"],
                "permissions_effective": perms,
                "note": "Bundles pilotent l’agrégat ; permissions_effective reflète les grants DB.",
            },
        }, 200


@platform_ops_ns.route("/search")
class PlatformSearchResource(Resource):
    """POST /api/v1/platform/search — InvestigationContext minimal (3 types d'IDs)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        q = (data.get("query") or data.get("q") or "").strip()
        if not q:
            return {"error": "validation", "message": "query requis."}, 400
        ctx = search_investigation(q)
        return {
            "investigation_context": ctx,
            "correlation_id": _correlation_id_from_request(data),
        }, 200


@platform_ops_ns.route("/services")
class PlatformServicesResource(Resource):
    """GET /api/v1/platform/services — catalogue services (§6bis)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self):
        return {"items": list_platform_services()}, 200


@platform_ops_ns.route("/feature-flags")
class PlatformFeatureFlagsResource(Resource):
    """GET /api/v1/platform/feature-flags — état flags infra (ML, etc.)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def get(self):
        return get_feature_flags_status(), 200


@platform_ops_ns.route("/change-requests")
class PlatformChangeRequestsListResource(Resource):
    """GET /api/v1/platform/change-requests — registre ChangeRequest (persisté SQL)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def get(self):
        limit = min(int(request.args.get("limit", 50)), 200)
        return {"items": list_change_requests(limit=limit)}, 200


@platform_ops_ns.route("/change-requests/<string:change_request_id>")
class PlatformChangeRequestDetailResource(Resource):
    """GET /api/v1/platform/change-requests/{id}"""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self, change_request_id: str):
        rec = get_change_request(change_request_id)
        if not rec:
            return {"error": "not_found", "message": "ChangeRequest inconnu."}, 404
        return rec, 200


@platform_ops_ns.route("/audit-events")
class PlatformAuditEventsResource(Resource):
    """GET /api/v1/platform/audit-events — filtre audit_logs (replay grossier)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self):
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(max(int(request.args.get("per_page", 50)), 1), 200)
        action_type = request.args.get("action_type")
        action_category = request.args.get("action_category")
        company_id = request.args.get("company_id")
        correlation_id = request.args.get("correlation_id")
        cid = int(company_id) if company_id and str(company_id).isdigit() else None
        return list_audit_events(
            page=page,
            per_page=per_page,
            action_type=action_type,
            action_category=action_category,
            company_id=cid,
            correlation_id=correlation_id,
        ), 200


@platform_ops_ns.route("/audit-events/replay")
class PlatformAuditReplayResource(Resource):
    """GET /api/v1/platform/audit-events/replay?correlation_id= — timeline ordonnée (replay V1)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def get(self):
        cid = (request.args.get("correlation_id") or "").strip()
        if not cid:
            return {"error": "validation", "message": "correlation_id requis."}, 400
        return replay_timeline_by_correlation_id(cid), 200


@platform_ops_ns.route("/reconciliation")
class PlatformReconciliationResource(Resource):
    """GET /api/v1/platform/reconciliation?tenant_id= — drift summary (§18)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self):
        tid = request.args.get("tenant_id")
        if not tid or not str(tid).isdigit():
            return {"error": "validation", "message": "tenant_id requis."}, 400
        out = drift_summary_for_tenant(int(tid))
        if not out:
            return {"error": "not_found", "message": "Tenant inconnu."}, 404
        return out, 200


@platform_ops_ns.route("/policies/evaluate")
class PlatformPoliciesEvaluateResource(Resource):
    """POST /api/v1/platform/policies/evaluate — enveloppe DEC-005 / §25."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def post(self):
        data = request.get_json(silent=True) or {}
        action_type = data.get("action_type") or ""
        scope_type = data.get("scope_type") or ""
        scope_id = data.get("scope_id")
        if scope_id is not None:
            scope_id = str(scope_id)

        perm = None
        if action_type == "governance.tenant.suspend":
            perm = PERM_GOVERNANCE_TENANT_SUSPEND
        elif action_type in (
            "governance.runbook.execute",
            "governance.runbook.rollback",
        ):
            perm = PERM_OPERATE_RUNBOOKS

        current_user = get_current_user_via_use_case()
        result = evaluate_policy(
            action_type=action_type,
            scope_type=scope_type,
            scope_id=scope_id,
            explicit_deny=bool(data.get("explicit_deny")),
            emergency_override=bool(data.get("emergency_override")),
            incident_id=data.get("incident_id"),
            requested_permission=perm,
            is_admin=True,
            user_id=current_user.id if current_user else None,
        )
        return {
            "policy_evaluation_result": result,
            "correlation_id": _correlation_id_from_request(data),
        }, 200


@platform_ops_ns.route("/tenants")
class PlatformTenantsListResource(Resource):
    """GET /api/v1/platform/tenants — liste paginée (tenant = Company)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def get(self):
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(max(int(request.args.get("per_page", 20)), 1), 100)
        q = Company.query.order_by(Company.id)
        pagination = q.paginate(page=page, per_page=per_page, error_out=False)
        items = [
            {
                "tenant_id": c.id,
                "tenant_type": "company",
                "name": c.name,
                "platform_suspended": bool(c.platform_suspended),
            }
            for c in pagination.items
        ]
        return {
            "items": items,
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": pagination.total,
            "pages": pagination.pages,
        }, 200


@platform_ops_ns.route("/tenants/<int:tenant_id>")
class PlatformTenantDetailResource(Resource):
    """GET /api/v1/platform/tenants/{id} — desired / observed / effective (§22)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self, tenant_id: int):
        company = get_company_or_404(tenant_id)
        if not company:
            return {"error": "not_found", "message": "Tenant inconnu."}, 404
        return tenant_governance_payload(company), 200


@platform_ops_ns.route("/tenants/<int:tenant_id>/suspend/preview")
class PlatformTenantSuspendPreviewResource(Resource):
    """POST /api/v1/platform/tenants/{id}/suspend/preview — blast radius."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def post(self, tenant_id: int):
        data = request.get_json(silent=True) or {}
        prev = suspend_preview(tenant_id)
        if not prev:
            return {"error": "not_found", "message": "Tenant inconnu."}, 404
        prev["correlation_id"] = _correlation_id_from_request(data)
        return prev, 200


@platform_ops_ns.route("/tenants/<int:tenant_id>/suspend")
class PlatformTenantSuspendResource(Resource):
    """POST /api/v1/platform/tenants/{id}/suspend — intention suspend + audit."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("20 per hour")
    def post(self, tenant_id: int):
        data = request.get_json(silent=True) or {}
        justification = (data.get("justification") or "").strip()
        if len(justification) < MIN_JUSTIFICATION_LENGTH:
            return {
                "decision": "blocked",
                "reason_code": "justification_required",
                "human_reason": "Une justification (>= 3 caracteres) est obligatoire.",
            }, 400

        company = get_company_or_404(tenant_id)
        if not company:
            return {"error": "not_found", "message": "Tenant inconnu."}, 404

        current_user = get_current_user_via_use_case()
        pol = evaluate_policy(
            action_type="governance.tenant.suspend",
            scope_type="tenant",
            scope_id=str(tenant_id),
            explicit_deny=bool(data.get("explicit_deny")),
            emergency_override=bool(data.get("emergency_override")),
            incident_id=data.get("incident_id"),
            requested_permission=PERM_GOVERNANCE_TENANT_SUSPEND,
            is_admin=True,
            user_id=current_user.id if current_user else None,
        )
        if pol.get("decision") != "allow":
            return {
                "decision": "denied",
                "policy_evaluation_result": pol,
            }, 403

        correlation_id = _correlation_id_from_request(data)

        if company.platform_suspended:
            payload = tenant_governance_payload(company)
            return {
                "decision": "already_applied",
                "reason_code": "tenant_already_suspended",
                "tenant": payload,
                "correlation_id": correlation_id,
                "policy_evaluation_result": pol,
            }, 200

        before = {"platform_suspended": bool(company.platform_suspended)}

        cr = create_change_request(
            change_type="tenant_suspension",
            tenant_id=tenant_id,
            justification=justification,
            correlation_id=correlation_id,
            incident_id=data.get("incident_id"),
            metadata={"policy_evaluation_result": pol},
            requested_by_user_id=current_user.id if current_user else None,
        )

        payload, outcome = apply_suspend(company)
        db.session.commit()

        complete_change_request(
            cr["id"],
            status="completed",
            result={
                "outcome": outcome,
                "reconciliation_status": payload.get("reconciliation_status"),
            },
        )
        db.session.commit()

        result_status = (
            "partially_applied" if outcome == "partially_applied" else "applied"
        )
        audit_rs = "partial" if outcome == "partially_applied" else "success"

        try:
            AuditLogger.log_action(
                action_type="platform_tenant_suspend",
                action_category="platform_ops",
                user_id=current_user.id if current_user else None,
                user_type=current_user.role.value
                if current_user and current_user.role
                else "admin",
                result_status=audit_rs,
                result_message=result_status,
                action_details={
                    "decision": result_status,
                    "tenant_id": tenant_id,
                    "justification": justification,
                    "policy_evaluation_result": pol,
                    "decision_basis": "permission",
                    "before_state": before,
                    "after_state": {"platform_suspended": True},
                    "tenant_governance": payload,
                },
                company_id=tenant_id,
                resource_type="tenant",
                resource_id=str(tenant_id),
                correlation_id=correlation_id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
                metadata={
                    "incident_id": data.get("incident_id"),
                    "emergency_override": bool(data.get("emergency_override")),
                },
            )
        except Exception as e:
            logger.warning("[platform/tenant/suspend] audit: %s", e)

        return {
            "decision": result_status,
            "outcome": outcome,
            "change_request_id": cr["id"],
            "policy_evaluation_result": pol,
            "tenant": payload,
            "correlation_id": correlation_id,
        }, 200


@platform_ops_ns.route("/runbooks")
class PlatformRunbooksListResource(Resource):
    """GET /api/v1/platform/runbooks — catalogue minimal."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def get(self):
        return {"items": list_runbooks()}, 200


@platform_ops_ns.route("/runbooks/<string:runbook_id>")
class PlatformRunbookDetailResource(Resource):
    """GET /api/v1/platform/runbooks/{id} — fiche runbook."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def get(self, runbook_id: str):
        rb = get_runbook(runbook_id)
        if not rb:
            return {"error": "not_found", "message": "Runbook inconnu."}, 404
        return rb, 200


@platform_ops_ns.route("/runbooks/<string:runbook_id>/preview")
class PlatformRunbookPreviewResource(Resource):
    """POST /api/v1/platform/runbooks/{id}/preview — prévisualisation d'exécution."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("60 per hour")
    def post(self, runbook_id: str):
        data = request.get_json(silent=True) or {}
        tid = data.get("tenant_id")
        if tid is None:
            return {"error": "validation", "message": "tenant_id requis."}, 400
        prev = preview_execution(
            runbook_id,
            tenant_id=int(tid),
            correlation_id=_correlation_id_from_request(data),
        )
        if not prev:
            return {"error": "not_found", "message": "Runbook inconnu."}, 404
        return prev, 200


@platform_ops_ns.route("/runbooks/<string:runbook_id>/executions")
class PlatformRunbookExecutionsResource(Resource):
    """POST /api/v1/platform/runbooks/{id}/executions — exécution + verify (V1 synchrone)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("30 per hour")
    def post(self, runbook_id: str):
        data = request.get_json(silent=True) or {}
        tid = data.get("tenant_id")
        if tid is None:
            return {"error": "validation", "message": "tenant_id requis."}, 400
        correlation_id = _correlation_id_from_request(data)
        current_user = get_current_user_via_use_case()
        pol = evaluate_policy(
            action_type="governance.runbook.execute",
            scope_type="tenant",
            scope_id=str(int(tid)),
            requested_permission=PERM_OPERATE_RUNBOOKS,
            is_admin=True,
            user_id=current_user.id if current_user else None,
        )
        if pol.get("decision") != "allow":
            return {
                "decision": "denied",
                "policy_evaluation_result": pol,
            }, 403

        try:
            out = execute_runbook(
                runbook_id,
                tenant_id=int(tid),
                correlation_id=correlation_id,
                triggered_by_user_id=current_user.id if current_user else None,
            )
        except PlatformRunbookConflict:
            return {
                "error_code": "runbook_conflict",
                "human_reason": (
                    "Une exécution runbook est déjà en cours pour ce tenant et ce runbook."
                ),
            }, 409
        if not out:
            return {"error": "not_found", "message": "Runbook ou tenant inconnu."}, 404

        try:
            AuditLogger.log_action(
                action_type="platform_runbook_execution",
                action_category="platform_ops",
                user_id=current_user.id if current_user else None,
                user_type=current_user.role.value
                if current_user and current_user.role
                else "admin",
                result_status="success"
                if out.get("result_status") == "success"
                else "partial"
                if out.get("result_status") == "partially_applied"
                else "failure",
                action_details={
                    "runbook_id": runbook_id,
                    "execution_id": out.get("id"),
                    "tenant_id": int(tid),
                    "verification_status": out.get("verification_status"),
                    "result_status": out.get("result_status"),
                },
                company_id=int(tid),
                resource_type="runbook_execution",
                resource_id=str(out.get("id")),
                correlation_id=correlation_id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
                metadata={"runbook_id": runbook_id},
            )
        except Exception as e:
            logger.warning("[platform/runbooks/executions] audit: %s", e)

        return out, 200


@platform_ops_ns.route("/runbooks/executions/<string:execution_id>")
class PlatformRunbookExecutionDetailResource(Resource):
    """GET /api/v1/platform/runbooks/executions/{execution_id} — détail exécution (lecture DB)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")
    def get(self, execution_id: str):
        ex = get_execution(execution_id)
        if not ex:
            return {"error": "not_found", "message": "Exécution inconnue."}, 404
        return ex, 200


@platform_ops_ns.route("/runbooks/executions/<string:execution_id>/rollback")
class PlatformRunbookRollbackResource(Resource):
    """POST /api/v1/platform/runbooks/executions/{id}/rollback — marque rolled_back (V1)."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("30 per hour")
    def post(self, execution_id: str):
        ex0 = get_execution(execution_id)
        if not ex0:
            return {"error": "not_found", "message": "Exécution inconnue."}, 404
        tid = ex0.get("tenant_id")
        if tid is None:
            return {
                "error": "validation",
                "message": "tenant_id manquant sur l'exécution.",
            }, 400
        current_user = get_current_user_via_use_case()
        pol = evaluate_policy(
            action_type="governance.runbook.rollback",
            scope_type="tenant",
            scope_id=str(int(tid)),
            requested_permission=PERM_OPERATE_RUNBOOKS,
            is_admin=True,
            user_id=current_user.id if current_user else None,
        )
        if pol.get("decision") != "allow":
            return {
                "decision": "denied",
                "policy_evaluation_result": pol,
            }, 403
        try:
            out = rollback_execution(execution_id)
        except PlatformRollbackNotAllowed as e:
            return {"error_code": e.code, "human_reason": e.message}, 409
        if not out:
            return {"error": "not_found", "message": "Exécution inconnue."}, 404
        rb_cid = ex0.get("correlation_id")
        try:
            AuditLogger.log_action(
                action_type="platform_runbook_rollback",
                action_category="platform_ops",
                user_id=current_user.id if current_user else None,
                user_type=current_user.role.value
                if current_user and current_user.role
                else "admin",
                result_status="success",
                action_details={
                    "execution_id": execution_id,
                    "runbook_id": out.get("runbook_id"),
                    "tenant_id": out.get("tenant_id"),
                },
                company_id=out.get("tenant_id"),
                resource_type="runbook_execution",
                resource_id=str(execution_id),
                correlation_id=rb_cid,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )
        except Exception as e:
            logger.warning("[platform/runbooks/rollback] audit: %s", e)
        return out, 200


@platform_ops_ns.route("/actions")
class PlatformActionsResource(Resource):
    """POST /api/v1/platform/actions — déprécié : préférer tenants/suspend et runbooks."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("30 per hour")
    def post(self):
        return {
            "error": "not_implemented",
            "message": (
                "Utiliser POST /api/v1/platform/tenants/{id}/suspend (gouvernance) ou "
                "POST /api/v1/platform/runbooks/{id}/executions (runbooks)."
            ),
            "deprecated": True,
        }, 501
