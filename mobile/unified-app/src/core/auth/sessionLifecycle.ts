/**
 * Taxonomie et policy du cycle de vie de session mobile (PR2).
 * Orthogonal au status UI legacy ("idle" | "bootstrapping" | …).
 */
import type { MobileSessionStatus } from "./mobileSessionStatus";

export type SessionLifecycleEvent =
  | "explicit_logout_claimed"
  | "explicit_logout_completed"
  | "terminal_revocation"
  | "auth_exhausted_socket"
  | "login_claimed"
  | "login_persisted"
  | "login_stale_orphaned"
  | "bootstrap_cold_start"
  | "bootstrap_login_success"
  | "bootstrap_manual_retry"
  | "bootstrap_auth_recovery"
  | "refresh_stale_ignored"
  | "interrupted_logout_finished";

export type BootstrapTrigger =
  | "cold_start_auto"
  | "login_success"
  | "manual_retry"
  | "auth_recovery";

export type SessionLifecyclePolicyContext = {
  hasOfflineSnapshot: boolean;
  currentStatus: MobileSessionStatus;
  autoBootstrapAllowed: boolean;
};

export type SessionLifecyclePolicyDecision = {
  /** Transition de statut mobile souhaitée (null = ne pas changer). */
  statusAction: MobileSessionStatus | null;
  /** Autoriser / refuser un auto-bootstrap ultérieur. */
  autoBootstrapAction: "allow" | "deny" | "unchanged";
  /** Quarantaine GPS requise pour cet événement. */
  quarantineRequired: boolean;
  /** Preuve terminale (invalidate*) requise. */
  terminalEvidenceRequired: boolean;
};

/**
 * Policy pure — aucun I/O. Les appelants appliquent les effets (mutex, réseau, UI).
 */
export function resolveSessionLifecyclePolicy(
  event: SessionLifecycleEvent,
  context: SessionLifecyclePolicyContext
): SessionLifecyclePolicyDecision {
  void context.hasOfflineSnapshot;
  switch (event) {
    case "explicit_logout_claimed":
      return {
        statusAction: "logging_out",
        autoBootstrapAction: "deny",
        quarantineRequired: true,
        terminalEvidenceRequired: false,
      };
    case "explicit_logout_completed":
    case "interrupted_logout_finished":
      return {
        statusAction: "anonymous",
        autoBootstrapAction: "deny",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    case "terminal_revocation":
      return {
        statusAction: "revoked",
        autoBootstrapAction: "deny",
        quarantineRequired: false,
        terminalEvidenceRequired: true,
      };
    case "auth_exhausted_socket":
      return {
        statusAction: null,
        autoBootstrapAction: "unchanged",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    case "login_claimed":
      return {
        statusAction: null,
        autoBootstrapAction: "allow",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    case "login_persisted":
      return {
        statusAction: "authenticated_online",
        autoBootstrapAction: "allow",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    case "login_stale_orphaned":
      return {
        statusAction: null,
        autoBootstrapAction: "unchanged",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    case "bootstrap_cold_start":
      return {
        statusAction: context.autoBootstrapAllowed ? "auth_recovering" : null,
        autoBootstrapAction: "unchanged",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    case "bootstrap_login_success":
    case "bootstrap_manual_retry":
    case "bootstrap_auth_recovery":
      return {
        statusAction: "auth_recovering",
        autoBootstrapAction: "unchanged",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    case "refresh_stale_ignored":
      return {
        statusAction: null,
        autoBootstrapAction: "unchanged",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
      return {
        statusAction: null,
        autoBootstrapAction: "unchanged",
        quarantineRequired: false,
        terminalEvidenceRequired: false,
      };
    }
  }
}

export function shouldAcceptBootstrapTrigger(
  trigger: BootstrapTrigger,
  autoBootstrapAllowed: boolean
): boolean {
  if (trigger === "cold_start_auto") {
    return autoBootstrapAllowed;
  }
  return true;
}

export function newLifecycleOperationId(): string {
  return `op-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}
