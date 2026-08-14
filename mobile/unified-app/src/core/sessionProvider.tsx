import { useQueryClient } from "@tanstack/react-query";
import {
  applyContextCachePolicyOnSwitch,
  clearAllContextCache,
  restoreContextCache,
} from "./cache/contextCache";
import { prefetchContextTarget } from "./cache/prefetchContextTarget";
import { emitContextSwitchKpi } from "./observability/perfKpi";
import { markBootMilestone } from "./observability/bootMilestones";
import { recordContextSwitchPhase } from "./observability/perfInstrumentation";
import {
  AuthContext,
  BootstrapResponse,
  hasPermission,
  resolveDefaultContext,
} from "./contracts/auth";
import {
  CLIENT_SURFACE_CONTRACT_VERSIONS,
  logContractMismatchEvent,
} from "./contracts/clientSurfaceVersions";
import {
  fetchBootstrap,
  getLastRefreshErrorCode,
  hasAuthToken,
  login,
  setActiveContextIdForApi,
  switchContext,
} from "./api/client";
import {
  applyTerminalRevocationIfCurrent,
  attemptRestRecovery,
  finishInterruptedExplicitLogout,
  flushPendingRevocationTombstone,
  performExplicitLogout,
  persistOfflineSnapshot,
  restoreOfflineSessionSnapshot,
} from "./auth/authRecoveryCoordinator";
import {
  getSessionGenerationId,
  isCurrentSessionGeneration,
  readSessionEnvelope,
  type SessionGenerationId,
} from "./auth/authCredentialStore";
import {
  clearContextSwitchOperationIfCurrent,
  isCurrentContextSwitchOperation,
} from "./auth/contextSwitchOperation";
import {
  emitTrackingAuthTerminalEvent,
  setTrackingAuthAvailability,
} from "./auth/sessionAuthDecision";
import {
  clearTrackingAuthSession,
  publishTrackingAuthSessionAvailable,
} from "./auth/trackingAuthPresence";
import {
  newLifecycleOperationId,
  shouldAcceptBootstrapTrigger,
  type BootstrapTrigger,
} from "./auth/sessionLifecycle";
import {
  resolveOfflineCapabilities,
  type MobileSessionStatus,
} from "./auth/mobileSessionStatus";
import { realtimeManager } from "./realtime/realtimeManager";
import { contextRealtimeRouter } from "./realtime/contextRealtimeRouter";
import { isFeatureEnabled, setRuntimeFeatureFlagOverrides } from "./featureFlags/registry";
import { QUERY_STALE_TIME_MS } from "./queryStaleTimes";
import { getDriverMissions } from "../features/driver/api/driverHttp";
import { driverQueryKeys } from "../features/driver/queryKeys";
import { emitDriverTelemetry } from "./observability/driverTelemetry";
import { appendSessionJournalEvent, clearSessionJournal, hydrateSessionJournal } from "./observability/sessionJournal";
import { purgeDriverProfileCache } from "../features/driver/services/driverProfileCache";
import {
  companyDriverSwitchBlockedMessage,
  isCompanyDriverCrossContextSwitch,
  isCompanyDriverSwitchAllowedForRequest,
} from "./contextSwitchPolicy";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

/**
 * Chargement lazy via require (compatible Jest) — `import()` dynamique échoue avec
 * ERR_VM_DYNAMIC_IMPORT_CALLBACK_MISSING_FLAG sans --experimental-vm-modules.
 */
function loadTrackingContextLease(): typeof import("../features/driver/services/trackingContextLease") {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require("../features/driver/services/trackingContextLease");
}

function loadDriverTrackingQueue(): typeof import("../features/driver/services/driverTrackingQueue") {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require("../features/driver/services/driverTrackingQueue");
}

function loadTrackingRuntimeRegistry(): typeof import("../features/driver/services/trackingRuntimeRegistry") {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require("../features/driver/services/trackingRuntimeRegistry");
}

function loadDriverTrackingBridge(): typeof import("../features/driver/services/driverTrackingBridge") {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require("../features/driver/services/driverTrackingBridge");
}

type SessionStatus = "idle" | "bootstrapping" | "ready" | "error";

type SessionContextValue = {
  status: SessionStatus;
  mobileSessionStatus: MobileSessionStatus;
  offlineCapabilities: ReturnType<typeof resolveOfflineCapabilities>;
  bootstrap: BootstrapResponse | null;
  activeContext: AuthContext | null;
  error: string | null;
  autoBootstrapAllowed: boolean;
  login: (email: string, password: string) => Promise<void>;
  bootstrapSession: (opts?: { trigger?: BootstrapTrigger }) => Promise<void>;
  changeContext: (targetContextId: string) => Promise<void>;
  contextSwitchInFlight: boolean;
  logout: () => Promise<void>;
  hasPermission: (permission: string) => boolean;
  can: (permission: string) => boolean;
};

type BootstrapInFlight = {
  generation: SessionGenerationId;
  promise: Promise<void>;
};
type PropsWithChildren<P = object> = P & { children?: any };

function extractScopedId(contextId: string, prefix: string): string | null {
  if (!contextId.startsWith(`${prefix}:`)) return null;
  const value = contextId.slice(prefix.length + 1).trim();
  return value.length > 0 ? value : null;
}

function getCompanyIdFromContext(context: AuthContext): string | number | null {
  if (context.context_type !== "company") return null;
  if (context.organization_id !== undefined && context.organization_id !== null) {
    return context.organization_id;
  }
  return extractScopedId(context.context_id, "company");
}

function getDriverIdFromContext(context: AuthContext): string | null {
  if (context.context_type !== "driver") return null;
  return extractScopedId(context.context_id, "driver");
}

function assertContextRuntimeInvariants(context: AuthContext | null) {
  if (!context) return;
  if (context.context_type === "company" && getCompanyIdFromContext(context) == null) {
    throw new Error("Invalid context invariant: company context requires company_id");
  }
  if (context.context_type === "driver" && getDriverIdFromContext(context) == null) {
    throw new Error("Invalid context invariant: driver context requires driver_id");
  }
}

/** Socket chauffeur : actif uniquement en contexte driver (évite logout fantôme company). */
export function syncDriverRealtimeForContext(
  context: AuthContext | null,
  options?: { enableSocket?: boolean }
) {
  if (context?.context_type === "driver") {
    realtimeManager.onContextSwitch(context.context_id, {
      enableSocket: options?.enableSocket ?? isFeatureEnabled("realtime_socket_enabled"),
    });
    return;
  }
  realtimeManager.disconnect();
}

function toUiErrorMessage(error: unknown, fallback: string): string {
  const axiosLikeStatus =
    error && typeof error === "object"
      ? typeof (error as { response?: { status?: unknown } }).response?.status === "number"
        ? ((error as { response?: { status?: number } }).response?.status ?? null)
        : null
      : null;
  const statusFromShape =
    error && typeof error === "object" && typeof (error as { status?: unknown }).status === "number"
      ? ((error as { status?: number }).status ?? null)
      : null;
  const effectiveStatus = axiosLikeStatus ?? statusFromShape;
  const isLoginFlow = fallback === "Login failed";
  if (isLoginFlow && effectiveStatus === 401) {
    return "Les données de connexion sont incorrectes.";
  }
  if (error && typeof error === "object") {
    const code = typeof (error as { code?: unknown }).code === "string"
      ? (error as { code: string }).code
      : null;
    if (code === "DEVICE_ID_UNAVAILABLE" || code === "device_identity_required") {
      return "Impossible de sécuriser la session sur cet appareil. Fermez puis rouvrez l'application et réessayez.";
    }
    if (code === "STORAGE_UNAVAILABLE" || code === "storage_locked") {
      return "Stockage sécurisé temporairement indisponible. Fermez puis rouvrez l'application et réessayez.";
    }
    if (code === "AUTH_LOGIN_CONTRACT_INCOMPLETE" || code === "mobile_session_contract_incomplete") {
      return "Le serveur n'a pas retourné les éléments nécessaires à une session sécurisée.";
    }
  }
  if (effectiveStatus === 401 || effectiveStatus === 403) {
    return "Session expirée ou invalide. Reconnectez-vous pour continuer.";
  }
  if (effectiveStatus === 503 || effectiveStatus === 502 || effectiveStatus === 504) {
    return "Serveur indisponible temporairement. Vérifiez que l’API est démarrée, puis réessayez.";
  }
  if (error instanceof Error) {
    if (/status code 503/i.test(error.message)) {
      return "Serveur indisponible temporairement. Vérifiez que l’API est démarrée, puis réessayez.";
    }
    return error.message;
  }
  if (error && typeof error === "object") {
    const candidate = error as { message?: unknown; code?: unknown; status?: unknown };
    const message = typeof candidate.message === "string" ? candidate.message : null;
    const code = typeof candidate.code === "string" ? candidate.code : null;
    const status =
      typeof candidate.status === "number" ? candidate.status : null;
    if (message && code && status != null) return `${message} (${code}, ${status})`;
    if (message && code) return `${message} (${code})`;
    if (message) return message;
  }
  return fallback;
}

const SessionContext = ReactRuntime.createContext(undefined as
  | SessionContextValue
  | undefined);

export function SessionProvider({ children }: PropsWithChildren) {
  const queryClient = useQueryClient();
  const [status, setStatus] = ReactRuntime.useState("idle" as SessionStatus);
  const [mobileSessionStatus, setMobileSessionStatus] = ReactRuntime.useState(
    "initializing" as MobileSessionStatus
  );
  const [bootstrap, setBootstrap] = ReactRuntime.useState(
    null as BootstrapResponse | null
  );
  const [activeContext, setActiveContext] = ReactRuntime.useState(
    null as AuthContext | null
  );
  const [error, setError] = ReactRuntime.useState(null as string | null);
  const [contextSwitchInFlight, setContextSwitchInFlight] = ReactRuntime.useState(false);
  const [autoBootstrapAllowed, setAutoBootstrapAllowed] = ReactRuntime.useState(true);
  const autoBootstrapAllowedRef = ReactRuntime.useRef(true);
  const bootstrapInFlightRef = ReactRuntime.useRef(null as BootstrapInFlight | null);
  const activeContextRef = ReactRuntime.useRef(activeContext);
  ReactRuntime.useLayoutEffect(() => {
    activeContextRef.current = activeContext;
  }, [activeContext]);

  const setAutoBootstrapAllowedSync = ReactRuntime.useCallback((allowed: boolean) => {
    autoBootstrapAllowedRef.current = allowed;
    setAutoBootstrapAllowed(allowed);
  }, []);

  ReactRuntime.useEffect(() => {
    void hydrateSessionJournal();
  }, []);

  const runDriverQuarantine = ReactRuntime.useCallback(
    async (args: {
      identity: { userId: string; driverId: string; companyId: string };
      lifecycleOperationId: string;
    }) => {
      const { driverTrackingQueue } = loadDriverTrackingQueue();
      await driverTrackingQueue.quarantineOnLogout({
        userId: args.identity.userId,
        driverId: args.identity.driverId,
        companyId: args.identity.companyId,
        lifecycleOperationId: args.lifecycleOperationId,
      });
    },
    []
  );

  const resumeSessionIfPossible = ReactRuntime.useCallback(async () => {
    const resumeGeneration = getSessionGenerationId();
    void appendSessionJournalEvent("session.resume.start");
    setMobileSessionStatus("auth_recovering");

    // 1. Flush pending réseau (ne crée pas de preuve terminale)
    await flushPendingRevocationTombstone().catch(() => false);
    try {
       
      const { flushPendingSessionConfirmation } = require("./auth/pendingSessionConfirmation") as {
        flushPendingSessionConfirmation: () => Promise<boolean>;
      };
      // Best-effort : confirmation provisional post-login (nécessite access token si déjà en mémoire)
      void flushPendingSessionConfirmation().catch(() => undefined);
    } catch {
      /* ignore */
    }

    // 2. Restore offline snapshot
    const offline = await restoreOfflineSessionSnapshot();
    if (!isCurrentSessionGeneration(resumeGeneration)) return;

    if (offline.kind === "storage_locked") {
      setMobileSessionStatus("storage_locked");
      setError("Stockage sécurisé temporairement indisponible");
      return;
    }
    if (offline.kind === "interrupted_logout") {
      await finishInterruptedExplicitLogout(offline.pending, {
        runQuarantine: runDriverQuarantine,
      });
      if (!isCurrentSessionGeneration(resumeGeneration)) return;
      setBootstrap(null);
      setActiveContext(null);
      activeContextRef.current = null;
      setActiveContextIdForApi(null);
      setRuntimeFeatureFlagOverrides(null);
      contextRealtimeRouter.setActiveContext(null);
      setMobileSessionStatus("anonymous");
      setStatus("idle");
      setAutoBootstrapAllowedSync(false);
      return;
    }
    if (offline.kind === "revoked") {
      setMobileSessionStatus("revoked");
      setAutoBootstrapAllowedSync(false);
      return;
    }
    if (offline.kind === "restored") {
      markBootMilestone("SESSION_RESTORED");
      setMobileSessionStatus("authenticated_offline");
      if (offline.bootstrap) setBootstrap(offline.bootstrap);
      if (offline.activeContext) {
        setActiveContext(offline.activeContext);
        setActiveContextIdForApi(offline.activeContext.context_id ?? null);
      }
    }
    if (hasAuthToken()) {
      void appendSessionJournalEvent("session.resume.skipped_has_access_token");
      setMobileSessionStatus("authenticated_online");
      try {
         
        const { flushPendingSessionConfirmation } = require("./auth/pendingSessionConfirmation") as {
          flushPendingSessionConfirmation: () => Promise<boolean>;
        };
        void flushPendingSessionConfirmation().catch(() => undefined);
      } catch {
        /* ignore */
      }
      return;
    }
    // 3. Recovery REST (refresh puis session-resume) — capture génération, pas de bump
    const outcome = await attemptRestRecovery("cold_start");
    if (!isCurrentSessionGeneration(resumeGeneration)) return;

    if (outcome === "recovered") {
      setMobileSessionStatus("authenticated_online");
      void appendSessionJournalEvent("session.resume.success", { via: "coordinator" });
      try {
         
        const { flushPendingSessionConfirmation } = require("./auth/pendingSessionConfirmation") as {
          flushPendingSessionConfirmation: () => Promise<boolean>;
        };
        void flushPendingSessionConfirmation().catch(() => undefined);
      } catch {
        /* ignore */
      }
      try {
        const ctx = activeContextRef.current;
        if (ctx?.context_type === "driver") {
          const { driverTrackingQueue } = loadDriverTrackingQueue();
          await driverTrackingQueue.resumeAfterAuthRecovery({
            userId: ctx.context_id,
            driverId: getDriverIdFromContext(ctx) ?? ctx.context_id,
            companyId: getCompanyIdFromContext(ctx) ?? "unknown",
          });
        }
      } catch {
        /* best-effort */
      }
      return;
    }
    if (outcome === "keep_local" && offline.kind === "restored") {
      setMobileSessionStatus("authenticated_offline");
      return;
    }
    if (outcome === "terminal") {
      await applyTerminalRevocationIfCurrent(
        resumeGeneration,
        getLastRefreshErrorCode() ?? "session_revoked",
        (terminalGeneration) => {
          if (!isCurrentSessionGeneration(terminalGeneration)) return false;
          setBootstrap(null);
          setActiveContext(null);
          activeContextRef.current = null;
          setActiveContextIdForApi(null);
          setRuntimeFeatureFlagOverrides(null);
          contextRealtimeRouter.setActiveContext(null);
          setMobileSessionStatus("revoked");
          setStatus("idle");
          setAutoBootstrapAllowedSync(false);
          return true;
        }
      );
      return;
    }
    if (offline.kind !== "restored") {
      setMobileSessionStatus("anonymous");
    }
  }, [runDriverQuarantine, setAutoBootstrapAllowedSync]);

  /* Avant les effets des écrans : en-têtes API alignés sur le contexte (driver + company, multi-rôles). */
  ReactRuntime.useLayoutEffect(() => {
    setActiveContextIdForApi(activeContext?.context_id ?? null);
  }, [activeContext?.context_id]);

  const bootstrapSession = ReactRuntime.useCallback(async (opts?: { trigger?: BootstrapTrigger }) => {
    const trigger: BootstrapTrigger = opts?.trigger ?? "manual_retry";
    if (!shouldAcceptBootstrapTrigger(trigger, autoBootstrapAllowedRef.current)) {
      void appendSessionJournalEvent("session.bootstrap.rejected_auto", { trigger });
      return;
    }

    const generation = getSessionGenerationId();
    const existing = bootstrapInFlightRef.current;
    if (existing && existing.generation === generation) {
      return existing.promise;
    }

    const cycle = (async () => {
      if (!isCurrentSessionGeneration(generation)) return;
      setStatus("bootstrapping");
      setError(null);
      const bootstrapStartedAt = Date.now();
      void appendSessionJournalEvent(
        "session.bootstrap.start",
        { trigger },
        activeContextRef.current?.context_id ?? null
      );
      try {
        await resumeSessionIfPossible();
        if (!isCurrentSessionGeneration(generation)) return;
        // Après logout interrompu / anonymous sans auto-bootstrap : ne pas fetcher
        if (!autoBootstrapAllowedRef.current && trigger === "cold_start_auto") {
          return;
        }
        const data = await fetchBootstrap(activeContextRef.current?.context_id ?? null);
        if (!isCurrentSessionGeneration(generation)) return;
      if (
        data.status_dictionary_version &&
        data.status_dictionary_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.statusDictionaryVersion
      ) {
        logContractMismatchEvent(
          "status",
          CLIENT_SURFACE_CONTRACT_VERSIONS.statusDictionaryVersion,
          data.status_dictionary_version
        );
      }
      if (
        data.pricing_contract_version &&
        data.pricing_contract_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.pricingContractVersion
      ) {
        logContractMismatchEvent(
          "pricing",
          CLIENT_SURFACE_CONTRACT_VERSIONS.pricingContractVersion,
          data.pricing_contract_version
        );
      }
      if (
        data.canonical_address_contract_version &&
        data.canonical_address_contract_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.canonicalAddressContractVersion
      ) {
        logContractMismatchEvent(
          "canonical_address",
          CLIENT_SURFACE_CONTRACT_VERSIONS.canonicalAddressContractVersion,
          data.canonical_address_contract_version
        );
      }
      if (
        data.preview_contract_version &&
        data.preview_contract_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.previewContractVersion
      ) {
        logContractMismatchEvent(
          "preview",
          CLIENT_SURFACE_CONTRACT_VERSIONS.previewContractVersion,
          data.preview_contract_version
        );
      }
      if (
        data.mission_status_version &&
        data.mission_status_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.missionStatusVersion
      ) {
        logContractMismatchEvent(
          "mission_status",
          CLIENT_SURFACE_CONTRACT_VERSIONS.missionStatusVersion,
          data.mission_status_version
        );
      }
      if (
        data.mission_snapshot_version &&
        data.mission_snapshot_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.missionSnapshotVersion
      ) {
        logContractMismatchEvent(
          "mission_snapshot",
          CLIENT_SURFACE_CONTRACT_VERSIONS.missionSnapshotVersion,
          data.mission_snapshot_version
        );
      }
      if (
        data.driver_socket_contract_version &&
        data.driver_socket_contract_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.driverSocketContractVersion
      ) {
        logContractMismatchEvent(
          "driver_socket",
          CLIENT_SURFACE_CONTRACT_VERSIONS.driverSocketContractVersion,
          data.driver_socket_contract_version
        );
      }
      if (
        data.driver_tracking_contract_version &&
        data.driver_tracking_contract_version !==
          CLIENT_SURFACE_CONTRACT_VERSIONS.driverTrackingContractVersion
      ) {
        logContractMismatchEvent(
          "driver_tracking",
          CLIENT_SURFACE_CONTRACT_VERSIONS.driverTrackingContractVersion,
          data.driver_tracking_contract_version
        );
      }
      const resolved = resolveDefaultContext(data.available_contexts, data.active_context_id);
      assertContextRuntimeInvariants(resolved);
      if (!isCurrentSessionGeneration(generation)) return;
      setBootstrap(data);
      setRuntimeFeatureFlagOverrides(data.feature_flags ?? {});
      setActiveContext(resolved);
      // Évite une fenêtre de course: certains appels peuvent partir avant useLayoutEffect.
      // On aligne l'en-tête API immédiatement dès que le contexte bootstrap est connu.
      setActiveContextIdForApi(resolved?.context_id ?? null);
      contextRealtimeRouter.setActiveContext(resolved?.context_type ?? null);
      setStatus("ready");
      markBootMilestone("SESSION_READY");
      setMobileSessionStatus(
        data.is_authenticated ? "authenticated_online" : "anonymous"
      );
      if (data.is_authenticated) {
        setAutoBootstrapAllowedSync(true);
      }
      void persistOfflineSnapshot(data, resolved).catch(() => undefined);
      void appendSessionJournalEvent(
        data.is_authenticated ? "session.bootstrap.success" : "session.bootstrap.unauthenticated",
        {
          context_type: resolved?.context_type ?? null,
          duration_ms: Date.now() - bootstrapStartedAt,
          trigger,
        },
        resolved?.context_id ?? null
      );
      // Réconciliation lease crash-safe (switching → inactive ou restore)
      void Promise.resolve()
        .then(async () => {
          const {
            reconcileTrackingContextLeaseFromBootstrap,
            setTrackingContextLeaseDriverActive,
          } = loadTrackingContextLease();
          const lease = await reconcileTrackingContextLeaseFromBootstrap({
            activeContextId: resolved?.context_id ?? null,
            activeContextType: resolved?.context_type ?? null,
            isAuthenticated: Boolean(data.is_authenticated),
          });
          if (
            resolved?.context_type === "driver" &&
            data.is_authenticated &&
            lease.state !== "driver_active"
          ) {
            const driverIdRaw = getDriverIdFromContext(resolved);
            const driverId = driverIdRaw != null ? Number(driverIdRaw) : NaN;
            if (Number.isFinite(driverId)) {
              const { startOrJoinTrackingRuntime } = loadTrackingRuntimeRegistry();
              const runtime = await startOrJoinTrackingRuntime({
                driverId,
                companyId: getCompanyIdFromContext(resolved),
                missionId: null,
                missionStatus: null,
              });
              await setTrackingContextLeaseDriverActive({
                contextId: resolved.context_id,
                driverId,
                sessionGenerationId: runtime.identity.sessionGenerationId,
                trackingGenerationId: runtime.identity.trackingGenerationId,
                trackingIdentityId: runtime.identity.trackingIdentityId,
                missionId: runtime.missionContext.missionId,
                missionContextVersion: runtime.missionContext.missionContextVersion,
              });
              // P0-B : presence persistée + cache SESSION_AVAILABLE (login / restore)
              await publishTrackingAuthSessionAvailable({
                driverId,
                trackingIdentityId: runtime.identity.trackingIdentityId,
                sessionGenerationId: runtime.identity.sessionGenerationId,
              });
            }
          }
        })
        .catch(() => undefined);
      // Socket en dernier
      syncDriverRealtimeForContext(resolved, {
        enableSocket: isFeatureEnabled("realtime_socket_enabled"),
      });
      // Reprise GPS après bootstrap auth
      if (resolved?.context_type === "driver" && data.is_authenticated) {
        void Promise.resolve()
          .then(() => {
            const { driverTrackingQueue } = loadDriverTrackingQueue();
            return driverTrackingQueue.resumeAfterAuthRecovery({
              userId: resolved.context_id,
              driverId: getDriverIdFromContext(resolved) ?? resolved.context_id,
              companyId: String(getCompanyIdFromContext(resolved) ?? "unknown"),
            });
          })
          .catch(() => undefined);
      }
      } catch (e) {
        if (!isCurrentSessionGeneration(generation)) return;
        const message = toUiErrorMessage(e, "Bootstrap failed");
        setError(message);
        setStatus("error");
        void appendSessionJournalEvent("session.bootstrap.error", { message }, activeContextRef.current?.context_id ?? null);
      }
    })();

    const entry: BootstrapInFlight = { generation, promise: cycle };
    bootstrapInFlightRef.current = entry;
    void cycle.finally(() => {
      if (bootstrapInFlightRef.current === entry) {
        bootstrapInFlightRef.current = null;
      }
    });
    return cycle;
  }, [resumeSessionIfPossible, setAutoBootstrapAllowedSync]);

  const loginAndBootstrap = ReactRuntime.useCallback(async (email: string, password: string) => {
    setError(null);
    setAutoBootstrapAllowedSync(true);
    try {
      await login(email, password);
      await bootstrapSession({ trigger: "login_success" });
    } catch (e) {
      const message = toUiErrorMessage(e, "Login failed");
      setError(message);
      throw e;
    }
  }, [bootstrapSession, setAutoBootstrapAllowedSync]);

  const changeContext = ReactRuntime.useCallback(async (targetContextId: string) => {
    const previousContextId = activeContext?.context_id ?? null;
    const previousContext = activeContext;
    const previousBootstrap = bootstrap;
    setError(null);
    const toCtx = bootstrap?.available_contexts.find(
      (c: AuthContext) => c.context_id === targetContextId
    );
    if (
      toCtx &&
      !isCompanyDriverSwitchAllowedForRequest(
        activeContext,
        toCtx,
        bootstrap?.user?.role
      )
    ) {
      const message =
        companyDriverSwitchBlockedMessage(
          activeContext,
          toCtx,
          bootstrap?.user?.role
        ) ?? "Changement de contexte refuse.";
      setError(message);
      throw new Error(message);
    }

    const rollbackOptimistic = () => {
      setActiveContext(previousContext);
      setActiveContextIdForApi(previousContext?.context_id ?? null);
      contextRealtimeRouter.setActiveContext(previousContext?.context_type ?? null);
      setBootstrap(previousBootstrap);
      setRuntimeFeatureFlagOverrides(previousBootstrap?.feature_flags ?? {});
    };

    const crossWorkspaceSwitch = isCompanyDriverCrossContextSwitch(
      previousContext,
      toCtx ?? null
    );

    const applyOptimisticContext = (ctx: AuthContext) => {
      assertContextRuntimeInvariants(ctx);
      setActiveContext(ctx);
      setActiveContextIdForApi(ctx.context_id);
      contextRealtimeRouter.setActiveContext(ctx.context_type);
      if (previousBootstrap) {
        setBootstrap({
          ...previousBootstrap,
          active_context_id: ctx.context_id,
        });
      }
    };

    const switchStartedAt = Date.now();
    // Bascule entreprise ↔ chauffeur : pas d'optimistic UI.
    // Sinon DriverContextGuard / CompanyLayout redirigent tout de suite, puis un
    // échec API rollback → rebond automatique vers l'écran d'origine.
    const usedOptimistic = Boolean(toCtx) && !crossWorkspaceSwitch;
    if (usedOptimistic && toCtx) {
      applyOptimisticContext(toCtx);
      prefetchContextTarget(queryClient, toCtx);
    }

    setContextSwitchInFlight(true);
    const leavingDriver = previousContext?.context_type === "driver";
    const enteringDriver = toCtx?.context_type === "driver";
    let leaseArmedSwitching = false;
    try {
      if (leavingDriver || enteringDriver) {
        const {
          readTrackingContextLease,
          setTrackingContextLeaseSwitching,
        } = loadTrackingContextLease();
        const previousLease = await readTrackingContextLease();
        await setTrackingContextLeaseSwitching({
          fromDriver: leavingDriver,
          previousDriverActive:
            previousLease?.state === "driver_active" ? previousLease : null,
        });
        leaseArmedSwitching = true;
        if (leavingDriver) {
          const { driverTrackingQueue } = loadDriverTrackingQueue();
          await driverTrackingQueue.activateContextInactiveGate("context_switching");
        }
      }

      const response = await switchContext(targetContextId, {
        sourceContextId: previousContextId,
      });
      const opId = (response as { contextSwitchOperationId?: string })
        .contextSwitchOperationId;
      if (opId && !isCurrentContextSwitchOperation(opId)) {
        throw new Error("CONTEXT_SWITCH_STALE");
      }
      if (
        typeof (response as { contextSwitchSessionGenerationId?: number })
          .contextSwitchSessionGenerationId === "number" &&
        !isCurrentSessionGeneration(
          (response as { contextSwitchSessionGenerationId: number })
            .contextSwitchSessionGenerationId
        )
      ) {
        throw new Error("CONTEXT_SWITCH_STALE");
      }
      const nextAvailableContexts: AuthContext[] =
        response.available_contexts ?? bootstrap?.available_contexts ?? [];
      const nextContext =
        nextAvailableContexts.find(
          (ctx: AuthContext) => ctx.context_id === response.active_context_id
        ) ?? toCtx ?? null;
      assertContextRuntimeInvariants(nextContext);

      // Header API immédiatement (coupe /driver/me/* dès COMPANY)
      if (nextContext) {
        setActiveContextIdForApi(nextContext.context_id);
      }

      const {
        setTrackingContextLeaseInactive,
        setTrackingContextLeaseDriverActive,
        reconcileTrackingContextLeaseFromBootstrap,
      } = loadTrackingContextLease();

      if (nextContext?.context_type !== "driver") {
        await setTrackingContextLeaseInactive();
      }

      setBootstrap((prev: BootstrapResponse | null) =>
        prev
          ? {
              ...prev,
              active_context_id: response.active_context_id,
              available_contexts: nextAvailableContexts,
              feature_flags: response.feature_flags ?? prev.feature_flags,
            }
          : prev
      );
      setRuntimeFeatureFlagOverrides(response.feature_flags ?? {});
      if (nextContext) {
        setActiveContext(nextContext);
        contextRealtimeRouter.setActiveContext(nextContext.context_type);
        if (crossWorkspaceSwitch) {
          prefetchContextTarget(queryClient, nextContext);
        }
      }

      // Persist SessionEnvelope immédiatement (autorité auth headless)
      if (previousBootstrap && nextContext) {
        await persistOfflineSnapshot(
          {
            ...previousBootstrap,
            active_context_id: response.active_context_id,
            available_contexts: nextAvailableContexts,
            feature_flags: response.feature_flags ?? previousBootstrap.feature_flags,
          },
          nextContext
        ).catch(() => undefined);
      }

      // Hard stop local sans flush après sortie chauffeur
      if (leavingDriver && nextContext?.context_type !== "driver") {
        const { hardStopDriverContextRuntime } = loadDriverTrackingBridge();
        await hardStopDriverContextRuntime("context_left_driver");
      }

      // Entrée chauffeur : lease driver_active avant toute /driver/me/*
      if (nextContext?.context_type === "driver") {
        const driverIdRaw = getDriverIdFromContext(nextContext);
        const driverId = driverIdRaw != null ? Number(driverIdRaw) : NaN;
        if (Number.isFinite(driverId)) {
          const {
            startOrJoinTrackingRuntime,
            resolveTrackingIdentityId,
          } = loadTrackingRuntimeRegistry();
          const runtime = await startOrJoinTrackingRuntime({
            driverId,
            companyId: getCompanyIdFromContext(nextContext),
            missionId: null,
            missionStatus: null,
          });
          await setTrackingContextLeaseDriverActive({
            contextId: nextContext.context_id,
            driverId,
            sessionGenerationId: runtime.identity.sessionGenerationId,
            trackingGenerationId: runtime.identity.trackingGenerationId,
            trackingIdentityId:
              runtime.identity.trackingIdentityId ||
              resolveTrackingIdentityId(driverId),
            missionId: runtime.missionContext.missionId,
            missionContextVersion: runtime.missionContext.missionContextVersion,
          });
          // P0-B : hydrater presence au bascule vers chauffeur
          await publishTrackingAuthSessionAvailable({
            driverId,
            trackingIdentityId:
              runtime.identity.trackingIdentityId ||
              resolveTrackingIdentityId(driverId),
            sessionGenerationId: runtime.identity.sessionGenerationId,
          });
          const { driverTrackingQueue } = loadDriverTrackingQueue();
          await driverTrackingQueue.clearContextInactiveGate("context_entered_driver");
          await driverTrackingQueue.resumeAfterAuthRecovery({
            userId: nextContext.context_id,
            driverId,
            companyId: getCompanyIdFromContext(nextContext) ?? "unknown",
          });
        } else {
          await reconcileTrackingContextLeaseFromBootstrap({
            activeContextId: nextContext.context_id,
            activeContextType: nextContext.context_type,
            isAuthenticated: true,
          });
        }
      }

      if (opId) {
        clearContextSwitchOperationIfCurrent(opId);
      }
      void appendSessionJournalEvent("session.context.switch", {
        previous_context_id: previousContextId,
        next_context_id: nextContext?.context_id ?? null,
      }, nextContext?.context_id ?? null);

      const runPostSwitchSideEffects = () => {
        if (previousContextId) {
          applyContextCachePolicyOnSwitch(queryClient, previousContextId);
        }
        const cacheHit =
          nextContext?.context_id != null
            ? restoreContextCache(queryClient, nextContext.context_id)
            : false;
        const socketPhaseStarted = Date.now();
        syncDriverRealtimeForContext(nextContext, {
          enableSocket: isFeatureEnabled("realtime_socket_enabled"),
        });
        const socketMs = Date.now() - socketPhaseStarted;
        recordContextSwitchPhase("socket", socketMs, { cache_hit: cacheHit });

        const prefetchPhaseStarted = Date.now();
        if (
          nextContext?.context_id &&
          !cacheHit &&
          nextContext.context_type === "driver"
        ) {
          void queryClient
            .prefetchQuery({
              queryKey: driverQueryKeys.missions(nextContext.context_id),
              queryFn: getDriverMissions,
              staleTime: QUERY_STALE_TIME_MS.default,
            })
            .catch(() => undefined);
        }
        const prefetchMs = Date.now() - prefetchPhaseStarted;
        recordContextSwitchPhase("prefetch", prefetchMs, { cache_hit: cacheHit });

        const totalMs = Date.now() - switchStartedAt;
        recordContextSwitchPhase("total", totalMs, {
          cache_hit: cacheHit,
          previous_context_id: previousContextId,
          next_context_id: nextContext?.context_id ?? null,
        });
        emitContextSwitchKpi({
          source: "sessionProvider.changeContext",
          duration_ms: totalMs,
          cache_hit: cacheHit,
          previous_context_id: previousContextId,
          next_context_id: nextContext?.context_id ?? null,
          context_switch_socket_ms: socketMs,
          context_switch_prefetch_ms: prefetchMs,
        });
        const renderPhaseStarted = Date.now();
        requestAnimationFrame(() => {
          recordContextSwitchPhase("render", Date.now() - renderPhaseStarted, {
            cache_hit: cacheHit,
          });
        });
      };

      if (crossWorkspaceSwitch) {
        queueMicrotask(runPostSwitchSideEffects);
      } else {
        runPostSwitchSideEffects();
      }
    } catch (e) {
      if (leaseArmedSwitching) {
        const {
          restoreTrackingContextLeaseDriverActiveFromSwitching,
          setTrackingContextLeaseInactive,
        } = loadTrackingContextLease();
        if (leavingDriver) {
          const restored = await restoreTrackingContextLeaseDriverActiveFromSwitching();
          if (restored) {
            const { driverTrackingQueue } = loadDriverTrackingQueue();
            await driverTrackingQueue.clearContextInactiveGate("switch_failed_restore");
          } else {
            await setTrackingContextLeaseInactive();
          }
        } else {
          await setTrackingContextLeaseInactive();
        }
      }
      if (usedOptimistic) {
        rollbackOptimistic();
      }
      const message = toUiErrorMessage(e, "Impossible de basculer d'espace.");
      setError(message);
      throw new Error(message);
    } finally {
      setContextSwitchInFlight(false);
    }
  }, [activeContext, queryClient, bootstrap]);

  const logout = ReactRuntime.useCallback(async () => {
    const sourceGeneration = getSessionGenerationId();
    const envelope = await readSessionEnvelope();
    const sourceSessionId =
      envelope.status === "found"
        ? envelope.value.session_id
        : `anon-${sourceGeneration}`;
    const lifecycleOperationId = newLifecycleOperationId();

    const ctx = activeContextRef.current;
    const quarantineRequired = ctx?.context_type === "driver";
    const trackingIdentity =
      quarantineRequired && ctx
        ? {
            user_id: String(ctx.context_id),
            driver_id: String(getDriverIdFromContext(ctx) ?? ctx.context_id),
            company_id: String(getCompanyIdFromContext(ctx) ?? "unknown"),
          }
        : null;

    await performExplicitLogout({
      sourceGeneration,
      sourceSessionId,
      lifecycleOperationId,
      trackingIdentity,
      quarantineRequired,
      onLogoutClaimed: () => {
        setAutoBootstrapAllowedSync(false);
        setMobileSessionStatus("logging_out");
        // P0-B : effacer presence persistée + snapshot mémoire immédiatement
        void clearTrackingAuthSession({ reason: "logout" });
        setTrackingAuthAvailability({ kind: "TRACKING_IDENTITY_UNAVAILABLE" });
        try {
          void loadTrackingContextLease().setTrackingContextLeaseInactive();
        } catch {
          /* best-effort */
        }
        emitTrackingAuthTerminalEvent({
          kind: "EXPLICIT_LOGOUT",
          sourceSessionGenerationId: sourceGeneration,
          operationId: lifecycleOperationId,
          trackingIdentityId: trackingIdentity
            ? `${trackingIdentity.user_id}:${trackingIdentity.driver_id}:${trackingIdentity.company_id}`
            : null,
        });
      },
      runQuarantine: runDriverQuarantine,
      clearQuarantineIfOperationMatches: async (opId) => {
        const { driverTrackingQueue } = loadDriverTrackingQueue();
        await driverTrackingQueue.clearQuarantineIfOperationMatches(opId);
      },
      commitSessionStateIfCurrent: (logoutGeneration) => {
        if (!isCurrentSessionGeneration(logoutGeneration)) return false;
        setBootstrap(null);
        setActiveContext(null);
        activeContextRef.current = null;
        setActiveContextIdForApi(null);
        setRuntimeFeatureFlagOverrides(null);
        contextRealtimeRouter.setActiveContext(null);
        setStatus("idle");
        setMobileSessionStatus("anonymous");
        setError(null);
        realtimeManager.disconnect();
        return true;
      },
    }).catch(() => undefined);

    // Nettoyages lourds hors identité / statut (best-effort, non authoritative)
    void purgeDriverProfileCache().catch(() => undefined);
    clearAllContextCache(queryClient);
    void appendSessionJournalEvent("session.logout", {
      lifecycle_operation_id: lifecycleOperationId,
    });
    void clearSessionJournal();
  }, [queryClient, runDriverQuarantine, setAutoBootstrapAllowedSync]);

  ReactRuntime.useEffect(() => {
    return realtimeManager.onAuthExhausted((reason, code) => {
      if (activeContextRef.current?.context_type !== "driver") {
        return;
      }
      emitDriverTelemetry("realtime.auth.exhausted", {
        source: "sessionProvider.onAuthExhausted",
        reason,
        error_code: code ?? null,
        terminal: reason === "terminal",
        // Ne jamais logout / clearSession ici : panne socket ≠ révocation session.
        action: "degraded_polling",
      });
      // Le realtimeManager bascule déjà en mode dégradé/polling.
      // Un refresh REST éventuel est géré par le coordinateur auth (vague 401),
      // sans purge SecureStore ni quarantaine GPS.
    });
  }, []);

  const offlineCapabilities = resolveOfflineCapabilities(mobileSessionStatus);

  const value = ReactRuntime.useMemo(
    () => ({
      status,
      mobileSessionStatus,
      offlineCapabilities,
      bootstrap,
      activeContext,
      error,
      autoBootstrapAllowed,
      login: loginAndBootstrap,
      bootstrapSession,
      changeContext,
      contextSwitchInFlight,
      logout,
      hasPermission: (permission: string) => hasPermission(activeContext, permission),
      can: (permission: string) => hasPermission(activeContext, permission),
    }),
    [
      status,
      mobileSessionStatus,
      offlineCapabilities,
      bootstrap,
      activeContext,
      error,
      autoBootstrapAllowed,
      loginAndBootstrap,
      bootstrapSession,
      changeContext,
      contextSwitchInFlight,
      logout,
    ]
  ) as SessionContextValue;

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const ctx = ReactRuntime.useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used within SessionProvider");
  return ctx;
}
