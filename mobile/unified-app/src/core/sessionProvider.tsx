import { useQueryClient } from "@tanstack/react-query";
import {
  applyContextCachePolicyOnSwitch,
  clearAllContextCache,
  restoreContextCache,
} from "./cache/contextCache";
import { prefetchContextTarget } from "./cache/prefetchContextTarget";
import { emitContextSwitchKpi } from "./observability/perfKpi";
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
  hasAuthToken,
  login,
  setActiveContextIdForApi,
  switchContext,
} from "./api/client";
import {
  attemptRestRecovery,
  flushPendingRevocationTombstone,
  performLogout,
  persistOfflineSnapshot,
  restoreOfflineSessionSnapshot,
} from "./auth/authRecoveryCoordinator";
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

type SessionStatus = "idle" | "bootstrapping" | "ready" | "error";

type SessionContextValue = {
  status: SessionStatus;
  mobileSessionStatus: MobileSessionStatus;
  offlineCapabilities: ReturnType<typeof resolveOfflineCapabilities>;
  bootstrap: BootstrapResponse | null;
  activeContext: AuthContext | null;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  bootstrapSession: () => Promise<void>;
  changeContext: (targetContextId: string) => Promise<void>;
  contextSwitchInFlight: boolean;
  logout: () => Promise<void>;
  hasPermission: (permission: string) => boolean;
  can: (permission: string) => boolean;
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
  const bootstrapInFlightRef = ReactRuntime.useRef(null as Promise<void> | null);
  const activeContextRef = ReactRuntime.useRef(activeContext);
  ReactRuntime.useLayoutEffect(() => {
    activeContextRef.current = activeContext;
  }, [activeContext]);

  ReactRuntime.useEffect(() => {
    void hydrateSessionJournal();
  }, []);

  const resumeSessionIfPossible = ReactRuntime.useCallback(async () => {
    void appendSessionJournalEvent("session.resume.start");
    setMobileSessionStatus("auth_recovering");
    // 1. Flush tombstone pending
    const flushed = await flushPendingRevocationTombstone();
    if (!flushed) {
      const pending = await restoreOfflineSessionSnapshot();
      if (pending.kind === "revoked") {
        setMobileSessionStatus("revoked");
        return;
      }
    }
    // 2. Restore offline snapshot
    const offline = await restoreOfflineSessionSnapshot();
    if (offline.kind === "storage_locked") {
      setMobileSessionStatus("storage_locked");
      setError("Stockage sécurisé temporairement indisponible");
      return;
    }
    if (offline.kind === "revoked") {
      setMobileSessionStatus("revoked");
      return;
    }
    if (offline.kind === "restored") {
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
      return;
    }
    // 3. Recovery REST (refresh puis session-resume)
    const outcome = await attemptRestRecovery("cold_start");
    if (outcome === "recovered") {
      setMobileSessionStatus("authenticated_online");
      void appendSessionJournalEvent("session.resume.success", { via: "coordinator" });
      // Reprise GPS après auth
      try {
        const ctx = activeContextRef.current;
        if (ctx?.context_type === "driver") {
          const { driverTrackingQueue } = await import(
            "../features/driver/services/driverTrackingQueue"
          );
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
      setMobileSessionStatus("revoked");
      return;
    }
    if (offline.kind !== "restored") {
      setMobileSessionStatus("anonymous");
    }
  }, []);

  /* Avant les effets des écrans : en-têtes API alignés sur le contexte (driver + company, multi-rôles). */
  ReactRuntime.useLayoutEffect(() => {
    setActiveContextIdForApi(activeContext?.context_id ?? null);
  }, [activeContext?.context_id]);

  const bootstrapSession = ReactRuntime.useCallback(async () => {
    if (bootstrapInFlightRef.current) {
      return bootstrapInFlightRef.current;
    }
    const cycle = (async () => {
      setStatus("bootstrapping");
      setError(null);
      const bootstrapStartedAt = Date.now();
      void appendSessionJournalEvent("session.bootstrap.start", undefined, activeContextRef.current?.context_id ?? null);
      try {
        await resumeSessionIfPossible();
        const data = await fetchBootstrap(activeContextRef.current?.context_id ?? null);
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
      setBootstrap(data);
      setRuntimeFeatureFlagOverrides(data.feature_flags ?? {});
      setActiveContext(resolved);
      // Évite une fenêtre de course: certains appels peuvent partir avant useLayoutEffect.
      // On aligne l'en-tête API immédiatement dès que le contexte bootstrap est connu.
      setActiveContextIdForApi(resolved?.context_id ?? null);
      contextRealtimeRouter.setActiveContext(resolved?.context_type ?? null);
      setStatus("ready");
      setMobileSessionStatus("authenticated_online");
      void persistOfflineSnapshot(data, resolved).catch(() => undefined);
      void appendSessionJournalEvent(
        data.is_authenticated ? "session.bootstrap.success" : "session.bootstrap.unauthenticated",
        {
          context_type: resolved?.context_type ?? null,
          duration_ms: Date.now() - bootstrapStartedAt,
        },
        resolved?.context_id ?? null
      );
      // Socket en dernier
      syncDriverRealtimeForContext(resolved, {
        enableSocket: isFeatureEnabled("realtime_socket_enabled"),
      });
      // Reprise GPS après bootstrap auth
      if (resolved?.context_type === "driver" && data.is_authenticated) {
        void import("../features/driver/services/driverTrackingQueue")
          .then(({ driverTrackingQueue }) =>
            driverTrackingQueue.resumeAfterAuthRecovery({
              userId: resolved.context_id,
              driverId: getDriverIdFromContext(resolved) ?? resolved.context_id,
              companyId: String(getCompanyIdFromContext(resolved) ?? "unknown"),
            })
          )
          .catch(() => undefined);
      }
      } catch (e) {
        const message = toUiErrorMessage(e, "Bootstrap failed");
        setError(message);
        setStatus("error");
        void appendSessionJournalEvent("session.bootstrap.error", { message }, activeContext?.context_id ?? null);
      }
    })();
    bootstrapInFlightRef.current = cycle.finally(() => {
      bootstrapInFlightRef.current = null;
    });
    return bootstrapInFlightRef.current;
  }, [resumeSessionIfPossible]);

  const loginAndBootstrap = ReactRuntime.useCallback(async (email: string, password: string) => {
    setError(null);
    try {
      await login(email, password);
      await bootstrapSession();
    } catch (e) {
      const message = toUiErrorMessage(e, "Login failed");
      setError(message);
      throw e;
    }
  }, [bootstrapSession]);

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
    try {
      const response = await switchContext(targetContextId);
      const nextAvailableContexts: AuthContext[] =
        response.available_contexts ?? bootstrap?.available_contexts ?? [];
      const nextContext =
        nextAvailableContexts.find(
          (ctx: AuthContext) => ctx.context_id === response.active_context_id
        ) ?? toCtx ?? null;
      assertContextRuntimeInvariants(nextContext);
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
        setActiveContextIdForApi(nextContext.context_id);
        contextRealtimeRouter.setActiveContext(nextContext.context_type);
        if (crossWorkspaceSwitch) {
          prefetchContextTarget(queryClient, nextContext);
        }
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

  const clearSessionState = ReactRuntime.useCallback(
    async (next: { status: SessionStatus; error: string | null; journalEvent: string }) => {
      await performLogout().catch(() => undefined);
      void purgeDriverProfileCache().catch(() => undefined);
      // Phase 0A : quarantaine file GPS — jamais de purge silencieuse
      const ctx = activeContextRef.current;
      if (ctx?.context_type === "driver") {
        const driverId = getDriverIdFromContext(ctx) ?? ctx.context_id;
        const companyId = getCompanyIdFromContext(ctx) ?? "unknown";
        void import("../features/driver/services/driverTrackingQueue")
          .then(({ driverTrackingQueue }) =>
            driverTrackingQueue.quarantineOnLogout({
              userId: ctx.context_id,
              driverId,
              companyId,
            })
          )
          .catch(() => undefined);
      }
      setBootstrap(null);
      setActiveContext(null);
      activeContextRef.current = null;
      setActiveContextIdForApi(null);
      setStatus(next.status);
      setMobileSessionStatus("anonymous");
      setError(next.error);
      setRuntimeFeatureFlagOverrides(null);
      contextRealtimeRouter.setActiveContext(null);
      clearAllContextCache(queryClient);
      realtimeManager.disconnect();
      void appendSessionJournalEvent(next.journalEvent, {
        reason: next.error ?? null,
      });
      void clearSessionJournal();
    },
    [queryClient]
  );

const logout = ReactRuntime.useCallback(async () => {
  await clearSessionState({
    status: "idle",
    error: null,
    journalEvent: "session.logout",
  });
  // Pas de bootstrap automatique après logout — évite les courses avec un login concurrent.
}, [clearSessionState]);

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
