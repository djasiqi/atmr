/**
 * Phase 2 PR B/C — gate D3.2 (fix G2 + G3)
 *
 * Hook unique qui consomme :
 *   - `company_data_stale_resync` (dispatché par `companyRealtimeBridge` après 5 min sans event)
 *   - `company_socket_reconnected` (dispatché par `companyRealtimeBridge` au reconnect)
 *
 * Sans ce listener, après un background long ou un reconnect, seul GPS était
 * rafraîchi. OPT-08 : dashboard + J observé + live (+ delays), pas inbox/chat/offres.
 *
 * Throttle 30 s par `contextId` pour éviter un storm d'invalidations sur des
 * reconnects rapprochés (transition WiFi/4G, captive portal, etc.).
 */

import { useEffect, useRef } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import { contextScopedKey } from "../../../core/cache/contextCache";
import { companyContextScope, companyQueryKeys } from "../companyQueryKeys";
import { recordCompanyRecoveryResync } from "../../../core/observability/realtimeMetrics";
import { traceInvalidateQueries } from "../../../core/observability/perfInstrumentation";
import {
  reconcileAuthoritativeMission,
  refetchExactDispatchDay,
  refetchObservedDispatchDays,
} from "../utils/dispatchMissionCachePatch";
import type { RidesFetchReason } from "../utils/ridesFetchReason";

export const RECOVERY_THROTTLE_MS = 30_000;

export type RecoveryTrigger = "stale" | "reconnect";

type RecoveryEvent = {
  event_type?: string;
  mission_id?: number;
  missionId?: number;
  date?: string;
};

function isRecoveryEvent(input: unknown): input is RecoveryEvent {
  return Boolean(input) && typeof input === "object";
}

export function resolveRecoveryTrigger(eventType: string | undefined): RecoveryTrigger | null {
  if (eventType === "company_data_stale_resync") return "stale";
  if (eventType === "company_socket_reconnected") return "reconnect";
  return null;
}

function recoveryRidesReason(trigger: RecoveryTrigger): RidesFetchReason {
  return trigger === "reconnect" ? "reconnect" : "recovery";
}

function parseRecoveryMissionId(event?: RecoveryEvent): number | undefined {
  const raw = event?.mission_id ?? event?.missionId;
  return typeof raw === "number" && Number.isFinite(raw) ? raw : undefined;
}

/**
 * OPT-04E + OPT-08 — resync ciblé : J observé, dashboard, delays, live.
 * Pas de famille rides, pas d’inbox / chat / offres / billing au reconnect.
 */
export function performCompanyRecoveryResync(
  queryClient: QueryClient,
  contextId: string,
  trigger: RecoveryTrigger,
  event?: RecoveryEvent
): void {
  recordCompanyRecoveryResync(trigger);
  const ridesReason = recoveryRidesReason(trigger);
  const missionId = parseRecoveryMissionId(event);
  const knownDate = typeof event?.date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(event.date)
    ? event.date
    : undefined;

  if (missionId != null) {
    void reconcileAuthoritativeMission(queryClient, contextId, missionId, ridesReason);
  } else if (knownDate) {
    void refetchExactDispatchDay(queryClient, contextId, knownDate, ridesReason);
  } else {
    void refetchObservedDispatchDays(queryClient, contextId, ridesReason);
  }

  const scope = companyContextScope(contextId);
  const dashboardKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.dashboard(contextId)] as unknown[]
  );
  const dispatchDelaysKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
  );

  void traceInvalidateQueries(dashboardKey, `recovery_resync_${trigger}_dashboard`, async () => {
    await queryClient.invalidateQueries({ queryKey: dashboardKey, exact: true });
  });
  void traceInvalidateQueries(dispatchDelaysKey, `recovery_resync_${trigger}_delays`, async () => {
    await queryClient.invalidateQueries({ queryKey: dispatchDelaysKey, exact: false });
  });
  const driversLocationsKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.driversLocations(contextId)] as unknown[]
  );
  void traceInvalidateQueries(
    driversLocationsKey,
    `recovery_resync_${trigger}_drivers_locations`,
    async () => {
      await queryClient.invalidateQueries({ queryKey: driversLocationsKey, exact: false });
    }
  );
}

export function useCompanyRecoveryListener(contextId: string | null): void {
  const queryClient = useQueryClient();
  const lastResyncAtRef = useRef<number>(0);

  useEffect(() => {
    if (!contextId) return undefined;

    return contextRealtimeRouter.subscribe(contextId, (event) => {
      if (!isRecoveryEvent(event)) return;
      const trigger = resolveRecoveryTrigger(event.event_type);
      if (!trigger) return;

      const now = Date.now();
      if (now - lastResyncAtRef.current < RECOVERY_THROTTLE_MS) {
        return;
      }
      lastResyncAtRef.current = now;
      performCompanyRecoveryResync(queryClient, contextId, trigger, event);
    });
  }, [contextId, queryClient]);
}
