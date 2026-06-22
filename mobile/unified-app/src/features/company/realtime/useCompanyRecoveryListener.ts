/**
 * Phase 2 PR B/C — gate D3.2 (fix G2 + G3)
 *
 * Hook unique qui consomme :
 *   - `company_data_stale_resync` (dispatché par `companyRealtimeBridge` après 5 min sans event)
 *   - `company_socket_reconnected` (dispatché par `companyRealtimeBridge` au reconnect)
 *
 * Sans ce listener, après un background long ou un reconnect, seul GPS était
 * rafraîchi (`useCompanyDriverLiveTracking`). Dashboard, missions, inbox et chat
 * restaient silencieusement obsolètes.
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

export const RECOVERY_THROTTLE_MS = 30_000;

export type RecoveryTrigger = "stale" | "reconnect";

type RecoveryEvent = {
  event_type?: string;
};

function isRecoveryEvent(input: unknown): input is RecoveryEvent {
  return Boolean(input) && typeof input === "object";
}

export function resolveRecoveryTrigger(eventType: string | undefined): RecoveryTrigger | null {
  if (eventType === "company_data_stale_resync") return "stale";
  if (eventType === "company_socket_reconnected") return "reconnect";
  return null;
}

/**
 * Exécute le resync coherent dashboard + missions + inbox + chat + delays.
 * Exporté pour testabilité (le hook lui-même est un thin wrapper React).
 */
export function performCompanyRecoveryResync(
  queryClient: QueryClient,
  contextId: string,
  trigger: RecoveryTrigger
): void {
  recordCompanyRecoveryResync(trigger);

  const scope = companyContextScope(contextId);
  const dashboardKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.dashboard(contextId)] as unknown[]
  );
  const missionsKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.root, "missions", scope] as unknown[]
  );
  const inboxKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.inbox(contextId)] as unknown[]
  );
  const dispatchDelaysKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
  );
  const chatKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.root, "chat", contextId] as unknown[]
  );

  void traceInvalidateQueries(dashboardKey, `recovery_resync_${trigger}_dashboard`, async () => {
    await queryClient.invalidateQueries({ queryKey: dashboardKey, exact: true });
  });
  void traceInvalidateQueries(missionsKey, `recovery_resync_${trigger}_missions`, async () => {
    await queryClient.invalidateQueries({ queryKey: missionsKey, exact: false });
  });
  void traceInvalidateQueries(inboxKey, `recovery_resync_${trigger}_inbox`, async () => {
    await queryClient.invalidateQueries({ queryKey: inboxKey, exact: false });
  });
  void traceInvalidateQueries(dispatchDelaysKey, `recovery_resync_${trigger}_delays`, async () => {
    await queryClient.invalidateQueries({ queryKey: dispatchDelaysKey, exact: false });
  });
  void traceInvalidateQueries(chatKey, `recovery_resync_${trigger}_chat`, async () => {
    await queryClient.invalidateQueries({ queryKey: chatKey, exact: false });
  });
  const institutionOffersKey = contextScopedKey(
    contextId,
    [...companyQueryKeys.institutionOffers(contextId, "PENDING")] as unknown[]
  );
  void traceInvalidateQueries(
    institutionOffersKey,
    `recovery_resync_${trigger}_institution_offers`,
    async () => {
      await queryClient.invalidateQueries({ queryKey: institutionOffersKey, exact: false });
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
      performCompanyRecoveryResync(queryClient, contextId, trigger);
    });
  }, [contextId, queryClient]);
}
