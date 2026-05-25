import type { QueryClient } from "@tanstack/react-query";
import { invalidateDriverMissionScope } from "./queryKeys";
import {
  buildRefreshScopeKey,
  scheduleScopedRefresh,
  type RefreshScope,
} from "./notificationRefreshScheduler";
import {
  emitNotificationResyncCompleted,
  emitNotificationResyncStarted,
} from "../../core/notifications/notificationTelemetry";

export const MIN_REFRESH_INTERVAL_MS = 1500;

const DRIVER_CHAT_QUERY_PREFIX = ["driver", "chat"] as const;

type DriverRealtimeSyncConfig = {
  queryClient: QueryClient;
  getContextId: () => string | null;
};

let config: DriverRealtimeSyncConfig | null = null;
const lastRefreshAtByScope = new Map<string, number>();

export function configureDriverRealtimeSync(next: DriverRealtimeSyncConfig | null): void {
  config = next;
}

function shouldSkipScopeRefresh(scopeKey: string): boolean {
  const last = lastRefreshAtByScope.get(scopeKey) ?? 0;
  const now = Date.now();
  if (now - last < MIN_REFRESH_INTERVAL_MS) {
    return true;
  }
  lastRefreshAtByScope.set(scopeKey, now);
  return false;
}

function runScopedRefresh(scope: RefreshScope, reason: string, missionId?: number | null): void {
  if (!config) return;
  const contextId = config.getContextId();
  if (!contextId) return;

  const scopeKey = buildRefreshScopeKey(scope);
  if (shouldSkipScopeRefresh(scopeKey)) return;

  emitNotificationResyncStarted({
    reason,
    scope: scopeKey,
    mission_id: missionId ?? null,
    context_id: contextId,
  });

  scheduleScopedRefresh(scope, () => {
    const client = config?.queryClient;
    const ctx = config?.getContextId();
    if (!client || !ctx) return;

    if (scope.kind === "missions" || scope.kind === "missionDetail" || scope.kind === "syncState") {
      invalidateDriverMissionScope(
        client,
        ctx,
        scope.kind === "missionDetail" ? scope.missionId : missionId ?? undefined
      );
    }
    if (scope.kind === "chat" && scope.threadId) {
      void client.invalidateQueries({
        queryKey: [...DRIVER_CHAT_QUERY_PREFIX, ctx, scope.threadId],
      });
      void client.invalidateQueries({
        queryKey: [...DRIVER_CHAT_QUERY_PREFIX, ctx],
      });
    }

    emitNotificationResyncCompleted({
      reason,
      scope: scopeKey,
      mission_id: missionId ?? null,
      context_id: ctx,
    });
  });
}

export function requestMissionRefresh(reason: string, missionId?: number | null): void {
  const contextId = config?.getContextId();
  if (!contextId) return;
  runScopedRefresh({ kind: "missions", contextId }, reason, missionId);
  if (missionId != null) {
    runScopedRefresh({ kind: "missionDetail", contextId, missionId }, reason, missionId);
  }
}

export function requestMissionDetailRefresh(missionId: number, reason: string): void {
  const contextId = config?.getContextId();
  if (!contextId) return;
  runScopedRefresh({ kind: "missionDetail", contextId, missionId }, reason, missionId);
}

export function requestChatRefresh(threadId: string, reason: string): void {
  const contextId = config?.getContextId();
  if (!contextId) return;
  runScopedRefresh({ kind: "chat", contextId, threadId }, reason, null);
}

export function resetDriverRealtimeSyncForTests(): void {
  lastRefreshAtByScope.clear();
  config = null;
}
