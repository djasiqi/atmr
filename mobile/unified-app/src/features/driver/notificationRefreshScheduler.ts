export const MAX_PENDING_SCOPES = 500;
const COALESCE_WINDOW_MS = 750;

export type RefreshScopeKind = "missions" | "missionDetail" | "chat" | "syncState";

export type RefreshScope = {
  kind: RefreshScopeKind;
  contextId: string;
  missionId?: number;
  threadId?: string;
};

type PendingScope = {
  scopeKey: string;
  flush: () => void;
  timer: ReturnType<typeof setTimeout>;
};

const pendingByScope = new Map<string, PendingScope>();

export function buildRefreshScopeKey(scope: RefreshScope): string {
  if (scope.kind === "missionDetail" && scope.missionId != null) {
    return `${scope.kind}:${scope.contextId}:${scope.missionId}`;
  }
  if (scope.kind === "chat" && scope.threadId) {
    return `${scope.kind}:${scope.contextId}:${scope.threadId}`;
  }
  return `${scope.kind}:${scope.contextId}`;
}

function evictOldestPending(): void {
  if (pendingByScope.size < MAX_PENDING_SCOPES) return;
  const firstKey = pendingByScope.keys().next().value as string | undefined;
  if (!firstKey) return;
  const entry = pendingByScope.get(firstKey);
  if (entry) {
    clearTimeout(entry.timer);
    pendingByScope.delete(firstKey);
  }
}

export function scheduleScopedRefresh(scope: RefreshScope, flush: () => void): void {
  const scopeKey = buildRefreshScopeKey(scope);
  const existing = pendingByScope.get(scopeKey);
  if (existing) {
    clearTimeout(existing.timer);
    pendingByScope.delete(scopeKey);
  } else if (pendingByScope.size >= MAX_PENDING_SCOPES) {
    evictOldestPending();
  }

  const timer = setTimeout(() => {
    pendingByScope.delete(scopeKey);
    flush();
  }, COALESCE_WINDOW_MS);

  pendingByScope.set(scopeKey, { scopeKey, flush, timer });
}

export function flushAllScheduledRefreshesForTests(): void {
  for (const entry of pendingByScope.values()) {
    clearTimeout(entry.timer);
    entry.flush();
  }
  pendingByScope.clear();
}

export function resetNotificationRefreshSchedulerForTests(): void {
  for (const entry of pendingByScope.values()) {
    clearTimeout(entry.timer);
  }
  pendingByScope.clear();
}
