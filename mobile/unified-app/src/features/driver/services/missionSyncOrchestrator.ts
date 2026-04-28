import { QueryClient } from "@tanstack/react-query";
import { reconcileDriverMissions } from "../sync";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";

type Trigger = "reconnect" | "foreground" | "push_update" | "manual";

type SyncState = {
  inFlight: Promise<void> | null;
  queued: boolean;
  timer: ReturnType<typeof setTimeout> | null;
  lastTrigger: Trigger | null;
};

const stateByContext = new Map<string, SyncState>();
const DEBOUNCE_MS = 350;

function getState(contextId: string): SyncState {
  const existing = stateByContext.get(contextId);
  if (existing) return existing;
  const created: SyncState = {
    inFlight: null,
    queued: false,
    timer: null,
    lastTrigger: null,
  };
  stateByContext.set(contextId, created);
  return created;
}

async function runSync(queryClient: QueryClient, contextId: string, trigger: Trigger) {
  const state = getState(contextId);
  state.lastTrigger = trigger;
  if (state.inFlight) {
    state.queued = true;
    return;
  }
  state.inFlight = reconcileDriverMissions(queryClient, contextId)
    .then(() => {
      emitDriverTelemetry("driver.runtime.reconcile", {
        source: "driver.sync.orchestrator",
        context_id: contextId,
        trigger,
      });
    })
    .catch((error) => {
      emitDriverTelemetry("driver.runtime.reconcile.failure", {
        source: "driver.sync.orchestrator",
        context_id: contextId,
        trigger,
        reason: error instanceof Error ? error.message : "reconcile_failed",
      });
    })
    .finally(async () => {
      state.inFlight = null;
      if (state.queued) {
        state.queued = false;
        await runSync(queryClient, contextId, state.lastTrigger ?? "manual");
      }
    });
  await state.inFlight;
}

export function scheduleDriverMissionSync(
  queryClient: QueryClient,
  contextId: string,
  trigger: Trigger
): void {
  const state = getState(contextId);
  if (state.timer) {
    clearTimeout(state.timer);
    state.timer = null;
  }
  state.timer = setTimeout(() => {
    state.timer = null;
    void runSync(queryClient, contextId, trigger);
  }, DEBOUNCE_MS);
}

export function disposeDriverMissionSyncOrchestrator(contextId: string): void {
  const state = stateByContext.get(contextId);
  if (!state) return;
  if (state.timer) {
    clearTimeout(state.timer);
  }
  stateByContext.delete(contextId);
}
