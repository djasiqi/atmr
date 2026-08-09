import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { realtimeManager } from "../../../core/realtime/realtimeManager";

export type RecoveryStep =
  | "restart_watch"
  | "restart_fgs"
  | "restart_socket"
  | "restart_engine";

const STEP_DELAYS_MS: Record<RecoveryStep, number> = {
  restart_watch: 60_000,
  restart_fgs: 120_000,
  restart_socket: 180_000,
  restart_engine: 300_000,
};

let lastCascadeAtMs = 0;
const CASCADE_COOLDOWN_MS = 30 * 60_000;

export type RecoveryHandlers = {
  restartWatch: (reason: string) => Promise<void>;
  restartFgs: (reason: string) => Promise<void>;
  restartEngine: (reason: string) => Promise<void>;
};

export async function runTrackingRecoveryCascade(
  reason: string,
  handlers: RecoveryHandlers
): Promise<void> {
  if (!isFeatureEnabled("tracking_recovery_cascade_enabled")) {
    await handlers.restartWatch(reason);
    return;
  }
  const now = Date.now();
  if (now - lastCascadeAtMs < CASCADE_COOLDOWN_MS) {
    return;
  }
  lastCascadeAtMs = now;

  const steps: { step: RecoveryStep; run: () => Promise<void> }[] = [
    { step: "restart_watch", run: () => handlers.restartWatch(reason) },
    { step: "restart_fgs", run: () => handlers.restartFgs(reason) },
    {
      step: "restart_socket",
      run: async () => {
        const snap = realtimeManager.getSnapshot();
        if (snap.activeContextId) {
          realtimeManager.connect(snap.activeContextId, { enableSocket: true });
        }
        emitDriverTelemetry("tracking.recovery.step", {
          step: "restart_socket",
          reason,
        });
      },
    },
    { step: "restart_engine", run: () => handlers.restartEngine(reason) },
  ];

  for (const { step, run } of steps) {
    const started = Date.now();
    await run();
    emitDriverTelemetry("tracking.recovery.step", {
      step,
      reason,
      elapsed_ms: Date.now() - started,
      delay_ms: STEP_DELAYS_MS[step],
    });
  }
}
