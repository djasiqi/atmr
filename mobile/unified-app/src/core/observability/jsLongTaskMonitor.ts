import { recordJsLongTask } from "./perfInstrumentation";
import { isPerfInstrumentationActive } from "./perfInstrumentationTier";

const FRAME_BUDGET_MS = Number(process.env.EXPO_PUBLIC_PERF_LONG_TASK_RECORD_THRESHOLD_MS ?? "16");
let rafId: number | null = null;
let lastFrameAt = 0;
let running = false;

function onFrame(now: number): void {
  if (!running) return;
  if (lastFrameAt > 0) {
    const delta = now - lastFrameAt;
    if (delta > FRAME_BUDGET_MS) {
      recordJsLongTask(delta);
    }
  }
  lastFrameAt = now;
  rafId = requestAnimationFrame(onFrame);
}

export function startJsLongTaskMonitor(): void {
  if (!isPerfInstrumentationActive() || running) return;
  if (typeof requestAnimationFrame !== "function") return;
  running = true;
  lastFrameAt = 0;
  rafId = requestAnimationFrame(onFrame);
}

export function stopJsLongTaskMonitor(): void {
  running = false;
  if (rafId != null && typeof cancelAnimationFrame === "function") {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  lastFrameAt = 0;
}
