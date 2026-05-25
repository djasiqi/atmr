import { recordJsHeapSnapshot } from "./perfInstrumentation";
import { isPerfInstrumentationActive } from "./perfInstrumentationTier";

const HEAP_SNAPSHOT_INTERVAL_MS = 30_000;

let timer: ReturnType<typeof setInterval> | null = null;
let sessionPeakMb = 0;

type HermesStats = {
  jsHeapSize?: number;
  jsAllocatedBytes?: number;
};

function readHeapUsedMb(): number | null {
  try {
    const g = globalThis as {
      HermesInternal?: { getInstrumentedStats?: () => HermesStats };
      performance?: { memory?: { usedJSHeapSize?: number } };
    };
    const stats = g.HermesInternal?.getInstrumentedStats?.();
    if (stats) {
      const bytes = stats.jsHeapSize ?? stats.jsAllocatedBytes;
      if (typeof bytes === "number" && bytes > 0) {
        return bytes / (1024 * 1024);
      }
    }
    const used = g.performance?.memory?.usedJSHeapSize;
    if (typeof used === "number" && used > 0) {
      return used / (1024 * 1024);
    }
  } catch {
    // ignore
  }
  return null;
}

function snapshotHeap(): void {
  const usedMb = readHeapUsedMb();
  if (usedMb == null) return;
  if (usedMb > sessionPeakMb) sessionPeakMb = usedMb;
  recordJsHeapSnapshot(usedMb, sessionPeakMb);
}

export function startPerfMemoryMonitor(): void {
  if (!isPerfInstrumentationActive() || timer) return;
  snapshotHeap();
  timer = setInterval(snapshotHeap, HEAP_SNAPSHOT_INTERVAL_MS);
}

export function stopPerfMemoryMonitor(): void {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

export function getPerfMemoryPeakMbForTests(): number {
  return sessionPeakMb;
}

export function resetPerfMemoryMonitorForTests(): void {
  stopPerfMemoryMonitor();
  sessionPeakMb = 0;
}
