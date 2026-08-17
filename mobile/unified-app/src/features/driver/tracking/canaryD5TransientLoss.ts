/**
 * Canary D5-C3 — injecte un trou local court sur le cache missions
 * (QA panel uniquement) pour valider la protection transient loss.
 *
 * Deep link : lirie://canary/d5-c3-transient?hole_ms=1000
 */
import * as Linking from "expo-linking";
import type { QueryClient } from "@tanstack/react-query";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isTrackingQaPanelEnabled } from "../hooks/useDriverBackgroundTrackingUi";
import { driverQueryKeys } from "../queryKeys";

const CANARY_PATH = "canary/d5-c3-transient";
const DEFAULT_HOLE_MS = 1000;

function parseHoleMs(url: string): number {
  try {
    const parsed = Linking.parse(url);
    const raw = parsed.queryParams?.hole_ms;
    const n = typeof raw === "string" ? Number(raw) : Array.isArray(raw) ? Number(raw[0]) : NaN;
    if (Number.isFinite(n) && n >= 200 && n <= 10_000) return Math.floor(n);
  } catch {
    /* ignore */
  }
  return DEFAULT_HOLE_MS;
}

function isCanaryTransientUrl(url: string | null | undefined): boolean {
  if (!url) return false;
  return url.includes(CANARY_PATH);
}

export function installCanaryD5TransientLossInject(opts: {
  queryClient: QueryClient;
  getContextId: () => string | null;
}): () => void {
  if (!isTrackingQaPanelEnabled()) {
    return () => undefined;
  }

  const runInject = (url: string) => {
    if (!isCanaryTransientUrl(url)) return;
    const contextId = opts.getContextId();
    if (!contextId) {
      console.warn("[D5-C3] inject_aborted_no_context");
      return;
    }
    const key = driverQueryKeys.missions(contextId);
    const previous = opts.queryClient.getQueryData(key);
    const holeMs = parseHoleMs(url);
    console.warn("[D5-C3] inject_hole_start", {
      hole_ms: holeMs,
      context_id: contextId,
      had_array: Array.isArray(previous),
      prev_len: Array.isArray(previous) ? previous.length : null,
    });
    emitDriverTelemetry("tracking.lifecycle.canary_c3_inject" as never, {
      source: "driver.canary.d5_c3",
      hole_ms: holeMs,
      context_id: contextId,
    });
    opts.queryClient.setQueryData(key, []);
    setTimeout(() => {
      opts.queryClient.setQueryData(key, previous);
      console.warn("[D5-C3] inject_hole_restore", {
        hole_ms: holeMs,
        context_id: contextId,
      });
      emitDriverTelemetry("tracking.lifecycle.canary_c3_restore" as never, {
        source: "driver.canary.d5_c3",
        hole_ms: holeMs,
        context_id: contextId,
      });
    }, holeMs);
  };

  void Linking.getInitialURL()
    .then((url) => {
      if (url) runInject(url);
    })
    .catch(() => undefined);

  const sub = Linking.addEventListener("url", ({ url }) => {
    runInject(url);
  });

  return () => {
    sub.remove();
  };
}
