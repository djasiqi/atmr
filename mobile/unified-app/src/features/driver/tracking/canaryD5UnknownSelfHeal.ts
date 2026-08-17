/**
 * Canary D5-C4 — force fraîcheur UNKNOWN puis recovery L1 non destructif
 * (QA panel uniquement).
 *
 * Deep link : lirie://canary/d5-c4-unknown-l1?started_age_sec=120
 */
import * as Linking from "expo-linking";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isTrackingQaPanelEnabled } from "../hooks/useDriverBackgroundTrackingUi";
import {
  __canaryD5ApplyUnknownFreshness,
  __canaryD5RestoreFreshness,
  __canaryD5SnapshotFreshness,
  __canaryD5WouldTriggerAntiZombie,
  forceRestartTrackingWatchFromBridge,
  getDriverTrackingBridgeSnapshot,
  type CanaryFreshnessSnapshot,
} from "../services/driverTrackingBridge";

const CANARY_PATH = "canary/d5-c4-unknown-l1";
const DEFAULT_STARTED_AGE_SEC = 120;
const RESTORE_MS = 5_000;

function parseStartedAgeSec(url: string): number {
  try {
    const parsed = Linking.parse(url);
    const raw = parsed.queryParams?.started_age_sec;
    const n = typeof raw === "string" ? Number(raw) : Array.isArray(raw) ? Number(raw[0]) : NaN;
    if (Number.isFinite(n) && n >= 60 && n <= 600) return Math.floor(n);
  } catch {
    /* ignore */
  }
  return DEFAULT_STARTED_AGE_SEC;
}

function isCanaryUnknownUrl(url: string | null | undefined): boolean {
  if (!url) return false;
  return url.includes(CANARY_PATH);
}

export function installCanaryD5UnknownSelfHealInject(): () => void {
  if (!isTrackingQaPanelEnabled()) {
    return () => undefined;
  }

  let restoreTimer: ReturnType<typeof setTimeout> | null = null;
  let prevFreshness: CanaryFreshnessSnapshot | null = null;

  const runInject = (url: string) => {
    if (!isCanaryUnknownUrl(url)) return;
    const snap = getDriverTrackingBridgeSnapshot();
    if (snap.missionId == null || !snap.isRunning) {
      console.warn("[D5-C4] inject_aborted_no_active_tracking", {
        mission_id: snap.missionId,
        is_running: snap.isRunning,
      });
      return;
    }

    if (restoreTimer) {
      clearTimeout(restoreTimer);
      restoreTimer = null;
    }

    const startedAgeSec = parseStartedAgeSec(url);
    prevFreshness = __canaryD5ApplyUnknownFreshness(startedAgeSec);
    const wouldTrigger = __canaryD5WouldTriggerAntiZombie();
    const after = __canaryD5SnapshotFreshness();

    console.warn("[D5-C4] inject_unknown_start", {
      mission_id: snap.missionId,
      started_age_sec: startedAgeSec,
      last_sent_at: after.lastSentAt,
      last_fix_ms: after.lastFixProducedAtMs,
      tracking_started_at_ms: after.trackingStartedAtMs,
      anti_zombie_would_trigger: wouldTrigger,
    });
    emitDriverTelemetry("tracking.lifecycle.canary_c4_inject" as never, {
      source: "driver.canary.d5_c4",
      mission_id: snap.missionId,
      started_age_sec: startedAgeSec,
      anti_zombie_would_trigger: wouldTrigger,
    });

    if (wouldTrigger) {
      console.warn("[D5-C4] FAIL_unknown_still_triggers_anti_zombie");
    } else {
      console.warn("[D5-C4] unknown_no_anti_zombie", {
        mission_id: snap.missionId,
      });
    }

    void forceRestartTrackingWatchFromBridge("canary_c4_l1")
      .then((ok) => {
        console.warn("[D5-C4] l1_restart_done", {
          ok,
          mission_id: snap.missionId,
          recovery_level: "L1",
          destructive: false,
        });
        emitDriverTelemetry("tracking.lifecycle.canary_c4_l1" as never, {
          source: "driver.canary.d5_c4",
          mission_id: snap.missionId,
          ok,
          recovery_level: "L1",
        });
      })
      .catch((err: unknown) => {
        console.warn("[D5-C4] l1_restart_error", {
          message: err instanceof Error ? err.message : String(err),
        });
      });

    restoreTimer = setTimeout(() => {
      if (prevFreshness) {
        __canaryD5RestoreFreshness(prevFreshness);
        console.warn("[D5-C4] inject_freshness_restore", {
          mission_id: snap.missionId,
        });
        prevFreshness = null;
      }
      restoreTimer = null;
    }, RESTORE_MS);
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
    if (restoreTimer) clearTimeout(restoreTimer);
    sub.remove();
  };
}
