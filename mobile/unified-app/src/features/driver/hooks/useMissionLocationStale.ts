import { useEffect, useRef, useState } from "react";

import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isTrackingActiveStatus } from "../domain/status";
import type { DriverMissionStatus } from "../types";
import { getTrackingSnapshot, subscribeTrackingSnapshot } from "../tracking";

const FIX_STALE_THRESHOLD_SECONDS = 300;
const TELEMETRY_DEBOUNCE_MS = 5 * 60_000;

function computeLastFixAgeSeconds(
  lastWatchAt: string | null,
  lastSentAt: string | null
): number | null {
  const candidates: number[] = [];
  for (const iso of [lastWatchAt, lastSentAt]) {
    if (!iso) continue;
    const ts = Date.parse(iso);
    if (!Number.isFinite(ts)) continue;
    const ageMs = Date.now() - ts;
    candidates.push(Math.max(0, Math.round(ageMs / 1000)));
  }
  if (candidates.length === 0) return null;
  return Math.min(...candidates);
}

export function useMissionLocationStale(missionId: number | null, missionStatus: string | null) {
  const [isStale, setIsStale] = useState(false);
  const lastTelemetryAtRef = useRef(0);

  useEffect(() => {
    const evaluate = () => {
      const snapshot = getTrackingSnapshot();
      const missionLive =
        missionId != null &&
        snapshot.missionId === missionId &&
        isTrackingActiveStatus(missionStatus as DriverMissionStatus);

      if (!missionLive) {
        setIsStale(false);
        return;
      }

      const ageSeconds = computeLastFixAgeSeconds(
        snapshot.lastWatchAt ?? null,
        snapshot.lastSentAt ?? null
      );
      const stale =
        ageSeconds !== null && ageSeconds > FIX_STALE_THRESHOLD_SECONDS;
      setIsStale(stale);

      if (
        stale &&
        Date.now() - lastTelemetryAtRef.current >= TELEMETRY_DEBOUNCE_MS
      ) {
        lastTelemetryAtRef.current = Date.now();
        emitDriverTelemetry("tracking.stale_fix_during_mission", {
          source: "driver.mission_location_stale",
          mission_id: missionId,
          last_fix_age_seconds: ageSeconds,
        });
      }
    };

    evaluate();
    const unsub = subscribeTrackingSnapshot(evaluate);
    const interval = setInterval(evaluate, 30_000);
    return () => {
      unsub();
      clearInterval(interval);
    };
  }, [missionId, missionStatus]);

  return isStale;
}
