import { Platform } from "react-native";
import * as Sentry from "@sentry/react-native";

export type FleetMarkerAnimationSkipReason =
  | "missing_previous"
  | "missing_next"
  | "invalid_previous"
  | "invalid_next"
  | "native_not_ready"
  | "marker_unavailable";

const reportedThisSession = new Set<FleetMarkerAnimationSkipReason>();

/** Diagnostic animation native — Android (Sentry) + iOS (log local STOP GATE / S3). */
export function reportFleetMarkerAnimationSkipped(
  reason: FleetMarkerAnimationSkipReason,
  extra?: Record<string, unknown>,
): void {
  if (Platform.OS === "ios") {
    if (__DEV__) {
      console.info("[FleetMarkerAnimationSkipped]", reason, extra ?? {});
    }
    return;
  }
  if (Platform.OS !== "android") {
    return;
  }
  if (reportedThisSession.has(reason)) {
    return;
  }
  reportedThisSession.add(reason);

  try {
    Sentry.captureMessage("FleetMarkerAnimationSkipped", {
      level: "warning",
      tags: { reason },
      extra: extra ?? {},
      fingerprint: ["FleetMarkerAnimationSkipped", reason],
    });
  } catch {
    // monitoring ne doit pas casser la carte
  }
}
