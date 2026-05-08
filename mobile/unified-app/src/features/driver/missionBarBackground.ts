import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { canUseNotifee, loadNotifee } from "./notifeeCompat";

export function registerMissionBarBackgroundHandlers(): void {
  if (!canUseNotifee()) {
    emitDriverTelemetry("driver.mission_bar.background.unavailable", {
      source: "driver.mission_bar.background",
    });
    return;
  }
  void (async () => {
    try {
      const mod = await loadNotifee();
      if (!mod) {
        emitDriverTelemetry("driver.mission_bar.background.unavailable", {
          source: "driver.mission_bar.background",
        });
        return;
      }
      const { default: notifee } = mod;
      notifee.onBackgroundEvent(async (event) => {
        const { type, detail } = event;
        const pressActionId =
          detail && typeof detail === "object" && "pressAction" in detail
            ? String((detail as { pressAction?: { id?: string } }).pressAction?.id ?? "")
            : "";
        emitDriverTelemetry("driver.mission_bar.background_event", {
          source: "driver.mission_bar.background",
          event_type: type,
          action_id: pressActionId || null,
          detail: typeof detail === "object" ? "present" : "none",
        });
      });
    } catch {
      emitDriverTelemetry("driver.mission_bar.background.unavailable", {
        source: "driver.mission_bar.background",
      });
    }
  })();
}
