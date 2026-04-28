import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";

export function registerMissionBarBackgroundHandlers(): void {
  void (async () => {
    try {
      const { default: notifee } = await import("@notifee/react-native");
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
