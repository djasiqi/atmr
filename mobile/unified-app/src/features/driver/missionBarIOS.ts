import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { NativeModules, Platform } from "react-native";
import { canUseNotifee, loadNotifee } from "./notifeeCompat";

type LiveActivityPayload = {
  missionId: number;
  status: string;
  etaMinutes?: number | null;
};

export async function configureMissionBarIOS(): Promise<void> {
  if (!canUseNotifee()) {
    emitDriverTelemetry("driver.mission_bar.ios.unavailable", {
      source: "driver.mission_bar.ios",
    });
    return;
  }
  try {
    const mod = await loadNotifee();
    if (!mod) {
      emitDriverTelemetry("driver.mission_bar.ios.unavailable", {
        source: "driver.mission_bar.ios",
      });
      return;
    }
    const { default: notifee } = mod;
    await notifee.setNotificationCategories([
      {
        id: "driver-mission-category",
        actions: [
          { id: "mission_accept", title: "Accept" },
          { id: "mission_reject", title: "Reject" },
        ],
      },
    ]);
  } catch {
    emitDriverTelemetry("driver.mission_bar.ios.unavailable", {
      source: "driver.mission_bar.ios",
    });
  }
}

function getLiveActivityModule():
  | {
      startActivity: (payload: LiveActivityPayload) => Promise<void>;
      updateActivity: (payload: LiveActivityPayload) => Promise<void>;
      endActivity: (missionId: number) => Promise<void>;
    }
  | null {
  if (Platform.OS !== "ios") return null;
  const module = (NativeModules as Record<string, unknown>).DriverLiveActivityModule;
  if (!module || typeof module !== "object") return null;
  return module as {
    startActivity: (payload: LiveActivityPayload) => Promise<void>;
    updateActivity: (payload: LiveActivityPayload) => Promise<void>;
    endActivity: (missionId: number) => Promise<void>;
  };
}

export async function startMissionLiveActivity(payload: LiveActivityPayload): Promise<void> {
  const module = getLiveActivityModule();
  if (!module) {
    emitDriverTelemetry("driver.mission_bar.ios.live_activity_unavailable", {
      source: "driver.mission_bar.ios",
      mission_id: payload.missionId,
    });
    return;
  }
  await module.startActivity(payload);
}

export async function updateMissionLiveActivity(payload: LiveActivityPayload): Promise<void> {
  const module = getLiveActivityModule();
  if (!module) return;
  await module.updateActivity(payload);
}

export async function stopMissionLiveActivity(missionId: number): Promise<void> {
  const module = getLiveActivityModule();
  if (!module) return;
  await module.endActivity(missionId);
}
