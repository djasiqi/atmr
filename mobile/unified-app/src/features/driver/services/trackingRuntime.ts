import { DriverTrackingMode } from "../runtimeContracts";

type TrackingRuntimeState = {
  missionId: number | null;
  mode: DriverTrackingMode;
  lastReconcileAt: string | null;
};

const STORAGE_KEY = "driver_tracking_runtime_v1";
let state: TrackingRuntimeState = {
  missionId: null,
  mode: "off",
  lastReconcileAt: null,
};

export function getTrackingRuntimeState(): TrackingRuntimeState {
  return { ...state };
}

export async function hydrateTrackingRuntimeState(): Promise<TrackingRuntimeState> {
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    const raw = await storage.default.getItem(STORAGE_KEY);
    if (!raw) return getTrackingRuntimeState();
    const parsed = JSON.parse(raw) as TrackingRuntimeState;
    if (parsed && typeof parsed === "object") {
      state = {
        missionId: typeof parsed.missionId === "number" ? parsed.missionId : null,
        mode:
          typeof parsed.mode === "string"
            ? (parsed.mode as DriverTrackingMode)
            : "off",
        lastReconcileAt:
          typeof parsed.lastReconcileAt === "string" ? parsed.lastReconcileAt : null,
      };
    }
  } catch {
    // best effort
  }
  return getTrackingRuntimeState();
}

export async function updateTrackingRuntimeState(
  patch: Partial<TrackingRuntimeState>
): Promise<TrackingRuntimeState> {
  state = {
    ...state,
    ...patch,
    lastReconcileAt: new Date().toISOString(),
  };
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    await storage.default.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // best effort
  }
  return getTrackingRuntimeState();
}
