import { DriverMission } from "../types";

type MissionStateSnapshot = {
  activeMissionId: number | null;
  lastHydratedAt: string | null;
  updatedAt: string | null;
};

const STORAGE_KEY = "driver_mission_state_v1";
let snapshot: MissionStateSnapshot = {
  activeMissionId: null,
  lastHydratedAt: null,
  updatedAt: null,
};

export async function hydrateMissionState(): Promise<MissionStateSnapshot> {
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    const raw = await storage.default.getItem(STORAGE_KEY);
    if (!raw) return snapshot;
    const parsed = JSON.parse(raw) as MissionStateSnapshot;
    if (parsed && typeof parsed === "object") {
      snapshot = {
        activeMissionId:
          typeof parsed.activeMissionId === "number" ? parsed.activeMissionId : null,
        lastHydratedAt: new Date().toISOString(),
        updatedAt: typeof parsed.updatedAt === "string" ? parsed.updatedAt : null,
      };
    }
  } catch {
    // best effort
  }
  return snapshot;
}

export function getMissionStateSnapshot(): MissionStateSnapshot {
  return { ...snapshot };
}

export async function setActiveMissionFromList(missions: DriverMission[]): Promise<void> {
  const first = missions.find((mission) => {
    const status = String(mission.status ?? "").toUpperCase();
    return status !== "COMPLETED" && status !== "CANCELLED" && status !== "FAILED";
  });
  snapshot = {
    activeMissionId: first?.id ?? null,
    lastHydratedAt: snapshot.lastHydratedAt,
    updatedAt: new Date().toISOString(),
  };
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    await storage.default.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  } catch {
    // best effort
  }
}
