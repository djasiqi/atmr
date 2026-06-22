import {
  disposeDriverTrackingBridge,
  flushDriverTrackingQueueNow,
  getDriverTrackingBridgeSnapshot,
  getDriverTrackingPresenceWindowActive,
  getDriverTrackingQueueSnapshot,
  setDriverTrackingPresenceWindow,
  syncBridgeQueueDepthFromPersistence,
  subscribeDriverTrackingBridge,
  startDriverTrackingBridge,
  stopDriverTrackingBridge,
  updateDriverTrackingBridgeStatus,
} from "./services/driverTrackingBridge";
import { DriverMissionStatus, type DriverMission } from "./types";

export type DriverMissionSchedulingContext = Pick<
  DriverMission,
  "scheduled_time" | "time_confirmed" | "scheduling"
>;

export function startDriverTracking(
  missionId: number,
  status: DriverMissionStatus,
  scheduling?: DriverMissionSchedulingContext | null
) {
  startDriverTrackingBridge(missionId, status, scheduling);
}

export function updateDriverTrackingStatus(status: DriverMissionStatus) {
  updateDriverTrackingBridgeStatus(status);
}

export function stopDriverTracking() {
  void stopDriverTrackingBridge();
}

export { syncBridgeQueueDepthFromPersistence };

export function getTrackingSnapshot() {
  return getDriverTrackingBridgeSnapshot();
}

export function subscribeTrackingSnapshot(
  listener: (snapshot: ReturnType<typeof getDriverTrackingBridgeSnapshot>) => void
) {
  return subscribeDriverTrackingBridge(listener);
}

export async function flushTrackingQueue() {
  await flushDriverTrackingQueueNow();
}

export async function getTrackingQueueSnapshot() {
  return getDriverTrackingQueueSnapshot();
}

export function disposeDriverTracking() {
  disposeDriverTrackingBridge();
}

/**
 * Pilote le mode présence par fenêtre horaire (07h–19h). Quand actif sans
 * mission, l'app envoie des points GPS de présence (locationMode = availability_presence).
 */
export function setDriverPresenceWindowActive(active: boolean) {
  setDriverTrackingPresenceWindow(active);
}

export function isDriverPresenceWindowActive(): boolean {
  return getDriverTrackingPresenceWindowActive();
}

