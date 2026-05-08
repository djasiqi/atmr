import {
  disposeDriverTrackingBridge,
  flushDriverTrackingQueueNow,
  getDriverTrackingBridgeSnapshot,
  getDriverTrackingPresenceWindowActive,
  getDriverTrackingQueueSnapshot,
  setDriverTrackingPresenceWindow,
  subscribeDriverTrackingBridge,
  startDriverTrackingBridge,
  stopDriverTrackingBridge,
  updateDriverTrackingBridgeStatus,
} from "./services/driverTrackingBridge";
import { DriverMissionStatus } from "./types";

export function startDriverTracking(missionId: number, status: DriverMissionStatus) {
  startDriverTrackingBridge(missionId, status);
}

export function updateDriverTrackingStatus(status: DriverMissionStatus) {
  updateDriverTrackingBridgeStatus(status);
}

export function stopDriverTracking() {
  stopDriverTrackingBridge();
}

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

