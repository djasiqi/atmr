import {
  disposeDriverTrackingBridge,
  flushDriverTrackingQueueNow,
  getDriverTrackingBridgeSnapshot,
  getDriverTrackingPresenceContext,
  getDriverTrackingPresenceWindowActive,
  getDriverTrackingQueueSnapshot,
  setDriverTrackingPresenceContext,
  setDriverTrackingPresenceWindow,
  syncBridgeQueueDepthFromPersistence,
  subscribeDriverTrackingBridge,
  startDriverTrackingBridge,
  stopDriverTrackingBridge,
  updateDriverTrackingBridgeStatus,
  type DriverPresenceContext,
} from "./services/driverTrackingBridge";
import { DriverMissionStatus, type DriverMission } from "./types";

export type { DriverPresenceContext };

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

/** Démonte le contexte mission ; si toujours éligible, le bridge retombe en PRESENCE (pas un STOP natif). */
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
 * Pilote le signal présence (Driver.is_available).
 * La décision start/stop passe par `resolveTrackingEligibility`.
 * `available=null` = UNKNOWN : pas PRESENCE/LIVE, pas hors service.
 */
export function setDriverPresenceContext(ctx: DriverPresenceContext) {
  setDriverTrackingPresenceContext(ctx);
}

/**
 * @deprecated Préférer `setDriverPresenceContext`.
 */
export function setDriverPresenceWindowActive(active: boolean) {
  setDriverTrackingPresenceWindow(active);
}

export function isDriverPresenceWindowActive(): boolean {
  return getDriverTrackingPresenceWindowActive();
}

export function getDriverPresenceContext(): DriverPresenceContext {
  return getDriverTrackingPresenceContext();
}

