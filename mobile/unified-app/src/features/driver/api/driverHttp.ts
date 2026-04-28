import type {
  DriverLocationPayload,
  DriverLocationAck,
  DriverMission,
  DriverMissionDetail,
  DriverPushRegistrationPayload,
  DriverStatusTransitionPayload,
} from "../types";
import type { DriverEtaSnapshot, DriverStatusUpdateResult } from "../api";
import {
  getDriverMissionDetail as getDriverMissionDetailFromApi,
  getDriverMissionEta as getDriverMissionEtaFromApi,
  getDriverMissions as getDriverMissionsFromApi,
  getDriverMissionsSince as getDriverMissionsSinceFromApi,
  registerDriverPushToken as registerDriverPushTokenFromApi,
  triggerDriverTestPush as triggerDriverTestPushFromApi,
  sendDriverLocation as sendDriverLocationFromApi,
  updateDriverAvailability as updateDriverAvailabilityFromApi,
  updateDriverMissionStatus as updateDriverMissionStatusFromApi,
} from "../api";
import { mapDriverMission, mapDriverMissionDetail } from "../domain/missionMappers";

export async function getDriverMissions(): Promise<DriverMission[]> {
  const missions = await getDriverMissionsFromApi();
  return missions.map(mapDriverMission);
}

export async function getDriverMissionsSince(sinceIso: string): Promise<DriverMission[]> {
  const missions = await getDriverMissionsSinceFromApi(sinceIso);
  return missions.map(mapDriverMission);
}

export async function getDriverMissionDetail(missionId: number): Promise<DriverMissionDetail> {
  const mission = await getDriverMissionDetailFromApi(missionId);
  return mapDriverMissionDetail(mission);
}

export async function getDriverMissionEta(missionId: number): Promise<DriverEtaSnapshot> {
  return getDriverMissionEtaFromApi(missionId);
}

export async function updateDriverMissionStatus(
  payload: DriverStatusTransitionPayload
): Promise<DriverStatusUpdateResult> {
  return updateDriverMissionStatusFromApi(payload);
}

export async function sendDriverLocation(payload: DriverLocationPayload): Promise<DriverLocationAck> {
  return sendDriverLocationFromApi(payload);
}

export async function registerDriverPushToken(
  payload: DriverPushRegistrationPayload
): Promise<void> {
  await registerDriverPushTokenFromApi(payload);
}

export async function updateDriverAvailability(isAvailable: boolean): Promise<void> {
  await updateDriverAvailabilityFromApi(isAvailable);
}

export async function triggerDriverTestPush(): Promise<void> {
  await triggerDriverTestPushFromApi();
}
