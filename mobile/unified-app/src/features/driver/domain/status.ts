import type { DriverMissionStatus } from "../types";

export const TRACKING_ACTIVE_STATUSES: DriverMissionStatus[] = [
  "ASSIGNED",
  "EN_ROUTE",
  "ARRIVED",
  "IN_PROGRESS",
];

export function isTrackingActiveStatus(status: DriverMissionStatus | null | undefined): boolean {
  if (!status) return false;
  return TRACKING_ACTIVE_STATUSES.includes(status);
}
