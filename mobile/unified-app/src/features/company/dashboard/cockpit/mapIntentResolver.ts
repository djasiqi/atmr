/** Intention opérationnelle — distincte du mode visuel. */
export type MapIntent =
  | "FleetOverviewIntent"
  | "DispatchIntent"
  | "DriverMonitoringIntent"
  | "IncidentResponseIntent";

export type MapIntentInput = {
  searchActive: boolean;
  filtersOpen: boolean;
  selectedDriverId: number | null;
  driverSheetOpen: boolean;
  urgentCount: number;
  dispatchActionOpen?: boolean;
};

export function resolveMapIntent(input: MapIntentInput): MapIntent {
  if (input.urgentCount > 0 && !input.driverSheetOpen) {
    return "IncidentResponseIntent";
  }
  if (input.dispatchActionOpen || input.filtersOpen) {
    return "DispatchIntent";
  }
  if (input.selectedDriverId != null && input.driverSheetOpen) {
    return "DriverMonitoringIntent";
  }
  if (input.searchActive) {
    return "FleetOverviewIntent";
  }
  return "FleetOverviewIntent";
}
