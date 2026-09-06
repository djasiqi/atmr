import type { CompanyDispatchMission } from "../../api/contracts";

export type CompanyRidesMissionRowProps = {
  mission: CompanyDispatchMission;
  isExpanded: boolean;
  isActionPending: boolean;
  delayMinutes: number | null | undefined;
  pickupEtaIso: string | null;
  contextId: string | null | undefined;
  canAssignRide: boolean;
  canEditRide: boolean;
  canTransferRide: boolean;
  canUrgentRide: boolean;
  canCancelRide: boolean;
  canScheduleRide: boolean;
  onToggleExpand: (missionId: number) => void;
  onOpenAssign: (missionId: number) => void;
  onGoDetails: (mission: CompanyDispatchMission) => void;
  onEdit: (missionId: number) => void;
  onSchedule: (missionId: number) => void;
  onTransfer: (missionId: number) => void;
  onCancel: (missionId: number) => void;
  onMarkUrgent: (missionId: number) => void;
};

/** Expand / pending : seules les 1–2 cartes concernées voient une prop changer. */
export function areCompanyRidesMissionRowPropsEqual(
  previous: CompanyRidesMissionRowProps,
  next: CompanyRidesMissionRowProps
): boolean {
  return (
    previous.mission === next.mission &&
    previous.isExpanded === next.isExpanded &&
    previous.isActionPending === next.isActionPending &&
    previous.delayMinutes === next.delayMinutes &&
    previous.pickupEtaIso === next.pickupEtaIso &&
    previous.contextId === next.contextId &&
    previous.canAssignRide === next.canAssignRide &&
    previous.canEditRide === next.canEditRide &&
    previous.canTransferRide === next.canTransferRide &&
    previous.canUrgentRide === next.canUrgentRide &&
    previous.canCancelRide === next.canCancelRide &&
    previous.canScheduleRide === next.canScheduleRide &&
    previous.onToggleExpand === next.onToggleExpand &&
    previous.onOpenAssign === next.onOpenAssign &&
    previous.onGoDetails === next.onGoDetails &&
    previous.onEdit === next.onEdit &&
    previous.onSchedule === next.onSchedule &&
    previous.onTransfer === next.onTransfer &&
    previous.onCancel === next.onCancel &&
    previous.onMarkUrgent === next.onMarkUrgent
  );
}

export function resolveRideCardDelayMinutes(
  mission: CompanyDispatchMission,
  dispatchDelaysFetched: boolean,
  delayPickupByBookingId: ReadonlyMap<number, number>
): number | null | undefined {
  const fromAssignment =
    mission.assignment_pickup_delay_minutes != null &&
    mission.assignment_pickup_delay_minutes > 0
      ? Math.round(mission.assignment_pickup_delay_minutes)
      : null;
  if (!dispatchDelaysFetched) {
    return fromAssignment ?? undefined;
  }
  const fromDelays =
    delayPickupByBookingId.get(mission.mission_id) ??
    delayPickupByBookingId.get(Number(mission.mission_id)) ??
    null;
  const etaPositive = typeof fromDelays === "number" && fromDelays > 0 ? fromDelays : null;
  return etaPositive ?? fromAssignment ?? null;
}

export function resolveRideCardPickupEtaIso(
  mission: CompanyDispatchMission,
  dispatchDelaysFetched: boolean,
  pickupEtaByBookingId: ReadonlyMap<number, string>
): string | null {
  if (!dispatchDelaysFetched) return null;
  return (
    pickupEtaByBookingId.get(mission.mission_id) ??
    pickupEtaByBookingId.get(Number(mission.mission_id)) ??
    null
  );
}

export function missionListKeyExtractor(item: CompanyDispatchMission): string {
  return String(item.mission_id);
}

/** Fenêtre FlatList déjà en place — pas de getItemLayout (hauteur variable à l’expand). */
export const COMPANY_RIDES_LIST_VIRTUALIZATION = {
  initialNumToRender: 12,
  windowSize: 8,
  maxToRenderPerBatch: 8,
  updateCellsBatchingPeriod: 50,
  removeClippedSubviews: true,
} as const;
