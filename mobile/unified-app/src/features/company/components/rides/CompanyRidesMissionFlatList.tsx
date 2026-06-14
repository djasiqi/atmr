import React, { memo, useCallback } from "react";
import { ActivityIndicator, FlatList, View, type ListRenderItem } from "react-native";
import { AppText } from "../../../../design/responsive";
import { DispatchRideListCard } from "../DispatchRideListCard";
import {
  EnterpriseActionChip,
  EnterpriseFooterActionRow,
  EnterpriseRoundIconAction,
} from "../EnterpriseActionChip";
import { E } from "../../theme/enterpriseOpsTheme";
import { isDispatchCompleted, isDispatchCancelled } from "../../utils/companyDispatchStatus";
import { isTimeUndefined } from "../../utils/pickupSentinel";
import type { CompanyDispatchMission } from "../../api/contracts";
import { Ionicons } from "@expo/vector-icons";

export type CompanyRidesMissionFlatListProps = {
  missions: CompanyDispatchMission[];
  isLoading: boolean;
  expandedMissionId: number | null;
  missionActionPendingId: number | null;
  contextId: string | null | undefined;
  dispatchDelaysFetched: boolean;
  delayPickupByBookingId: Map<number, number>;
  pickupEtaByBookingId: Map<number, string>;
  canAssignRide: boolean;
  canEditRide: boolean;
  canTransferRide: boolean;
  canUrgentRide: boolean;
  canCancelRide: boolean;
  canScheduleRide: boolean;
  listHeaderComponent: React.ReactElement | null;
  listFooterComponent?: React.ReactElement | null;
  refreshControl?: React.ReactElement;
  contentContainerStyle?: object;
  onToggleExpand: (missionId: number) => void;
  onOpenAssign: (missionId: number) => void;
  onGoDetails: (mission: CompanyDispatchMission) => void;
  onEdit: (missionId: number) => void;
  onSchedule: (missionId: number) => void;
  onTransfer: (missionId: number) => void;
  onCancel: (missionId: number) => void;
  onMarkUrgent: (missionId: number) => void;
};

function missionListKey(item: CompanyDispatchMission) {
  return String(item.mission_id);
}

function resolveDelayMinutes(
  mission: CompanyDispatchMission,
  dispatchDelaysFetched: boolean,
  delayPickupByBookingId: Map<number, number>
): number | null | undefined {
  const fromAssignment =
    mission.assignment_pickup_delay_minutes != null && mission.assignment_pickup_delay_minutes > 0
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

type MissionRowProps = Omit<
  CompanyRidesMissionFlatListProps,
  "missions" | "isLoading" | "listHeaderComponent" | "listFooterComponent" | "refreshControl" | "contentContainerStyle"
> & {
  mission: CompanyDispatchMission;
};

const CompanyRidesMissionRow = memo(function CompanyRidesMissionRow({
  mission,
  expandedMissionId,
  missionActionPendingId,
  contextId,
  dispatchDelaysFetched,
  delayPickupByBookingId,
  pickupEtaByBookingId,
  canAssignRide,
  canEditRide,
  canTransferRide,
  canUrgentRide,
  canCancelRide,
  canScheduleRide,
  onToggleExpand,
  onOpenAssign,
  onGoDetails,
  onEdit,
  onSchedule,
  onTransfer,
  onCancel,
  onMarkUrgent,
}: MissionRowProps) {
  const isExpanded = expandedMissionId === mission.mission_id;
  const thisBusy = missionActionPendingId === mission.mission_id;
  const completed = isDispatchCompleted(mission);
  const cancelled = isDispatchCancelled(mission);
  const showUrgent = isTimeUndefined(mission);
  const unassigned = mission.driver_id == null;

  const timeSentinelAction =
    !completed && !cancelled && showUrgent ? (
      <EnterpriseRoundIconAction
        icon="flash"
        variant="urgent"
        accessibilityLabel="Urgence"
        onPress={() => void onMarkUrgent(mission.mission_id)}
        disabled={!contextId || thisBusy || !canUrgentRide}
        showSpinner={thisBusy}
        spinnerColor="#FFFFFF"
      />
    ) : undefined;

  return (
    <DispatchRideListCard
      mission={mission}
      bookingDelayPickupMinutes={resolveDelayMinutes(
        mission,
        dispatchDelaysFetched,
        delayPickupByBookingId
      )}
      bookingPickupEtaIso={
        dispatchDelaysFetched
          ? pickupEtaByBookingId.get(mission.mission_id) ??
            pickupEtaByBookingId.get(Number(mission.mission_id)) ??
            null
          : null
      }
      expanded={isExpanded}
      onToggleExpand={() => onToggleExpand(mission.mission_id)}
      timeSentinelAction={timeSentinelAction}
      onUnassignedPress={
        !completed && !cancelled && unassigned ? () => void onOpenAssign(mission.mission_id) : undefined
      }
      unassignedPressDisabled={!contextId || !canAssignRide}
      footer={
        isExpanded ? (
          <EnterpriseFooterActionRow>
            <EnterpriseActionChip
              icon="open-outline"
              label="Détails"
              tone="details"
              onPress={() => onGoDetails(mission)}
            />
            {!completed && !cancelled ? (
              <>
                {mission.driver_id != null ? (
                  <EnterpriseActionChip
                    icon="person-add-outline"
                    label="Réassigner"
                    onPress={() => void onOpenAssign(mission.mission_id)}
                    disabled={!contextId || !canAssignRide}
                  />
                ) : null}
                <EnterpriseActionChip
                  icon="create-outline"
                  label="Éditer"
                  onPress={() => onEdit(mission.mission_id)}
                  disabled={!contextId || !canEditRide}
                />
                <EnterpriseActionChip
                  icon="time-outline"
                  label={thisBusy ? "Planif…" : "Planifier"}
                  onPress={() => void onSchedule(mission.mission_id)}
                  disabled={!contextId || thisBusy || !canScheduleRide}
                  showSpinner={thisBusy}
                  spinnerColor={E.BRAND}
                />
                <EnterpriseActionChip
                  icon="swap-horizontal-outline"
                  label="Transférer"
                  tone="transfer"
                  onPress={() => void onTransfer(mission.mission_id)}
                  disabled={!contextId || !canTransferRide}
                />
                <EnterpriseActionChip
                  icon="close-circle-outline"
                  label={thisBusy ? "Annulation…" : "Annuler"}
                  tone="danger"
                  onPress={() => void onCancel(mission.mission_id)}
                  disabled={!contextId || thisBusy || !canCancelRide}
                />
              </>
            ) : null}
          </EnterpriseFooterActionRow>
        ) : null
      }
    />
  );
});

export function CompanyRidesMissionFlatList(props: CompanyRidesMissionFlatListProps) {
  const {
    missions,
    isLoading,
    listHeaderComponent,
    listFooterComponent,
    refreshControl,
    contentContainerStyle,
  } = props;

  const renderItem: ListRenderItem<CompanyDispatchMission> = useCallback(
    ({ item }) => <CompanyRidesMissionRow mission={item} {...props} />,
    [
      props.expandedMissionId,
      props.missionActionPendingId,
      props.contextId,
      props.dispatchDelaysFetched,
      props.delayPickupByBookingId,
      props.pickupEtaByBookingId,
      props.canAssignRide,
      props.canEditRide,
      props.canTransferRide,
      props.canUrgentRide,
      props.canCancelRide,
      props.canScheduleRide,
      props.onToggleExpand,
      props.onOpenAssign,
      props.onGoDetails,
      props.onEdit,
      props.onSchedule,
      props.onTransfer,
      props.onCancel,
      props.onMarkUrgent,
    ]
  );

  const listEmpty = isLoading ? (
    <View style={{ paddingVertical: 24, alignItems: "center" }} accessibilityRole="progressbar">
      <ActivityIndicator color={E.BRAND} />
      <AppText variant="bodyMuted" style={{ marginTop: 8 }}>
        Chargement…
      </AppText>
    </View>
  ) : (
    <View style={{ paddingVertical: 32, alignItems: "center" }} accessibilityRole="text">
      <Ionicons name="car-outline" size={28} color={E.BRAND} />
      <AppText variant="body" style={{ marginTop: 12, fontWeight: "600" }}>
        Aucune course
      </AppText>
      <AppText variant="caption" style={{ marginTop: 4, textAlign: "center", opacity: 0.7 }}>
        Aucune course pour ce filtre ou cette date. Utilisez le bouton + pour en créer une.
      </AppText>
    </View>
  );

  return (
    <FlatList
      style={{ flex: 1 }}
      data={isLoading ? [] : missions}
      keyExtractor={missionListKey}
      renderItem={renderItem}
      ListHeaderComponent={listHeaderComponent}
      ListFooterComponent={listFooterComponent}
      ListEmptyComponent={listEmpty}
      refreshControl={refreshControl}
      contentContainerStyle={contentContainerStyle}
      initialNumToRender={12}
      windowSize={8}
      maxToRenderPerBatch={8}
      removeClippedSubviews
      keyboardShouldPersistTaps="handled"
    />
  );
}

export default CompanyRidesMissionFlatList;
