import React, { memo, useCallback } from "react";
import { FlatList, View, type ListRenderItem, type RefreshControlProps } from "react-native";
import { AppText } from "../../../../design/responsive";
import { DispatchRideListCard } from "../DispatchRideListCard";
import {
  EnterpriseActionChip,
  EnterpriseFooterActionRow,
  EnterpriseRoundIconAction,
} from "../EnterpriseActionChip";
import { E } from "../../theme/enterpriseOpsTheme";
import { isDispatchCompleted, isDispatchCancelled } from "../../utils/companyDispatchStatus";
import { canMarkRideUrgent } from "../../utils/pickupSentinel";
import type { CompanyDispatchMission } from "../../api/contracts";
import { Ionicons } from "@expo/vector-icons";
import {
  COMPANY_RIDES_LIST_VIRTUALIZATION,
  areCompanyRidesMissionRowPropsEqual,
  missionListKeyExtractor,
  resolveRideCardDelayMinutes,
  resolveRideCardPickupEtaIso,
  type CompanyRidesMissionRowProps,
} from "./companyRidesMissionRowProps";
import {
  COMPANY_OFFLINE_DAY_BODY,
  COMPANY_OFFLINE_DAY_TITLE,
  type DispatchDayEmptyKind,
} from "../../utils/companyOfflinePolicy";

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
  refreshControl?: React.ReactElement<RefreshControlProps>;
  contentContainerStyle?: object;
  searchQuery?: string;
  isDayComplete?: boolean;
  remainingUnloadedCount?: number;
  emptyKind?: DispatchDayEmptyKind;
  onToggleExpand: (missionId: number) => void;
  onOpenAssign: (missionId: number) => void;
  onGoDetails: (mission: CompanyDispatchMission) => void;
  onEdit: (missionId: number) => void;
  onSchedule: (missionId: number) => void;
  onTransfer: (missionId: number) => void;
  onCancel: (missionId: number) => void;
  onMarkUrgent: (missionId: number) => void;
};

const CompanyRidesMissionRow = memo(function CompanyRidesMissionRow({
  mission,
  isExpanded,
  isActionPending,
  delayMinutes,
  pickupEtaIso,
  contextId,
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
}: CompanyRidesMissionRowProps) {
  const completed = isDispatchCompleted(mission);
  const cancelled = isDispatchCancelled(mission);
  const showUrgent = canMarkRideUrgent(mission);
  const unassigned = mission.driver_id == null;

  const handleToggleExpand = useCallback(() => {
    onToggleExpand(mission.mission_id);
  }, [mission.mission_id, onToggleExpand]);

  const handleOpenAssign = useCallback(() => {
    void onOpenAssign(mission.mission_id);
  }, [mission.mission_id, onOpenAssign]);

  const handleGoDetails = useCallback(() => {
    onGoDetails(mission);
  }, [mission, onGoDetails]);

  const handleEdit = useCallback(() => {
    onEdit(mission.mission_id);
  }, [mission.mission_id, onEdit]);

  const handleSchedule = useCallback(() => {
    void onSchedule(mission.mission_id);
  }, [mission.mission_id, onSchedule]);

  const handleTransfer = useCallback(() => {
    void onTransfer(mission.mission_id);
  }, [mission.mission_id, onTransfer]);

  const handleCancel = useCallback(() => {
    void onCancel(mission.mission_id);
  }, [mission.mission_id, onCancel]);

  const handleMarkUrgent = useCallback(() => {
    void onMarkUrgent(mission.mission_id);
  }, [mission.mission_id, onMarkUrgent]);

  const timeSentinelAction =
    !completed && !cancelled && showUrgent ? (
      <EnterpriseRoundIconAction
        icon="flash"
        variant="urgent"
        accessibilityLabel="Urgence"
        onPress={handleMarkUrgent}
        disabled={!contextId || isActionPending || !canUrgentRide}
        showSpinner={isActionPending}
        spinnerColor="#FFFFFF"
      />
    ) : undefined;

  return (
    <DispatchRideListCard
      mission={mission}
      bookingDelayPickupMinutes={delayMinutes}
      bookingPickupEtaIso={pickupEtaIso}
      expanded={isExpanded}
      onToggleExpand={handleToggleExpand}
      timeSentinelAction={timeSentinelAction}
      onUnassignedPress={
        !completed && !cancelled && unassigned ? handleOpenAssign : undefined
      }
      unassignedPressDisabled={!contextId || !canAssignRide}
      footer={
        isExpanded ? (
          <EnterpriseFooterActionRow>
            <EnterpriseActionChip
              icon="open-outline"
              label="Détails"
              tone="details"
              onPress={handleGoDetails}
            />
            {!completed && !cancelled ? (
              <>
                {mission.driver_id != null ? (
                  <EnterpriseActionChip
                    icon="person-add-outline"
                    label="Réassigner"
                    onPress={handleOpenAssign}
                    disabled={!contextId || !canAssignRide}
                  />
                ) : null}
                <EnterpriseActionChip
                  icon="create-outline"
                  label="Éditer"
                  onPress={handleEdit}
                  disabled={!contextId || !canEditRide}
                />
                <EnterpriseActionChip
                  icon="time-outline"
                  label={isActionPending ? "Planif…" : "Planifier"}
                  onPress={handleSchedule}
                  disabled={!contextId || isActionPending || !canScheduleRide}
                  showSpinner={isActionPending}
                  spinnerColor={E.BRAND}
                />
                <EnterpriseActionChip
                  icon="swap-horizontal-outline"
                  label="Transférer"
                  tone="transfer"
                  onPress={handleTransfer}
                  disabled={!contextId || !canTransferRide}
                />
                <EnterpriseActionChip
                  icon="close-circle-outline"
                  label={isActionPending ? "Annulation…" : "Annuler"}
                  tone="danger"
                  onPress={handleCancel}
                  disabled={!contextId || isActionPending || !canCancelRide}
                />
              </>
            ) : null}
          </EnterpriseFooterActionRow>
        ) : null
      }
    />
  );
}, areCompanyRidesMissionRowPropsEqual);

export function CompanyRidesMissionFlatList({
  missions,
  isLoading,
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
  listHeaderComponent,
  listFooterComponent,
  refreshControl,
  contentContainerStyle,
  searchQuery = "",
  isDayComplete = true,
  remainingUnloadedCount = 0,
  emptyKind,
  onToggleExpand,
  onOpenAssign,
  onGoDetails,
  onEdit,
  onSchedule,
  onTransfer,
  onCancel,
  onMarkUrgent,
}: CompanyRidesMissionFlatListProps) {
  const renderItem: ListRenderItem<CompanyDispatchMission> = useCallback(
    ({ item }) => (
      <CompanyRidesMissionRow
        mission={item}
        isExpanded={expandedMissionId === item.mission_id}
        isActionPending={missionActionPendingId === item.mission_id}
        delayMinutes={resolveRideCardDelayMinutes(
          item,
          dispatchDelaysFetched,
          delayPickupByBookingId
        )}
        pickupEtaIso={resolveRideCardPickupEtaIso(
          item,
          dispatchDelaysFetched,
          pickupEtaByBookingId
        )}
        contextId={contextId}
        canAssignRide={canAssignRide}
        canEditRide={canEditRide}
        canTransferRide={canTransferRide}
        canUrgentRide={canUrgentRide}
        canCancelRide={canCancelRide}
        canScheduleRide={canScheduleRide}
        onToggleExpand={onToggleExpand}
        onOpenAssign={onOpenAssign}
        onGoDetails={onGoDetails}
        onEdit={onEdit}
        onSchedule={onSchedule}
        onTransfer={onTransfer}
        onCancel={onCancel}
        onMarkUrgent={onMarkUrgent}
      />
    ),
    [
      canAssignRide,
      canCancelRide,
      canEditRide,
      canScheduleRide,
      canTransferRide,
      canUrgentRide,
      contextId,
      delayPickupByBookingId,
      dispatchDelaysFetched,
      expandedMissionId,
      missionActionPendingId,
      onCancel,
      onEdit,
      onGoDetails,
      onMarkUrgent,
      onOpenAssign,
      onSchedule,
      onToggleExpand,
      onTransfer,
      pickupEtaByBookingId,
    ]
  );

  const searchActive = searchQuery.trim().length > 0;
  const resolvedEmptyKind: DispatchDayEmptyKind =
    emptyKind ?? (isLoading ? "loading" : searchActive && !isDayComplete ? "search_pending" : searchActive ? "search_none" : "empty");
  const listEmpty =
    resolvedEmptyKind === "offline_unavailable" ? (
    <View style={emptyStyles.emptyWrap} accessibilityRole="text">
      <Ionicons name="cloud-offline-outline" size={28} color={E.BRAND} />
      <AppText variant="body" style={emptyStyles.emptyTitle}>
        {COMPANY_OFFLINE_DAY_TITLE}
      </AppText>
      <AppText variant="caption" style={emptyStyles.emptyCaption}>
        {COMPANY_OFFLINE_DAY_BODY}
      </AppText>
    </View>
  ) : resolvedEmptyKind === "loading" || isLoading ? (
    <View
      style={emptyStyles.skeletonWrap}
      accessibilityRole="progressbar"
      accessibilityLabel="Chargement des courses de ce jour"
    >
      <View style={emptyStyles.skeletonBar64} />
      <View style={emptyStyles.skeletonBar64Muted} />
      <View style={emptyStyles.skeletonBar64Faint} />
      <AppText variant="caption" style={emptyStyles.skeletonCaption}>
        Chargement de cette journée…
      </AppText>
    </View>
  ) : searchActive && !isDayComplete ? (
    <View style={emptyStyles.emptyWrap} accessibilityRole="text">
      <Ionicons name="search-outline" size={28} color={E.BRAND} />
      <AppText variant="body" style={emptyStyles.emptyTitle}>
        Recherche en cours
      </AppText>
      <AppText variant="caption" style={emptyStyles.emptyCaption}>
        {remainingUnloadedCount > 0
          ? `Recherche dans les ${remainingUnloadedCount} courses restantes…`
          : "Recherche dans le reste de la journée…"}
      </AppText>
    </View>
  ) : searchActive ? (
    <View style={emptyStyles.emptyWrap} accessibilityRole="text">
      <Ionicons name="search-outline" size={28} color={E.BRAND} />
      <AppText variant="body" style={emptyStyles.emptyTitle}>
        Aucun résultat
      </AppText>
      <AppText variant="caption" style={emptyStyles.emptyCaption}>
        Aucune course de cette journée ne correspond à « {searchQuery.trim()} ».
      </AppText>
    </View>
  ) : (
    <View style={emptyStyles.emptyWrap} accessibilityRole="text">
      <Ionicons name="car-outline" size={28} color={E.BRAND} />
      <AppText variant="body" style={emptyStyles.emptyTitle}>
        Aucune course
      </AppText>
      <AppText variant="caption" style={emptyStyles.emptyCaption}>
        Aucune course pour ce filtre ou cette date. Utilisez le bouton + pour en créer une.
      </AppText>
    </View>
  );

  return (
    <FlatList
      style={emptyStyles.list}
      data={missions}
      keyExtractor={missionListKeyExtractor}
      renderItem={renderItem}
      ListHeaderComponent={listHeaderComponent}
      ListFooterComponent={listFooterComponent}
      ListEmptyComponent={listEmpty}
      refreshControl={refreshControl}
      contentContainerStyle={contentContainerStyle}
      initialNumToRender={COMPANY_RIDES_LIST_VIRTUALIZATION.initialNumToRender}
      windowSize={COMPANY_RIDES_LIST_VIRTUALIZATION.windowSize}
      maxToRenderPerBatch={COMPANY_RIDES_LIST_VIRTUALIZATION.maxToRenderPerBatch}
      updateCellsBatchingPeriod={COMPANY_RIDES_LIST_VIRTUALIZATION.updateCellsBatchingPeriod}
      removeClippedSubviews={COMPANY_RIDES_LIST_VIRTUALIZATION.removeClippedSubviews}
      keyboardShouldPersistTaps="handled"
    />
  );
}

const emptyStyles = {
  list: { flex: 1 } as const,
  skeletonWrap: { paddingVertical: 12, gap: 10 } as const,
  skeletonBar64: {
    height: 64,
    borderRadius: 14,
    backgroundColor: "rgba(15, 23, 42, 0.06)",
  } as const,
  skeletonBar64Muted: {
    height: 64,
    borderRadius: 14,
    backgroundColor: "rgba(15, 23, 42, 0.05)",
  } as const,
  skeletonBar64Faint: {
    height: 64,
    borderRadius: 14,
    backgroundColor: "rgba(15, 23, 42, 0.04)",
  } as const,
  skeletonCaption: { marginTop: 4, textAlign: "center" as const, opacity: 0.65 },
  emptyWrap: { paddingVertical: 32, alignItems: "center" as const },
  emptyTitle: { marginTop: 12, fontWeight: "600" as const },
  emptyCaption: { marginTop: 4, textAlign: "center" as const, opacity: 0.7 },
};

export default CompanyRidesMissionFlatList;
