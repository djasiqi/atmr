import { useMemo, useState } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useActiveMissionScreenTracking } from "../../../../src/core/notifications/useActiveScreenTracking";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import {
  useDriverMissionDetailQuery,
  useDriverMissionDetailOpenResync,
  useDriverStatusTransition,
  useDynamicEtaQuery,
} from "../../../../src/features/driver/hooks";
import { getDriverStatusUx } from "../../../../src/features/driver/statusDictionary";
import { DriverMissionStatus } from "../../../../src/features/driver/types";
import {
  AppCard,
  AppSpinner,
  AppText,
  brandSurfaceSoft,
  Screen,
  useAppViewport,
} from "../../../../src/design/responsive";
import { MissionMap } from "../../../../src/features/driver/components/MissionMap";
import { ConfirmCompletionModal } from "../../../../src/features/driver/components/ConfirmCompletionModal";
import { CancelJustificationModal } from "../../../../src/features/driver/components/CancelJustificationModal";
import { getMissionClientDisplayName } from "../../../../src/features/driver/domain/missionDisplay";
import { StatusSwitch } from "../../../../src/features/driver/components/StatusSwitch";
import { DriverStateBanners } from "../../../../src/features/driver/components/DriverStateBanners";
import { useMissionLayout } from "../../../../src/features/driver/hooks/useMissionLayout";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../../src/features/driver/navigation/DriverFloatingTabBar";
import { FONT_SIZE } from "../../../../src/design/responsive/typographyTokens";

export default function DriverMissionDetailScreen() {
  const router = useRouter();
  const { width } = useAppViewport();
  const isCompactMobile = width < 380;
  const { missionId } = useLocalSearchParams<{ missionId: string }>();
  const missionIdNumber = Number(missionId);
  useActiveMissionScreenTracking(Number.isFinite(missionIdNumber) ? missionIdNumber : null);
  const missionQuery = useDriverMissionDetailQuery(
    Number.isFinite(missionIdNumber) ? missionIdNumber : null
  );
  const etaQuery = useDynamicEtaQuery(Number.isFinite(missionIdNumber) ? missionIdNumber : null, {
    missionStatus: mission?.status ?? null,
  });
  const transition = useDriverStatusTransition();
  const mission = missionQuery.data;
  // NB: le tracking GPS est piloté au niveau du `_layout.tsx` driver
  // via `DriverTrackingHost` pour qu'il reste actif quel que soit
  // l'écran ouvert. Pas besoin de le re-déclencher ici.
  useDriverMissionDetailOpenResync(Number.isFinite(missionIdNumber) ? missionIdNumber : null);

  const statusUx = useMemo(() => getDriverStatusUx(mission?.status as string), [mission?.status]);
  const missionLayout = useMissionLayout();
  const [confirmCompletionOpen, setConfirmCompletionOpen] = useState(false);
  const [cancelOpen, setCancelOpen] = useState(false);
  const invalidMissionId = !Number.isFinite(missionIdNumber);
  const missionError = missionQuery.error as { status?: number; message?: string } | undefined;
  const missionNotFound = missionQuery.isError && missionError?.status === 404;
  const missionUnavailable = invalidMissionId || missionNotFound;

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:update_status">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[
            styles.page,
            {
              paddingHorizontal: missionLayout.horizontalPadding,
            },
            isCompactMobile && styles.pageCompact,
          ]}
        >
          <View style={[styles.content, { width: missionLayout.contentWidth }]}>
          <AppText variant="sectionTitle" style={isCompactMobile ? styles.titleCompact : undefined}>Mission detail</AppText>
          <DriverStateBanners />
          {missionUnavailable ? (
            <AppCard style={[styles.detailsCard, isCompactMobile && styles.detailsCardCompact]}>
              <AppText variant="label">Mission indisponible</AppText>
              <AppText variant="bodyMuted">
                {invalidMissionId
                  ? "Identifiant mission invalide."
                  : "Cette mission n'est plus accessible (annulee, terminee ou reattribuee)."}
              </AppText>
              <Pressable
                style={({ pressed }) => [styles.backButton, pressed && styles.backButtonPressed]}
                onPress={() => router.replace("/(app)/(driver)/missions")}
              >
                <AppText variant="label" style={styles.backButtonText}>
                  Retour aux missions
                </AppText>
              </Pressable>
            </AppCard>
          ) : null}
          {!missionUnavailable && missionQuery.isLoading ? <AppSpinner size="small" /> : null}
          {!missionUnavailable && missionQuery.isError ? (
            <AppText variant="error">
              Impossible de charger le detail mission: {(missionQuery.error as Error)?.message ?? "Erreur"}
            </AppText>
          ) : null}
          {!missionUnavailable && mission ? (
            <AppCard style={[styles.detailsCard, isCompactMobile && styles.detailsCardCompact]}>
              <AppText variant="label">Mission #{mission.id}</AppText>
              <AppText variant="body">Statut: {statusUx.label}</AppText>
              <AppText variant="body" style={styles.routeLine}>
                {(mission.pickup_location as string | undefined) ?? "Depart"}
                {" -> "}
                {(mission.dropoff_location as string | undefined) ?? "Arrivee"}
              </AppText>
              <AppText variant="body">
                Planifie: {(mission.scheduled_time as string | undefined) ?? "N/A"}
              </AppText>
              <AppText variant="body">
                ETA dynamique:{" "}
                {etaQuery.data?.eta_minutes != null ? `${etaQuery.data.eta_minutes} min` : "indisponible"}
              </AppText>
            </AppCard>
          ) : null}

          {!missionUnavailable && mission ? (
            <View style={styles.mapWrap}>
              <MissionMap
                mission={mission}
                height={missionLayout.mapHeight}
                etaMinutes={etaQuery.data?.eta_minutes ?? null}
              />
            </View>
          ) : null}

          {!missionUnavailable && mission ? (
            <StatusSwitch
              mode="mission"
              missionStatus={mission.status as DriverMissionStatus}
              pending={transition.isPending}
              onTransition={(nextStatus) => {
                if (!mission?.id) return;
                if (nextStatus === "COMPLETED") {
                  setConfirmCompletionOpen(true);
                  return;
                }
                if (nextStatus === "CANCELLED") {
                  setCancelOpen(true);
                  return;
                }
                transition.mutate({ missionId: mission.id, targetStatus: nextStatus });
              }}
            />
          ) : null}

          {!missionUnavailable && transition.isPending ? (
            <AppText variant="bodyMuted">Transition en cours...</AppText>
          ) : null}
          </View>
        </Screen>
        <ConfirmCompletionModal
          visible={confirmCompletionOpen}
          missionId={mission?.id ?? null}
          clientLabel={mission ? getMissionClientDisplayName(mission) : null}
          pending={transition.isPending}
          onCancel={() => setConfirmCompletionOpen(false)}
          onConfirm={() => {
            if (!mission?.id) return;
            transition.mutate({ missionId: mission.id, targetStatus: "COMPLETED" });
            setConfirmCompletionOpen(false);
          }}
        />
        <CancelJustificationModal
          visible={cancelOpen}
          pending={transition.isPending}
          onCancel={() => setCancelOpen(false)}
          onConfirm={(reason) => {
            if (!mission?.id) return;
            transition.mutate({ missionId: mission.id, targetStatus: "CANCELLED", reason });
            setCancelOpen(false);
          }}
        />
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingVertical: 20,
    gap: 12,
    alignItems: "center",
    paddingBottom: 32,
  },
  pageCompact: {
    paddingVertical: 14,
    gap: 10,
    paddingBottom: 28,
  },
  content: {
    gap: 12,
  },
  titleCompact: {
    fontSize: FONT_SIZE.px20,
    lineHeight: 25,
  },
  detailsCard: {
    gap: 6,
  },
  detailsCardCompact: {
    padding: 12,
    gap: 4,
  },
  routeLine: {
    lineHeight: 20,
  },
  mapWrap: {
    borderRadius: 12,
    overflow: "hidden",
  },
  backButton: {
    marginTop: 8,
    minHeight: 40,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(30, 58, 138, 0.28)",
    backgroundColor: "rgba(37, 99, 235, 0.08)",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
  },
  backButtonPressed: {
    opacity: 0.86,
  },
  backButtonText: {
    color: "#1D4ED8",
    fontWeight: "700",
  },
});

