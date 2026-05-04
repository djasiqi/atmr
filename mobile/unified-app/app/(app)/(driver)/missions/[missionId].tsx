import { useEffect, useMemo, useRef, useState } from "react";
import { Alert, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import {
  useDriverMissionDetailQuery,
  useDriverMissionDetailOpenResync,
  useDriverStatusTransition,
  useDriverTracking,
  useDynamicEtaQuery,
} from "../../../../src/features/driver/hooks";
import { getDriverStatusUx } from "../../../../src/features/driver/statusDictionary";
import { DriverMissionStatus } from "../../../../src/features/driver/types";
import { AppCard, AppSpinner, AppText, brandSurfaceSoft, Screen } from "../../../../src/design/responsive";
import { MissionMap } from "../../../../src/features/driver/components/MissionMap";
import { ConfirmCompletionModal } from "../../../../src/features/driver/components/ConfirmCompletionModal";
import { CancelJustificationModal } from "../../../../src/features/driver/components/CancelJustificationModal";
import { StatusSwitch } from "../../../../src/features/driver/components/StatusSwitch";
import { DriverStateBanners } from "../../../../src/features/driver/components/DriverStateBanners";
import { useMissionLayout } from "../../../../src/features/driver/hooks/useMissionLayout";

export default function DriverMissionDetailScreen() {
  const router = useRouter();
  const { missionId } = useLocalSearchParams<{ missionId: string }>();
  const missionIdNumber = Number(missionId);
  const hasHandledFallbackRef = useRef(false);
  const missionQuery = useDriverMissionDetailQuery(
    Number.isFinite(missionIdNumber) ? missionIdNumber : null
  );
  const etaQuery = useDynamicEtaQuery(Number.isFinite(missionIdNumber) ? missionIdNumber : null);
  const transition = useDriverStatusTransition();
  const mission = missionQuery.data;
  useDriverTracking(
    mission?.id ?? null,
    typeof mission?.status === "string" ? mission.status : null
  );
  useDriverMissionDetailOpenResync(Number.isFinite(missionIdNumber) ? missionIdNumber : null);

  const statusUx = useMemo(() => getDriverStatusUx(mission?.status as string), [mission?.status]);
  const missionLayout = useMissionLayout();
  const [confirmCompletionOpen, setConfirmCompletionOpen] = useState(false);
  const [cancelOpen, setCancelOpen] = useState(false);

  useEffect(() => {
    if (hasHandledFallbackRef.current) return;
    if (!Number.isFinite(missionIdNumber)) {
      hasHandledFallbackRef.current = true;
      Alert.alert("Mission indisponible", "Identifiant mission invalide.", [
        {
          text: "Retour aux missions",
          onPress: () => router.replace("/(app)/(driver)/missions"),
        },
      ]);
      return;
    }
    if (!missionQuery.isError) return;
    const error = missionQuery.error as { status?: number; message?: string } | undefined;
    const status = typeof error?.status === "number" ? error.status : null;
    if (status !== 404) return;
    hasHandledFallbackRef.current = true;
    Alert.alert(
      "Mission indisponible",
      "Cette mission n'est plus accessible (annulee, terminee ou reattribuee).",
      [
        {
          text: "Retour aux missions",
          onPress: () => router.replace("/(app)/(driver)/missions"),
        },
      ]
    );
  }, [missionIdNumber, missionQuery.isError, missionQuery.error, router]);

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:update_status">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          contentContainerStyle={{
            paddingHorizontal: missionLayout.horizontalPadding,
            paddingVertical: 20,
            gap: 12,
            alignItems: "center",
            paddingBottom: 32,
          }}
        >
          <View style={{ width: missionLayout.contentWidth, gap: 12 }}>
          <AppText variant="sectionTitle">Mission detail</AppText>
          <DriverStateBanners />
          {missionQuery.isLoading ? <AppSpinner size="small" /> : null}
          {missionQuery.isError ? (
            <AppText variant="error">
              Impossible de charger le detail mission: {(missionQuery.error as Error)?.message ?? "Erreur"}
            </AppText>
          ) : null}
          {mission ? (
            <AppCard>
              <AppText variant="label">Mission #{mission.id}</AppText>
              <AppText variant="body">Statut: {statusUx.label}</AppText>
              <AppText variant="body">
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

          {mission ? (
            <MissionMap
              pickupLat={mission.pickup_lat as number | null | undefined}
              pickupLng={mission.pickup_lng as number | null | undefined}
              dropoffLat={mission.dropoff_lat as number | null | undefined}
              dropoffLng={mission.dropoff_lng as number | null | undefined}
              height={missionLayout.mapHeight}
            />
          ) : null}

          {mission ? (
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

          {transition.isPending ? (
            <AppText variant="bodyMuted">Transition en cours...</AppText>
          ) : null}
          </View>
        </Screen>
        <ConfirmCompletionModal
          visible={confirmCompletionOpen}
          missionId={mission?.id ?? null}
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

