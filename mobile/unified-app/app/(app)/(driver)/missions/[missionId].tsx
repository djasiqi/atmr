import { useMemo } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useActiveMissionScreenTracking } from "../../../../src/core/notifications/useActiveScreenTracking";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import {
  useDriverMissionDetailOpenResync,
  useDriverMissionDetailQuery,
  useDynamicEtaQuery,
} from "../../../../src/features/driver/hooks";
import { driverCoursePickupIsLate } from "../../../../src/features/driver/domain/driverCourseLateness";
import {
  AppCard,
  AppSpinner,
  AppText,
  brandSurfaceSoft,
  Screen,
  useAppBottomInsets,
  useAppViewport,
} from "../../../../src/design/responsive";
import { MissionMap } from "../../../../src/features/driver/components/MissionMap";
import { DriverStateBanners } from "../../../../src/features/driver/components/DriverStateBanners";
import { MissionConsultationSheet } from "../../../../src/features/driver/components/MissionConsultationSheet";
import { useMissionLayout } from "../../../../src/features/driver/hooks/useMissionLayout";
import { FONT_SIZE } from "../../../../src/design/responsive/typographyTokens";
import { ABSENT_TIME } from "../../../../src/features/driver/domain/missionConsultation";

/**
 * Fiche de consultation. Aucune transition et aucune écriture financière.
 * L'exécution reste sur le Dashboard.
 */
export default function DriverMissionDetailScreen() {
  const router = useRouter();
  const { width } = useAppViewport();
  const appBottom = useAppBottomInsets();
  const isCompactMobile = width < 380;
  const { missionId } = useLocalSearchParams<{ missionId: string }>();
  const missionIdNumber = Number(missionId);
  useActiveMissionScreenTracking(Number.isFinite(missionIdNumber) ? missionIdNumber : null);
  const missionQuery = useDriverMissionDetailQuery(
    Number.isFinite(missionIdNumber) ? missionIdNumber : null
  );
  const mission = missionQuery.data;
  const etaQuery = useDynamicEtaQuery(Number.isFinite(missionIdNumber) ? missionIdNumber : null, {
    missionStatus: mission?.status ?? null,
  });
  useDriverMissionDetailOpenResync(Number.isFinite(missionIdNumber) ? missionIdNumber : null);

  const missionLate = useMemo(
    () => (mission ? driverCoursePickupIsLate(mission, Date.now()) : false),
    [mission]
  );
  const missionLayout = useMissionLayout();
  const invalidMissionId = !Number.isFinite(missionIdNumber);
  const missionError = missionQuery.error as { status?: number; code?: string; message?: string } | undefined;
  const missionReassigned =
    missionQuery.isError &&
    (missionError?.status === 403 || missionError?.code === "BOOKING_ASSIGNED_TO_OTHER_DRIVER");
  const missionNotFound = missionQuery.isError && missionError?.status === 404;
  const missionUnavailable = invalidMissionId || missionNotFound || missionReassigned;
  const etaMinutes = etaQuery.data?.eta_minutes;
  const etaLabel =
    typeof etaMinutes === "number" && Number.isFinite(etaMinutes)
      ? `${etaMinutes} min`
      : ABSENT_TIME;

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          automaticallyAdjustKeyboardInsets
          extraScrollBottomPadding={
            appBottom.hasFloatingNavigation ? appBottom.navigationClearance : 0
          }
          contentContainerStyle={[
            styles.page,
            {
              paddingHorizontal: missionLayout.horizontalPadding,
            },
            isCompactMobile && styles.pageCompact,
          ]}
        >
          <View style={[styles.content, { width: missionLayout.contentWidth }]}>
            <AppText variant="sectionTitle" style={isCompactMobile ? styles.titleCompact : undefined}>
              Détails de mission
            </AppText>
            <DriverStateBanners />
            {missionUnavailable ? (
              <AppCard style={styles.detailsCard}>
                <AppText variant="label">Mission indisponible</AppText>
                <AppText variant="bodyMuted">
                  {invalidMissionId
                    ? "Identifiant mission invalide."
                    : missionReassigned
                      ? "Ce transport ne vous est plus assigné."
                      : "Cette mission n'est plus accessible."}
                </AppText>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Retour aux missions"
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
                {"Impossible de charger le détail de la mission. Aucune modification n'a été envoyée."}
              </AppText>
            ) : null}
            {!missionUnavailable && mission ? (
              <MissionConsultationSheet
                mission={mission}
                etaLabel={etaLabel}
                stacked={isCompactMobile}
                late={missionLate}
                map={
                  <View style={styles.mapWrap}>
                    <MissionMap
                      mission={mission}
                      height={missionLayout.mapHeight}
                      etaMinutes={etaQuery.data?.eta_minutes ?? null}
                    />
                  </View>
                }
              />
            ) : null}
          </View>
        </Screen>
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
  mapWrap: {
    borderRadius: 12,
    overflow: "hidden",
  },
  backButton: {
    marginTop: 8,
    minHeight: 48,
    borderRadius: 12,
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
