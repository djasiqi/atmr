import { StyleSheet } from "react-native";
import { usePerfScreenReady } from "../../../src/core/observability/usePerfScreenReady";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import { useDriverMissionsQuery } from "../../../src/features/driver/hooks";
import { getDriverStatusUx } from "../../../src/features/driver/statusDictionary";
import {
  AppButton,
  AppCard,
  AppSpinner,
  AppText,
  brandSurfaceSoft,
  Screen,
  useAppViewport,
} from "../../../src/design/responsive";
import { groupMissionsByPickupWindow } from "../../../src/features/driver/domain/missionGrouping";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import { FONT_SIZE } from "../../../src/design/responsive/typographyTokens";
import { DriverTrackingBanner } from "../../../src/features/driver/components/DriverTrackingBanner";
import { DriverTrackingQaPanel } from "../../../src/features/driver/components/DriverTrackingQaPanel";
import {
  isTrackingQaPanelEnabled,
  useDriverBackgroundTrackingUi,
} from "../../../src/features/driver/hooks/useDriverBackgroundTrackingUi";
import * as Location from "expo-location";

export default function DriverMissionsScreen() {
  const router = useRouter();
  const { width } = useAppViewport();
  const isCompactMobile = width < 380;
  const missionsQuery = useDriverMissionsQuery();
  const trackingUi = useDriverBackgroundTrackingUi();
  const handleRequestBgPermission = async () => {
    await Location.requestBackgroundPermissionsAsync().catch(() => undefined);
  };
  usePerfScreenReady(
    "driver.missions",
    "driver.missions.data_ready",
    missionsQuery.isSuccess || missionsQuery.isError
  );

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[styles.page, isCompactMobile && styles.pageCompact]}
        >
          <AppText variant="sectionTitle" style={[styles.title, isCompactMobile && styles.titleCompact]}>
            Missions chauffeur
          </AppText>
          {isTrackingQaPanelEnabled() ? <DriverTrackingQaPanel ui={trackingUi} /> : null}
          <DriverTrackingBanner ui={trackingUi} onRequestPermission={handleRequestBgPermission} />
          {missionsQuery.isLoading ? <AppSpinner size="small" /> : null}
          {missionsQuery.isError ? (
            <AppText variant="error" style={styles.error}>
              Impossible de charger les missions : {(missionsQuery.error as Error)?.message ?? "Erreur"}
            </AppText>
          ) : null}
          {!missionsQuery.isLoading && !missionsQuery.isError && (missionsQuery.data?.length ?? 0) === 0 ? (
            <AppCard variant="surface" style={styles.emptyCard}>
              <AppText variant="label" style={styles.groupTitle}>
                Aucune mission disponible
              </AppText>
              <AppText variant="bodyMuted" style={styles.muted}>
                Les nouvelles missions apparaîtront automatiquement ici.
              </AppText>
            </AppCard>
          ) : null}

          {groupMissionsByPickupWindow(missionsQuery.data ?? []).map((group) => (
            <AppCard key={group.id} variant="surface" style={isCompactMobile ? styles.groupCardCompact : undefined}>
              <AppText variant="label" style={[styles.groupTitle, isCompactMobile && styles.groupTitleCompact]}>
                {group.isGrouped ? `Groupe ${group.missions.length} missions` : "Mission"}
              </AppText>
              <AppText variant="bodyMuted" style={[styles.muted, styles.metaLine]}>
                Départ : {group.displayLabel}
              </AppText>
              {group.missions.map((mission, index) => {
                const ux = getDriverStatusUx(mission.status as string);
                return (
                  <AppCard
                    key={mission.id}
                    variant="surface"
                    style={[
                      index === 0 ? styles.missionCardFirst : styles.missionCard,
                      isCompactMobile && styles.missionCardCompact,
                    ]}
                  >
                    <AppText variant="label" style={[styles.missionTitle, isCompactMobile && styles.missionTitleCompact]}>
                      Mission #{mission.id}
                    </AppText>
                    <AppText variant="body" style={styles.body}>
                      {ux.label}
                    </AppText>
                    <AppText variant="body" style={styles.body}>
                      {(mission.pickup_location as string | undefined) ?? "Départ"}
                      {" → "}
                      {(mission.dropoff_location as string | undefined) ?? "Arrivée"}
                    </AppText>
                    <AppButton
                      title="Ouvrir mission"
                      variant="secondary"
                      onPress={() =>
                        router.push({
                          pathname: "/(app)/(driver)/missions/[missionId]",
                          params: { missionId: String(mission.id) },
                        })
                      }
                    />
                  </AppCard>
                );
              })}
            </AppCard>
          ))}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    padding: 20,
    gap: 12,
    paddingBottom: 28,
  },
  pageCompact: {
    padding: 14,
    gap: 10,
    paddingBottom: 24,
  },
  title: {
    color: "#0f172a",
  },
  titleCompact: {
    fontSize: FONT_SIZE.px20,
    lineHeight: 25,
  },
  groupCardCompact: {
    padding: 12,
  },
  groupTitle: {
    fontWeight: "700",
    color: "#0f172a",
  },
  groupTitleCompact: {
    fontSize: FONT_SIZE.px14,
    lineHeight: 18,
  },
  missionTitle: {
    fontWeight: "700",
    color: "#0f172a",
  },
  missionTitleCompact: {
    fontSize: FONT_SIZE.px14,
    lineHeight: 18,
  },
  body: {
    color: "#334155",
    lineHeight: 20,
  },
  muted: {
    color: "#64748b",
  },
  metaLine: {
    lineHeight: 19,
  },
  error: {
    color: "#B42318",
  },
  emptyCard: {
    paddingVertical: 16,
  },
  missionCardFirst: {
    marginTop: 8,
  },
  missionCard: {
    marginTop: 10,
  },
  missionCardCompact: {
    padding: 12,
  },
});
