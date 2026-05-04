import { StyleSheet } from "react-native";
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
} from "../../../src/design/responsive";
import { groupMissionsByPickupWindow } from "../../../src/features/driver/domain/missionGrouping";

export default function DriverMissionsScreen() {
  const router = useRouter();
  const missionsQuery = useDriverMissionsQuery();

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen scroll backgroundColor={brandSurfaceSoft} withHorizontalPadding={false} contentContainerStyle={styles.page}>
          <AppText variant="sectionTitle" style={styles.title}>
            Missions chauffeur
          </AppText>
          {missionsQuery.isLoading ? <AppSpinner size="small" /> : null}
          {missionsQuery.isError ? (
            <AppText variant="error" style={styles.error}>
              Impossible de charger les missions : {(missionsQuery.error as Error)?.message ?? "Erreur"}
            </AppText>
          ) : null}

          {groupMissionsByPickupWindow(missionsQuery.data ?? []).map((group) => (
            <AppCard key={group.id} variant="surface">
              <AppText variant="label" style={styles.groupTitle}>
                {group.isGrouped ? `Groupe ${group.missions.length} missions` : "Mission"}
              </AppText>
              <AppText variant="bodyMuted" style={styles.muted}>
                Départ : {group.displayLabel}
              </AppText>
              {group.missions.map((mission, index) => {
                const ux = getDriverStatusUx(mission.status as string);
                return (
                  <AppCard key={mission.id} variant="surface" style={{ marginTop: index === 0 ? 8 : 10 }}>
                    <AppText variant="label" style={styles.missionTitle}>
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
  title: {
    color: "#0f172a",
  },
  groupTitle: {
    fontWeight: "700",
    color: "#0f172a",
  },
  missionTitle: {
    fontWeight: "700",
    color: "#0f172a",
  },
  body: {
    color: "#334155",
  },
  muted: {
    color: "#64748b",
  },
  error: {
    color: "#B42318",
  },
});
