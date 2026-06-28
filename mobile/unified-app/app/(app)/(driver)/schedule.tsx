import { useMemo, useState } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  AppInput,
  AppText,
  brandPrimary,
  brandSurfaceSoft,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../src/design/responsive";
import { useDriverMissionsQuery } from "../../../src/features/driver/hooks";
import { MISSION_ROUTE_ARROW } from "../../../src/features/driver/domain/missionDisplay";
import type { DriverMission } from "../../../src/features/driver/types";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";

function byDate(missions: DriverMission[], selectedDate: string) {
  return missions.filter((mission) => {
    const raw = mission.scheduled_time;
    if (typeof raw !== "string" || raw.length < 10) return false;
    return raw.slice(0, 10) === selectedDate;
  });
}

export default function DriverScheduleScreen() {
  const router = useRouter();
  const { horizontalPadding } = useAppViewport();
  const t = useResponsiveTokens();
  const today = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const [selectedDate, setSelectedDate] = useState(today);
  const missionsQuery = useDriverMissionsQuery();
  const filtered = byDate((missionsQuery.data as DriverMission[] | undefined) ?? [], selectedDate);

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[
            styles.page,
            { paddingHorizontal: horizontalPadding, gap: t.spacingSm, paddingBottom: t.spacingMd },
          ]}
        >
          <AppText variant="sectionTitle" style={styles.title}>
            Planning chauffeur
          </AppText>
          <AppInput value={selectedDate} onChangeText={setSelectedDate} placeholder="YYYY-MM-DD" />
          <AppText variant="caption" style={styles.hint}>
            Format attendu : YYYY-MM-DD. Exemple : {today}
          </AppText>
          {missionsQuery.isLoading ? (
            <AppText variant="bodyMuted" style={styles.muted}>
              Chargement du planning…
            </AppText>
          ) : null}
          {filtered.map((mission) => (
            <View key={String(mission.id)} style={styles.card}>
              <AppText variant="label" style={styles.cardTitle}>
                Mission #{mission.id}
              </AppText>
              <AppText variant="body" style={styles.body}>
                Statut : {mission.status}
              </AppText>
              <AppText variant="body" style={styles.body}>
                Heure : {mission.scheduled_time ?? "n/a"}
              </AppText>
              <AppText variant="body" style={styles.body}>
                {(mission.pickup_location as string | undefined) ?? "Départ"}
                {MISSION_ROUTE_ARROW}
                {(mission.dropoff_location as string | undefined) ?? "Arrivée"}
              </AppText>
              <Pressable
                onPress={() =>
                  router.push({
                    pathname: "/(app)/(driver)/missions/[missionId]",
                    params: { missionId: String(mission.id) },
                  })
                }
              >
                <AppText variant="label" style={styles.link}>
                  Voir le détail
                </AppText>
              </Pressable>
            </View>
          ))}
          {!missionsQuery.isLoading && filtered.length === 0 ? (
            <AppText variant="bodyMuted" style={styles.muted}>
              Aucune mission planifiée pour cette date.
            </AppText>
          ) : null}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingTop: 24,
  },
  title: {
    color: "#0f172a",
  },
  hint: {
    color: "#64748b",
  },
  card: {
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 10,
    padding: 12,
    gap: 4,
    backgroundColor: "#fff",
  },
  cardTitle: {
    fontWeight: "700",
    color: "#0f172a",
  },
  body: {
    color: "#334155",
  },
  muted: {
    color: "#64748b",
  },
  link: {
    color: brandPrimary,
    fontWeight: "600",
    marginTop: 4,
  },
});
