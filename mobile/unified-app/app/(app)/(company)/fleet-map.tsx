import { useCallback, useMemo } from "react";
import { usePerfScreenReady } from "../../../src/core/observability/usePerfScreenReady";
import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { useCompanyDriverLiveTracking } from "../../../src/features/company/realtime/useCompanyDriverLiveTracking";
import { OperationalFleetMap } from "../../../src/features/company/components/maps/OperationalFleetMap";
import { useCompanyDispatchMissionsQuery } from "../../../src/features/company/hooks";
import { AppText, Screen, useAppViewport } from "../../../src/design/responsive";

const C = {
  text: "#163A34",
  pageBg: "#F4F7F9",
  brand: "#14B8A6",
} as const;

export default function CompanyFleetMapScreen() {
  const router = useRouter();
  const { usableHeight } = useAppViewport();
  const live = useCompanyDriverLiveTracking();
  const date = new Date().toISOString().slice(0, 10);
  const missionsQuery = useCompanyDispatchMissionsQuery({ date, status: "all" });
  usePerfScreenReady(
    "company.fleet_map",
    "company.fleet_map.data_ready",
    (missionsQuery.isSuccess || missionsQuery.isError) && !live.isLoading
  );

  const mapHeight = useMemo(() => Math.max(420, Math.min(usableHeight - 96, 640)), [usableHeight]);

  const back = useCallback(() => {
    if (router.canGoBack()) router.back();
    else router.replace("/(app)/(company)/dashboard");
  }, [router]);

  const openUrgent = useCallback(() => {
    router.push({
      pathname: "/(app)/(company)/rides",
      params: { filter: "delayed" },
    } as Parameters<typeof router.push>[0]);
  }, [router]);

  return (
    <PermissionGuard permission="company:dashboard:read">
      <Screen backgroundColor={C.pageBg} withHorizontalPadding={false} scroll={false}>
        <View style={styles.fill}>
          <View style={styles.header}>
            <Pressable
              onPress={back}
              style={({ pressed }) => [styles.backBtn, pressed && { opacity: 0.85 }]}
              hitSlop={8}
            >
              <Ionicons name="chevron-back" size={24} color={C.brand} />
            </Pressable>
            <AppText variant="sectionTitle" style={styles.title}>
              Carte flotte
            </AppText>
          </View>

          <View style={[styles.mapWrap, { height: mapHeight }]}>
            <OperationalFleetMap
              drivers={live.drivers}
              missions={missionsQuery.data?.missions ?? []}
              mapHeight={mapHeight}
              onPressUrgent={openUrgent}
              onViewMission={(missionId) =>
                router.push({
                  pathname: "/(app)/(company)/ride-details",
                  params: { rideId: String(missionId) },
                } as Parameters<typeof router.push>[0])
              }
              onMessage={() =>
                router.push("/(app)/(company)/chat" as Parameters<typeof router.push>[0])
              }
            />
          </View>
        </View>
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 12,
    paddingBottom: 10,
    paddingTop: 4,
  },
  backBtn: { padding: 4 },
  title: { color: C.text, fontWeight: "800", flex: 1 },
  mapWrap: {
    marginHorizontal: 12,
    borderRadius: 22,
    overflow: "hidden",
  },
});
