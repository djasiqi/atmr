import { useMemo, useState } from "react";
import { StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import {
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
} from "../../../src/features/driver/hooks";
import { DriverCompletedTrip, getDriverCompletedTrips } from "../../../src/features/driver/api";
import { AppButton, AppText, brandSurfaceSoft, Screen } from "../../../src/design/responsive";
import { MissionCard } from "../../../src/features/driver/components/MissionCard";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";

export default function DriverTripsScreen() {
  const router = useRouter();
  const { activeContext } = useSession();
  const missionsQuery = useDriverMissionsQuery();
  useDriverMissionsListFocusResync();
  const [historyPending, setHistoryPending] = useState(false);
  const [historyTrips, setHistoryTrips] = useState<DriverCompletedTrip[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [mode, setMode] = useState<"active" | "history">("active");

  const driverId = useMemo(() => {
    const contextId = activeContext?.context_id ?? "";
    if (!contextId.startsWith("driver:")) return null;
    const parsed = Number.parseInt(contextId.slice("driver:".length), 10);
    return Number.isFinite(parsed) ? parsed : null;
  }, [activeContext?.context_id]);

  async function loadHistory() {
    if (!driverId) {
      setHistoryError("Driver ID introuvable dans le contexte actif.");
      return;
    }
    setHistoryPending(true);
    setHistoryError(null);
    try {
      const trips = await getDriverCompletedTrips(driverId);
      setHistoryTrips(trips);
      setMode("history");
    } catch (error) {
      setHistoryError(
        error instanceof Error ? error.message : "Erreur de chargement de l'historique."
      );
    } finally {
      setHistoryPending(false);
    }
  }

  const activeMissions = missionsQuery.data ?? [];

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={styles.page}
        >
          <AppText variant="sectionTitle" style={styles.title}>
            Courses
          </AppText>
          <View style={styles.row}>
            <AppButton
              title="Missions actives"
              variant={mode === "active" ? "primary" : "secondary"}
              onPress={() => setMode("active")}
            />
            <AppButton
              title={historyPending ? "Chargement…" : "Historique opérations"}
              variant={mode === "history" ? "primary" : "secondary"}
              disabled={historyPending}
              onPress={loadHistory}
            />
          </View>

          {missionsQuery.isLoading ? (
            <AppText variant="bodyMuted" style={styles.muted}>
              Chargement des courses…
            </AppText>
          ) : null}
          {missionsQuery.error ? (
            <AppText variant="error" style={styles.error}>
              {missionsQuery.error instanceof Error ? missionsQuery.error.message : "Erreur chargement courses."}
            </AppText>
          ) : null}
          {historyError ? (
            <AppText variant="error" style={styles.error}>
              {historyError}
            </AppText>
          ) : null}

          {mode === "active" &&
            activeMissions.map((mission) => {
              return (
                <View key={mission.id} style={styles.block}>
                  <MissionCard mission={mission} />
                  <AppButton
                    title="Voir détails course"
                    variant="secondary"
                    onPress={() =>
                      router.push({
                        pathname: `/(app)/(driver)/trips/${mission.id}` as any,
                        params: {
                          pickup: String(mission.pickup_location ?? ""),
                          dropoff: String(mission.dropoff_location ?? ""),
                          status: String(mission.status ?? ""),
                          scheduled: String(mission.scheduled_time ?? ""),
                          source: "active",
                        },
                      } as any)
                    }
                  />
                </View>
              );
            })}

          {mode === "history" &&
            historyTrips.map((trip) => {
              const id = String(trip.id);
              return (
                <View key={id} style={styles.historyCard}>
                  <AppText variant="label" style={styles.cardTitle}>
                    Course #{id}
                  </AppText>
                  <AppText variant="body" style={styles.body}>
                    Pickup : {trip.pickup_location ?? "N/A"}
                  </AppText>
                  <AppText variant="body" style={styles.body}>
                    Dropoff : {trip.dropoff_location ?? "N/A"}
                  </AppText>
                  <AppText variant="body" style={styles.body}>
                    Statut : {trip.status ?? "N/A"}
                  </AppText>
                  <AppButton
                    title="Voir détails course"
                    variant="secondary"
                    onPress={() =>
                      router.push({
                        pathname: `/(app)/(driver)/trips/${id}` as any,
                        params: {
                          pickup: String(trip.pickup_location ?? ""),
                          dropoff: String(trip.dropoff_location ?? ""),
                          status: String(trip.status ?? ""),
                          source: "history",
                        },
                      } as any)
                    }
                  />
                </View>
              );
            })}

          {mode === "active" && !missionsQuery.isLoading && activeMissions.length === 0 ? (
            <AppText variant="bodyMuted" style={styles.muted}>
              Aucune course trouvée.
            </AppText>
          ) : null}
          {mode === "history" && !historyPending && historyTrips.length === 0 ? (
            <AppText variant="bodyMuted" style={styles.muted}>
              Historique vide ou indisponible.
            </AppText>
          ) : null}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    padding: 20,
    gap: 10,
    paddingBottom: 28,
  },
  title: {
    color: "#0f172a",
  },
  row: {
    flexDirection: "row",
    gap: 8,
    flexWrap: "wrap",
  },
  block: {
    gap: 6,
  },
  historyCard: {
    padding: 12,
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 8,
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
  error: {
    color: "#B42318",
  },
});
