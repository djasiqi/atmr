import { useMemo, useState } from "react";
import { ScrollView, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import { useDriverMissionsQuery } from "../../../src/features/driver/hooks";
import { DriverCompletedTrip, getDriverCompletedTrips } from "../../../src/features/driver/api";
import { Button } from "../../../src/components/ui";
import { MissionCard } from "../../../src/features/driver/components/MissionCard";

export default function DriverTripsScreen() {
  const router = useRouter();
  const { activeContext } = useSession();
  const missionsQuery = useDriverMissionsQuery();
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
        <ScrollView contentContainerStyle={{ padding: 20, gap: 10 }}>
          <Text style={{ fontSize: 22, fontWeight: "700" }}>Courses</Text>
          <View style={{ flexDirection: "row", gap: 8 }}>
            <Button
              label="Missions actives"
              variant={mode === "active" ? "primary" : "secondary"}
              onPress={() => setMode("active")}
            />
            <Button
              label={historyPending ? "Chargement..." : "Historique operations"}
              variant={mode === "history" ? "primary" : "secondary"}
              disabled={historyPending}
              onPress={loadHistory}
            />
          </View>

          {missionsQuery.isLoading ? <Text>Chargement des courses...</Text> : null}
          {missionsQuery.error ? (
            <Text style={{ color: "#B00020" }}>
              {missionsQuery.error instanceof Error ? missionsQuery.error.message : "Erreur chargement courses."}
            </Text>
          ) : null}
          {historyError ? <Text style={{ color: "#B00020" }}>{historyError}</Text> : null}

          {mode === "active" &&
            activeMissions.map((mission) => {
              return (
                <View
                  key={mission.id}
                  style={{ gap: 6 }}
                >
                  <MissionCard
                    mission={mission}
                    onOpen={(missionId) =>
                      router.push({
                        pathname: "/(app)/(driver)/missions/[missionId]",
                        params: { missionId: String(missionId) },
                      })
                    }
                  />
                  <Button
                    label="Voir details course"
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
                <View key={id} style={{ padding: 12, borderWidth: 1, borderColor: "#E3E3E3", borderRadius: 8, gap: 4 }}>
                  <Text style={{ fontWeight: "700" }}>Course #{id}</Text>
                  <Text>Pickup: {trip.pickup_location ?? "N/A"}</Text>
                  <Text>Dropoff: {trip.dropoff_location ?? "N/A"}</Text>
                  <Text>Status: {trip.status ?? "N/A"}</Text>
                  <Button
                    label="Voir details course"
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
            <Text style={{ color: "#666" }}>Aucune course trouvee.</Text>
          ) : null}
          {mode === "history" && !historyPending && historyTrips.length === 0 ? (
            <Text style={{ color: "#666" }}>Historique vide ou indisponible.</Text>
          ) : null}
        </ScrollView>
      </PermissionGuard>
    </DriverContextGuard>
  );
}
