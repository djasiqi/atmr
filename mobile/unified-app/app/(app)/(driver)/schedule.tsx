import { useMemo, useState } from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import { InputField } from "../../../src/components/ui";
import { useDriverMissionsQuery } from "../../../src/features/driver/hooks";
import type { DriverMission } from "../../../src/features/driver/types";

function byDate(missions: DriverMission[], selectedDate: string) {
  return missions.filter((mission) => {
    const raw = mission.scheduled_time;
    if (typeof raw !== "string" || raw.length < 10) return false;
    return raw.slice(0, 10) === selectedDate;
  });
}

export default function DriverScheduleScreen() {
  const router = useRouter();
  const today = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const [selectedDate, setSelectedDate] = useState(today);
  const missionsQuery = useDriverMissionsQuery();
  const filtered = byDate((missionsQuery.data as DriverMission[] | undefined) ?? [], selectedDate);

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <ScrollView contentContainerStyle={{ padding: 24, gap: 10 }}>
          <Text style={{ fontSize: 22, fontWeight: "700" }}>Planning chauffeur</Text>
          <InputField
            value={selectedDate}
            onChangeText={setSelectedDate}
            placeholder="YYYY-MM-DD"
          />
          <Text style={{ color: "#666" }}>
            Format attendu: YYYY-MM-DD. Exemple: {today}
          </Text>
          {missionsQuery.isLoading ? <Text>Chargement du planning...</Text> : null}
          {filtered.map((mission) => (
            <View
              key={String(mission.id)}
              style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 4 }}
            >
              <Text style={{ fontWeight: "700" }}>Mission #{mission.id}</Text>
              <Text>Statut: {mission.status}</Text>
              <Text>Heure: {mission.scheduled_time ?? "n/a"}</Text>
              <Text>
                {(mission.pickup_location as string | undefined) ?? "Depart"}
                {" -> "}
                {(mission.dropoff_location as string | undefined) ?? "Arrivee"}
              </Text>
              <Pressable
                onPress={() =>
                  router.push({
                    pathname: "/(app)/(driver)/missions/[missionId]",
                    params: { missionId: String(mission.id) },
                  })
                }
              >
                <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Voir le detail</Text>
              </Pressable>
            </View>
          ))}
          {!missionsQuery.isLoading && filtered.length === 0 ? (
            <Text style={{ color: "#666" }}>Aucune mission planifiee pour cette date.</Text>
          ) : null}
        </ScrollView>
      </PermissionGuard>
    </DriverContextGuard>
  );
}
