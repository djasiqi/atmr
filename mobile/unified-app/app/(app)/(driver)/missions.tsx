import { ScrollView, Text } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import { useDriverMissionsQuery } from "../../../src/features/driver/hooks";
import { getDriverStatusUx } from "../../../src/features/driver/statusDictionary";
import { Button, Card, Loader } from "../../../src/components/ui";
import { groupMissionsByPickupWindow } from "../../../src/features/driver/domain/missionGrouping";

export default function DriverMissionsScreen() {
  const router = useRouter();
  const missionsQuery = useDriverMissionsQuery();

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <ScrollView contentContainerStyle={{ padding: 20, gap: 12 }}>
          <Text style={{ fontSize: 22, fontWeight: "700" }}>Missions chauffeur</Text>
          {missionsQuery.isLoading ? <Loader /> : null}
          {missionsQuery.isError ? (
            <Text>
              Impossible de charger les missions: {(missionsQuery.error as Error)?.message ?? "Erreur"}
            </Text>
          ) : null}

          {groupMissionsByPickupWindow(missionsQuery.data ?? []).map((group) => (
            <Card key={group.id}>
              <Text style={{ fontWeight: "700" }}>
                {group.isGrouped
                  ? `Groupe ${group.missions.length} missions`
                  : "Mission"}
              </Text>
              <Text style={{ color: "#666" }}>Depart: {group.displayLabel}</Text>
              {group.missions.map((mission, index) => {
                const ux = getDriverStatusUx(mission.status as string);
                return (
                  <Card key={mission.id} style={{ marginTop: index === 0 ? 8 : 10 }}>
                    <Text style={{ fontWeight: "700" }}>Mission #{mission.id}</Text>
                    <Text>{ux.label}</Text>
                    <Text>
                      {(mission.pickup_location as string | undefined) ?? "Depart"}
                      {" -> "}
                      {(mission.dropoff_location as string | undefined) ?? "Arrivee"}
                    </Text>
                    <Button
                      label="Ouvrir mission"
                      onPress={() =>
                        router.push({
                          pathname: "/(app)/(driver)/missions/[missionId]",
                          params: { missionId: String(mission.id) },
                        })
                      }
                    />
                  </Card>
                );
              })}
            </Card>
          ))}
        </ScrollView>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

