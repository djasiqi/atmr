import { Pressable, Text, View } from "react-native";
import { getDriverStatusUx } from "../statusDictionary";
import type { DriverMission } from "../types";

type Props = {
  mission: DriverMission;
  onOpen: (missionId: number) => void;
};

export function MissionCard({ mission, onOpen }: Props) {
  const statusUx = getDriverStatusUx(mission.status);

  return (
    <View style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 6 }}>
      <Text style={{ fontWeight: "700" }}>Mission #{mission.id}</Text>
      <Text>Statut: {statusUx.label}</Text>
      <Text>
        {(mission.pickup_location as string | undefined) ?? "Depart"}
        {" -> "}
        {(mission.dropoff_location as string | undefined) ?? "Arrivee"}
      </Text>
      <Pressable onPress={() => onOpen(mission.id)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Ouvrir mission</Text>
      </Pressable>
    </View>
  );
}
