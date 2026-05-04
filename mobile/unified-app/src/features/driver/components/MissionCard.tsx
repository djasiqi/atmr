import { Pressable, View } from "react-native";
import { brandPrimary } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
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
      <AppText variant="sectionTitle">Mission #{mission.id}</AppText>
      <AppText variant="body">Statut: {statusUx.label}</AppText>
      <AppText variant="body">
        {(mission.pickup_location as string | undefined) ?? "Depart"}
        {" -> "}
        {(mission.dropoff_location as string | undefined) ?? "Arrivee"}
      </AppText>
      <Pressable onPress={() => onOpen(mission.id)}>
        <AppText variant="label" style={{ color: brandPrimary }}>
          Ouvrir mission
        </AppText>
      </Pressable>
    </View>
  );
}
