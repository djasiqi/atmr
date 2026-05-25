import { StyleSheet, View } from "react-native";
import type { DriverMission } from "../types";
import { useDynamicEtaQuery } from "../hooks";
import { resolveDriverStatusForUx } from "../statusDictionary";
import { MissionMap } from "./MissionMap";
import { createShadow } from "../../../styles/shadowStyles";
import { D, missionActiveCardShadow } from "../theme/driverDashboardTheme";

type Props = {
  mission: DriverMission;
  height?: number;
};

const mapShadow = createShadow(missionActiveCardShadow);

/** Carte mission dashboard : GPS live, itinéraire dynamique, badge distance/durée. */
export function DashboardMissionMap({ mission, height }: Props) {
  const statusKey = resolveDriverStatusForUx(mission.status);
  const etaQuery = useDynamicEtaQuery(mission.id, { missionStatus: statusKey });

  return (
    <View style={styles.wrap} accessibilityLabel="Carte trajet mission">
      <MissionMap
        mission={mission}
        height={height}
        showRecenterControl
        etaMinutes={etaQuery.data?.eta_minutes ?? null}
        etaSnapshot={etaQuery.data ?? null}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignSelf: "stretch",
    borderRadius: 16,
    overflow: "hidden",
    borderWidth: 0,
    backgroundColor: D.cardBg,
    ...mapShadow,
  },
});
