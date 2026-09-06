import { StyleSheet, View } from "react-native";
import { D } from "../theme/driverDashboardTheme";
import { DRIVER_DASHBOARD_MISSION_SLOT_MIN } from "./driverDashboardShell";

/** Skeleton à la géométrie de la carte mission (DRIVER-COLD-02). */
export function DashboardMissionSlotSkeleton() {
  return (
    <View style={s.root} accessibilityLabel="Chargement de la mission">
      <View style={s.card}>
        <View style={s.lineWide} />
        <View style={s.lineMid} />
        <View style={s.lineShort} />
        <View style={s.metrics}>
          <View style={s.metric} />
          <View style={s.metric} />
          <View style={s.metric} />
        </View>
        <View style={s.cta} />
      </View>
    </View>
  );
}

const BONE = "#E8EEF0";

const s = StyleSheet.create({
  root: {
    alignSelf: "stretch",
    minHeight: DRIVER_DASHBOARD_MISSION_SLOT_MIN,
  },
  card: {
    backgroundColor: D.cardBg,
    borderRadius: D.cardRadius,
    paddingHorizontal: 16,
    paddingTop: 15,
    paddingBottom: 14,
    gap: 11,
    minHeight: DRIVER_DASHBOARD_MISSION_SLOT_MIN,
  },
  lineWide: {
    height: 16,
    width: "62%",
    borderRadius: 6,
    backgroundColor: BONE,
  },
  lineMid: {
    height: 13,
    width: "78%",
    borderRadius: 6,
    backgroundColor: BONE,
  },
  lineShort: {
    height: 13,
    width: "44%",
    borderRadius: 6,
    backgroundColor: BONE,
  },
  metrics: {
    flexDirection: "row",
    gap: 10,
    paddingTop: 6,
  },
  metric: {
    flex: 1,
    height: 28,
    borderRadius: 8,
    backgroundColor: BONE,
  },
  cta: {
    height: 40,
    borderRadius: 10,
    backgroundColor: BONE,
    marginTop: 4,
  },
});
