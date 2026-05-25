import { Platform, StyleSheet, View } from "react-native";
import type { FleetMissionAnchorStyle } from "./fleetMapMissionVisual";

type Props = {
  anchor: FleetMissionAnchorStyle;
  selected?: boolean;
};

export function MissionMapAnchor({ anchor, selected = false }: Props) {
  const size = anchor.radius * 2 + (selected ? 6 : 4);
  const halo =
    anchor.role === "urgent" || anchor.role === "active"
      ? {
          shadowColor: anchor.fill,
          shadowOpacity: 0.45,
          shadowRadius: selected ? 10 : 6,
          shadowOffset: { width: 0, height: 0 },
          elevation: 8,
        }
      : Platform.select({
          ios: {
            shadowColor: "#0F172A",
            shadowOpacity: 0.12,
            shadowRadius: 4,
            shadowOffset: { width: 0, height: 2 },
          },
          android: { elevation: 3 },
          default: {},
        });

  return (
    <View
      style={[
        styles.wrap,
        { width: size, height: size, opacity: anchor.opacity },
        halo,
      ]}
      accessibilityLabel={
        anchor.role === "pickup"
          ? "Point de prise en charge"
          : anchor.role === "dropoff"
            ? "Destination"
            : "Point mission"
      }
    >
      <View
        style={[
          styles.core,
          {
            width: anchor.radius * 2,
            height: anchor.radius * 2,
            borderRadius: anchor.radius,
            backgroundColor: anchor.fill,
            borderColor: anchor.stroke,
            borderWidth: selected ? 3 : 2,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: "center",
    justifyContent: "center",
  },
  core: {},
});
