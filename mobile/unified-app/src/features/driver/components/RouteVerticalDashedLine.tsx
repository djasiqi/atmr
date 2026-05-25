import { StyleSheet, View } from "react-native";
import { D } from "../theme/driverDashboardTheme";

type Props = {
  height?: number;
};

/** Ligne verticale pointillée (compatible Android). */
export function RouteVerticalDashedLine({ height = 36 }: Props) {
  const dotCount = Math.max(4, Math.floor(height / 5));
  return (
    <View style={[styles.col, { minHeight: height }]} accessibilityElementsHidden>
      {Array.from({ length: dotCount }).map((_, index) => (
        <View key={index} style={styles.dot} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  col: {
    width: 2,
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 3,
  },
  dot: {
    width: 2,
    height: 4,
    borderRadius: 1,
    backgroundColor: D.stepLine,
  },
});
