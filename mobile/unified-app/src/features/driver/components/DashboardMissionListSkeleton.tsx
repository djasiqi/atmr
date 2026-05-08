import { StyleSheet, View } from "react-native";
import { E } from "../../company/theme/enterpriseOpsTheme";

/** Aligné cartes dashboard entreprise : rayon 16, bordure `E.BORDER`. */
export function DashboardMissionListSkeleton() {
  return (
    <View style={styles.stack}>
      {[0, 1, 2].map((item) => (
        <View key={item} style={styles.row}>
          <View style={{ height: 14, backgroundColor: "#E2E8F0", borderRadius: 6, width: "55%" }} />
          <View style={{ height: 12, backgroundColor: "#EEF2F6", borderRadius: 6, width: "80%" }} />
          <View style={{ height: 12, backgroundColor: "#EEF2F6", borderRadius: 6, width: "65%" }} />
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  stack: { gap: 14 },
  row: {
    borderWidth: 1,
    borderColor: E.BORDER,
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 14,
    gap: 8,
    backgroundColor: E.CARD,
  },
});
