import React from "react";
import { View, StyleSheet } from "react-native";

type Props = {
  count?: number;
  horizontalPadding: number;
};

/**
 * Placeholders liste mission (Sprint 1 STW-02 / Sprint 2 UX-01).
 */
export function MissionListSkeleton({
  count = 3,
  horizontalPadding,
}: Props) {
  return (
    <View style={[styles.wrap, { paddingHorizontal: horizontalPadding, paddingTop: 4 }]}>
      {Array.from({ length: count }).map((_, i) => (
        <View key={i} style={styles.card}>
          <View style={styles.row}>
            <View style={styles.pill} />
            <View style={styles.pillShort} />
          </View>
          <View style={styles.line} />
          <View style={styles.lineMedium} />
          <View style={styles.rowActions}>
            <View style={[styles.btn, styles.btnLeft]} />
            <View style={styles.btn} />
          </View>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    paddingBottom: 24,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
    elevation: 2,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  pill: {
    height: 14,
    width: "45%",
    borderRadius: 6,
    backgroundColor: "#e8ecf2",
  },
  pillShort: {
    height: 14,
    width: "22%",
    borderRadius: 6,
    backgroundColor: "#e8ecf2",
  },
  line: {
    height: 12,
    width: "100%",
    borderRadius: 4,
    backgroundColor: "#eef1f5",
    marginBottom: 8,
  },
  lineMedium: {
    height: 12,
    width: "70%",
    borderRadius: 4,
    backgroundColor: "#eef1f5",
    marginBottom: 14,
  },
  rowActions: {
    flexDirection: "row",
  },
  btn: {
    flex: 1,
    height: 36,
    borderRadius: 8,
    backgroundColor: "#e8ecf2",
  },
  btnLeft: {
    marginRight: 10,
  },
});
