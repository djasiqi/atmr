import { StyleSheet, View } from "react-native";

export function InboxThreadRowSkeleton() {
  return (
    <View style={styles.row}>
      <View style={styles.avatar} />
      <View style={styles.body}>
        <View style={styles.lineWide} />
        <View style={styles.lineMid} />
        <View style={styles.lineShort} />
      </View>
    </View>
  );
}

export function InboxThreadListSkeleton({ count = 6 }: { count?: number }) {
  return (
    <View>
      {Array.from({ length: count }, (_, i) => (
        <InboxThreadRowSkeleton key={i} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 16,
    gap: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#E5E7EB",
  },
  avatar: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: "#E5E7EB",
  },
  body: { flex: 1, gap: 8 },
  lineWide: { height: 14, borderRadius: 4, backgroundColor: "#E5E7EB", width: "72%" },
  lineMid: { height: 12, borderRadius: 4, backgroundColor: "#F3F4F6", width: "48%" },
  lineShort: { height: 12, borderRadius: 4, backgroundColor: "#F3F4F6", width: "85%" },
});
