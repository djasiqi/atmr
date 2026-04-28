import { View } from "react-native";

export function DashboardMissionListSkeleton() {
  return (
    <View style={{ gap: 8 }}>
      {[0, 1, 2].map((item) => (
        <View
          key={item}
          style={{
            borderWidth: 1,
            borderColor: "#eee",
            borderRadius: 10,
            padding: 12,
            gap: 8,
          }}
        >
          <View style={{ height: 14, backgroundColor: "#ececec", borderRadius: 6, width: "55%" }} />
          <View style={{ height: 12, backgroundColor: "#f1f1f1", borderRadius: 6, width: "80%" }} />
          <View style={{ height: 12, backgroundColor: "#f1f1f1", borderRadius: 6, width: "65%" }} />
        </View>
      ))}
    </View>
  );
}
