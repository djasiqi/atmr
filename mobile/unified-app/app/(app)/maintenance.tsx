import { Text, View } from "react-native";

export default function MaintenanceScreen() {
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24 }}>
      <Text style={{ fontSize: 20, fontWeight: "700" }}>Maintenance</Text>
      <Text>Service temporairement indisponible.</Text>
    </View>
  );
}
