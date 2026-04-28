import { Text, View } from "react-native";

export default function UnauthorizedScreen() {
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24 }}>
      <Text style={{ fontSize: 20, fontWeight: "700" }}>Acces refuse</Text>
      <Text>Permission insuffisante pour cette action.</Text>
    </View>
  );
}
