import { ActivityIndicator, View } from "react-native";

export function Loader() {
  return (
    <View style={{ paddingVertical: 12 }}>
      <ActivityIndicator size="small" color="#0a7ea4" />
    </View>
  );
}

