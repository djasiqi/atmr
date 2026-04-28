import { Text, View } from "react-native";

export default function OnboardingScreen() {
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24 }}>
      <Text style={{ fontSize: 20, fontWeight: "700" }}>Onboarding requis</Text>
      <Text>Completer le parcours onboarding avant acces aux espaces metiers.</Text>
    </View>
  );
}
