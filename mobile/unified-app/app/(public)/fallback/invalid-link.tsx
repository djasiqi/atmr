import { Pressable, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function InvalidLinkScreen() {
  const router = useRouter();
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "800", color: "#7f1d1d" }}>Lien invalide</Text>
      <Text style={{ color: "#475569" }}>
        Le format du lien est incorrect ou incomplet. Reessayez depuis la source d&apos;origine.
      </Text>
      <Pressable onPress={() => router.replace("/(public)" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Retour a l&apos;accueil</Text>
      </Pressable>
      <Pressable onPress={() => router.replace("/(public)/contact" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Contacter le support</Text>
      </Pressable>
    </View>
  );
}
