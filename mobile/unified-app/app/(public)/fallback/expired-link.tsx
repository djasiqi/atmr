import { Pressable, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function ExpiredLinkScreen() {
  const router = useRouter();
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "800", color: "#7f1d1d" }}>Lien expire</Text>
      <Text style={{ color: "#475569" }}>
        Ce lien n&apos;est plus valide. Demandez un nouveau lien depuis l&apos;application.
      </Text>
      <Pressable onPress={() => router.replace("/(public)/login" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Se connecter</Text>
      </Pressable>
      <Pressable onPress={() => router.replace("/(public)/help" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Voir l&apos;aide</Text>
      </Pressable>
    </View>
  );
}
