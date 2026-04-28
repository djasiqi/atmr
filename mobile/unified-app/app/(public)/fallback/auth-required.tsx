import { Pressable, Text, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";

export default function AuthRequiredScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ next?: string }>();
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "800", color: "#0f172a" }}>
        Connexion requise
      </Text>
      <Text style={{ color: "#475569" }}>
        Connectez-vous pour poursuivre cette action en toute securite.
      </Text>
      <Pressable
        onPress={() =>
          router.replace({
            pathname: "/(public)/login",
            params: params.next ? { next: params.next } : {},
          } as any)
        }
      >
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Se connecter</Text>
      </Pressable>
      <Pressable onPress={() => router.replace("/(public)/signup" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Creer un compte</Text>
      </Pressable>
    </View>
  );
}
