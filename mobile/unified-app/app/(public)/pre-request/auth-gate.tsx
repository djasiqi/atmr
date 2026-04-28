import { useMemo } from "react";
import { Pressable, ScrollView, Text } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";

export default function PreRequestAuthGateScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ draftId?: string }>();

  const nextRoute = useMemo(() => {
    const draftId = typeof params.draftId === "string" ? params.draftId.trim() : "";
    if (!draftId) return "/(app)/(client)/booking/new";
    return `/(app)/(client)/booking/new?publicDraftId=${encodeURIComponent(draftId)}`;
  }, [params.draftId]);

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 24, justifyContent: "center", gap: 14 }}>
      <Text style={{ fontSize: 26, fontWeight: "800", color: "#0f172a" }}>
        Paiement et confirmation
      </Text>
      <Text style={{ color: "#334155", lineHeight: 22 }}>
        Votre trajet est deja enregistre. Pour regler et confirmer la reservation, connectez-vous ou creez un
        compte : le parcours reprendra sans ressaisir les adresses.
      </Text>

      <Pressable
        onPress={() =>
          router.replace({
            pathname: "/(public)/login",
            params: { next: nextRoute },
          } as any)
        }
        style={{ backgroundColor: "#0a7ea4", borderRadius: 10, padding: 14, alignItems: "center" }}
      >
        <Text style={{ color: "#fff", fontWeight: "700" }}>Se connecter pour finaliser</Text>
      </Pressable>
      <Pressable
        onPress={() =>
          router.replace({
            pathname: "/(public)/signup",
            params: { next: nextRoute },
          } as any)
        }
        style={{ borderWidth: 1, borderColor: "#0a7ea4", borderRadius: 10, padding: 14, alignItems: "center" }}
      >
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Creer un compte pour finaliser</Text>
      </Pressable>
    </ScrollView>
  );
}
