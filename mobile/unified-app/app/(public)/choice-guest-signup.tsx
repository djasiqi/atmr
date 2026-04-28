import { Pressable, ScrollView, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function ChoiceGuestOrSignupScreen() {
  const router = useRouter();

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 24, justifyContent: "center", gap: 14 }}>
      <Text style={{ fontSize: 26, fontWeight: "800", color: "#0f172a" }}>
        Comment souhaitez-vous continuer ?
      </Text>
      <Text style={{ color: "#334155", lineHeight: 22 }}>
        Un formulaire pour le trajet et un telephone de confirmation, sans creer de profil. Paiement a
        l&apos;etape suivante.
      </Text>

      <View style={{ borderWidth: 1, borderColor: "#dbeafe", backgroundColor: "#eff6ff", borderRadius: 12, padding: 12, gap: 6 }}>
        <Text style={{ fontWeight: "700", color: "#0f172a" }}>Sans compte</Text>
        <Text style={{ color: "#1e293b" }}>
          Le parcours le plus court : tout sur une page, puis identification pour valider la reservation.
        </Text>
      </View>

      <Pressable
        onPress={() => router.push("/(public)/pre-request/step-1" as any)}
        style={{ backgroundColor: "#0a7ea4", borderRadius: 10, padding: 14, alignItems: "center" }}
      >
        <Text style={{ color: "#fff", fontWeight: "700" }}>Continuer sans compte</Text>
      </Pressable>

      <View style={{ borderWidth: 1, borderColor: "#e2e8f0", borderRadius: 12, padding: 12, gap: 6 }}>
        <Text style={{ fontWeight: "700", color: "#0f172a" }}>Creer un compte</Text>
        <Text style={{ color: "#334155" }}>
          Retrouvez vos reservations, vos preferences et votre historique.
        </Text>
      </View>
      <Pressable
        onPress={() => router.push("/(public)/signup" as any)}
        style={{ borderWidth: 1, borderColor: "#0a7ea4", borderRadius: 10, padding: 14, alignItems: "center", backgroundColor: "#fff" }}
      >
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Creer un compte</Text>
      </Pressable>
      <Pressable onPress={() => router.push("/(public)/login" as any)}>
        <Text style={{ color: "#0a7ea4", textAlign: "center", fontWeight: "600" }}>J&apos;ai deja un compte</Text>
      </Pressable>
    </ScrollView>
  );
}
