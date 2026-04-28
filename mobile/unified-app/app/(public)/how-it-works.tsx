import { Pressable, ScrollView, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function HowItWorksScreen() {
  const router = useRouter();
  return (
    <ScrollView contentContainerStyle={{ padding: 24, gap: 14 }}>
      <Text style={{ fontSize: 24, fontWeight: "800", color: "#0f172a" }}>
        Comment ca marche
      </Text>
      <Text style={{ color: "#334155", lineHeight: 22 }}>
        Lirie simplifie l&apos;organisation des transports medicaux en 5 etapes.
      </Text>
      <View style={{ gap: 10 }}>
        <Text style={{ fontWeight: "700" }}>1. Je fais une demande</Text>
        <Text style={{ color: "#475569" }}>
          Depart, destination, date et besoins specifiques. Vous pouvez commencer sans compte.
        </Text>
      </View>
      <View style={{ gap: 10 }}>
        <Text style={{ fontWeight: "700" }}>2. Un transporteur accepte</Text>
        <Text style={{ color: "#475569" }}>
          L&apos;ecosysteme Lirie assigne la demande selon disponibilite et contexte.
        </Text>
      </View>
      <View style={{ gap: 10 }}>
        <Text style={{ fontWeight: "700" }}>3. Vous recevez la confirmation</Text>
        <Text style={{ color: "#475569" }}>
          Notification claire de l&apos;etat: en attente, confirme, en route, termine.
        </Text>
      </View>
      <View style={{ gap: 10 }}>
        <Text style={{ fontWeight: "700" }}>4. Le transport est assure</Text>
        <Text style={{ color: "#475569" }}>
          Chauffeur, compagnie et etablissement restent synchronises selon votre contexte.
        </Text>
      </View>
      <View style={{ gap: 10 }}>
        <Text style={{ fontWeight: "700" }}>5. Paiement ou facturation</Text>
        <Text style={{ color: "#475569" }}>
          Selon votre situation: parcours patient, institutionnel ou prise en charge.
        </Text>
      </View>
      <Pressable
        onPress={() => router.push("/(public)/pre-request/step-1" as any)}
        style={{ marginTop: 8, backgroundColor: "#0a7ea4", borderRadius: 10, padding: 14, alignItems: "center" }}
      >
        <Text style={{ color: "#fff", fontWeight: "700" }}>Demarrer une pre-demande</Text>
      </Pressable>
    </ScrollView>
  );
}
