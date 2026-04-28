import { Pressable, ScrollView, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function ConditionsScreen() {
  const router = useRouter();

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 24, gap: 14 }}>
      <Pressable onPress={() => router.back()} style={{ alignSelf: "flex-start", paddingVertical: 6 }}>
        <Text style={{ color: "#0A8F7A", fontWeight: "700" }}>Retour</Text>
      </Pressable>

      <Text style={{ fontSize: 28, lineHeight: 34, color: "#163A34", fontWeight: "700" }}>
        Conditions d&apos;utilisation
      </Text>

      <View style={{ gap: 10 }}>
        <Text style={{ color: "#45655D", lineHeight: 22 }}>
          En utilisant LIRIE, vous acceptez de fournir des informations exactes pour organiser vos
          réservations.
        </Text>
        <Text style={{ color: "#45655D", lineHeight: 22 }}>
          L&apos;utilisation du service doit rester conforme au cadre légal et aux règles de sécurité
          applicables au transport médical.
        </Text>
        <Text style={{ color: "#45655D", lineHeight: 22 }}>
          Ces conditions peuvent évoluer; la version en vigueur est celle affichée au moment de votre
          utilisation.
        </Text>
      </View>
    </ScrollView>
  );
}
