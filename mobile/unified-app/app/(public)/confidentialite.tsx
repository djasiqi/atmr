import { Pressable, ScrollView, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function ConfidentialiteScreen() {
  const router = useRouter();

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 24, gap: 14 }}>
      <Pressable onPress={() => router.back()} style={{ alignSelf: "flex-start", paddingVertical: 6 }}>
        <Text style={{ color: "#0A8F7A", fontWeight: "700" }}>Retour</Text>
      </Pressable>

      <Text style={{ fontSize: 28, lineHeight: 34, color: "#163A34", fontWeight: "700" }}>
        Politique de confidentialité
      </Text>

      <View style={{ gap: 10 }}>
        <Text style={{ color: "#45655D", lineHeight: 22 }}>
          Les données personnelles sont utilisées uniquement pour gérer votre compte, vos réservations
          et les notifications associées.
        </Text>
        <Text style={{ color: "#45655D", lineHeight: 22 }}>
          Nous appliquons des mesures de sécurité adaptées pour protéger vos informations de contact
          et d&apos;accès.
        </Text>
        <Text style={{ color: "#45655D", lineHeight: 22 }}>
          Vous pouvez demander la consultation, la correction ou la suppression de vos données selon
          les règles en vigueur.
        </Text>
      </View>
    </ScrollView>
  );
}
