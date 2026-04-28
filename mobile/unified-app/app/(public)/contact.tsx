import { Pressable, Text, View } from "react-native";
import * as Linking from "expo-linking";

export default function ContactScreen() {
  return (
    <View style={{ flex: 1, padding: 24, justifyContent: "center", gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "700" }}>Contact support</Text>
      <Text style={{ color: "#444" }}>
        Notre équipe vous répond pour les problèmes de connexion, d’activation de compte et de
        réservation.
      </Text>
      <Text style={{ fontWeight: "700" }}>Etablissements</Text>
      <Text style={{ color: "#444" }}>
        Demandez un acces institution pour la planification patient et le suivi des transports.
      </Text>
      <Text style={{ fontWeight: "700" }}>Transporteurs</Text>
      <Text style={{ color: "#444" }}>
        Rejoignez Lirie pour recevoir des missions et connecter votre flotte.
      </Text>
      <Text style={{ color: "#444" }}>Email: support@lirie.ch</Text>
      <Text style={{ color: "#444" }}>Téléphone: +41 22 000 00 00 (08:00-18:00)</Text>
      <Pressable onPress={() => void Linking.openURL("mailto:support@lirie.ch")}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Envoyer un email</Text>
      </Pressable>
      <Pressable onPress={() => void Linking.openURL("tel:+41220000000")}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Appeler le support</Text>
      </Pressable>
      <Text style={{ color: "#666" }}>
        Si vous n’avez pas accès à l’application, les flux d’activation et de mot de passe restent
        disponibles depuis le web.
      </Text>
    </View>
  );
}
