import { Pressable, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function HelpScreen() {
  const router = useRouter();
  return (
    <View style={{ flex: 1, padding: 24, justifyContent: "center", gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "700" }}>Aide et FAQ operationnelle</Text>
      <Text style={{ color: "#444" }}>
        Retrouvez ici les solutions les plus frequentes avant connexion.
      </Text>
      <Text style={{ fontWeight: "700" }}>Puis-je reserver pour quelqu&apos;un d&apos;autre ?</Text>
      <Text style={{ color: "#334" }}>Oui, depuis la pre-demande, ajoutez les besoins specifiques.</Text>
      <Text style={{ fontWeight: "700" }}>Qui paie le transport ?</Text>
      <Text style={{ color: "#334" }}>
        Selon votre contexte: patient, compagnie, institution ou prise en charge.
      </Text>
      <Text style={{ fontWeight: "700" }}>Combien de temps a l&apos;avance reserver ?</Text>
      <Text style={{ color: "#334" }}>
        Le plus tot possible. Le systeme vous indique la disponibilite lors du check de zone.
      </Text>
      <Text style={{ fontWeight: "700" }}>Je ne peux pas me connecter</Text>
      <Pressable onPress={() => router.push("/(public)/forgot-password" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Réinitialiser mon mot de passe</Text>
      </Pressable>
      <Text style={{ fontWeight: "700" }}>Mon compte n&apos;est pas activé</Text>
      <Pressable onPress={() => router.push("/(public)/login" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>
          Reprendre l&apos;activation depuis la connexion
        </Text>
      </Pressable>
      <Text style={{ fontWeight: "700" }}>Je n&apos;ai pas reçu le SMS ou l&apos;email</Text>
      <Pressable onPress={() => router.push("/(public)/contact" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Contacter le support</Text>
      </Pressable>
      <Pressable onPress={() => router.push("/(public)/how-it-works" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Voir comment ca marche</Text>
      </Pressable>
    </View>
  );
}
