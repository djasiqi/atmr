import { Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../src/design/responsive";

export default function HelpScreen() {
  const router = useRouter();
  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Aide et FAQ</Text>
          <Text style={styles.intro}>
            Retrouvez ici les réponses les plus fréquentes avant connexion.
          </Text>

          <Text style={styles.question}>Puis-je réserver pour quelqu&apos;un d&apos;autre ?</Text>
          <Text style={styles.answer}>
            Oui : depuis la pré-demande, ajoutez les besoins spécifiques dans les champs prévus.
          </Text>

          <Text style={styles.question}>Qui paie le transport ?</Text>
          <Text style={styles.answer}>
            Selon votre contexte : patient, compagnie, institution ou prise en charge.
          </Text>

          <Text style={styles.question}>Combien de temps à l&apos;avance réserver ?</Text>
          <Text style={styles.answer}>
            Le plus tôt possible. Le système indique la disponibilité lors du contrôle de zone.
          </Text>

          <Text style={styles.question}>Je ne peux pas me connecter</Text>
          <Pressable onPress={() => router.push("/(public)/forgot-password" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Réinitialiser mon mot de passe</Text>
          </Pressable>

          <Text style={styles.question}>Mon compte n&apos;est pas activé</Text>
          <Pressable onPress={() => router.push("/(public)/login" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Reprendre l&apos;activation depuis la connexion</Text>
          </Pressable>

          <Text style={styles.question}>Je n&apos;ai pas reçu le SMS ou l&apos;e-mail</Text>
          <Pressable onPress={() => router.push("/(public)/contact" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Contacter le support</Text>
          </Pressable>

          <Pressable onPress={() => router.push("/(public)/how-it-works" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Voir comment ça marche</Text>
          </Pressable>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    paddingVertical: 24,
  },
  block: {
    gap: 14,
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#163A34",
  },
  intro: {
    fontSize: 15,
    lineHeight: 22,
    color: "#5F7369",
    marginBottom: 4,
  },
  question: {
    fontSize: 16,
    fontWeight: "700",
    color: "#163A34",
    marginTop: 6,
  },
  answer: {
    fontSize: 15,
    lineHeight: 22,
    color: "#475569",
  },
  linkWrap: {
    alignSelf: "flex-start",
  },
  link: {
    color: "#0A8F7A",
    fontWeight: "600",
    fontSize: 15,
    textDecorationLine: "underline",
  },
});
