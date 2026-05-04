import { Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../src/design/responsive";

export default function ConditionsScreen() {
  const router = useRouter();

  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <Pressable onPress={() => router.back()} style={styles.back}>
          <Text style={styles.backText}>Retour</Text>
        </Pressable>

        <Text style={styles.title}>Conditions d&apos;utilisation</Text>

        <View style={styles.paragraphs}>
          <Text style={styles.p}>
            En utilisant LIRIE, vous acceptez de fournir des informations exactes pour organiser vos
            réservations.
          </Text>
          <Text style={styles.p}>
            L&apos;utilisation du service doit rester conforme au cadre légal et aux règles de sécurité
            applicables au transport médical.
          </Text>
          <Text style={styles.p}>
            Ces conditions peuvent évoluer; la version en vigueur est celle affichée au moment de votre
            utilisation.
          </Text>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    paddingVertical: 16,
  },
  back: {
    alignSelf: "flex-start",
    paddingVertical: 8,
    marginBottom: 8,
  },
  backText: {
    color: "#0A8F7A",
    fontWeight: "700",
    fontSize: 16,
  },
  title: {
    fontSize: 28,
    lineHeight: 34,
    color: "#163A34",
    fontWeight: "700",
    marginBottom: 16,
  },
  paragraphs: {
    gap: 14,
  },
  p: {
    color: "#45655D",
    lineHeight: 22,
    fontSize: 15,
  },
});
