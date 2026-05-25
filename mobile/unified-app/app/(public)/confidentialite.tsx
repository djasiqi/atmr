import { Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

export default function ConfidentialiteScreen() {
  const router = useRouter();

  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <Pressable onPress={() => router.back()} style={styles.back}>
          <Text style={styles.backText}>Retour</Text>
        </Pressable>

        <Text style={styles.title}>Politique de confidentialité</Text>

        <View style={styles.paragraphs}>
          <Text style={styles.p}>
            Les données personnelles sont utilisées uniquement pour gérer votre compte, vos réservations
            et les notifications associées.
          </Text>
          <Text style={styles.p}>
            Nous appliquons des mesures de sécurité adaptées pour protéger vos informations de contact
            et d&apos;accès.
          </Text>
          <Text style={styles.p}>
            Vous pouvez demander la consultation, la correction ou la suppression de vos données selon
            les règles en vigueur.
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
    fontSize: FONT_SIZE.px16,
  },
  title: {
    fontSize: FONT_SIZE.px28,
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
    fontSize: FONT_SIZE.px15,
  },
});
