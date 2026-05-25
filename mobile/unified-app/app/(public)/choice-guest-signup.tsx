import { Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

export default function ChoiceGuestOrSignupScreen() {
  const router = useRouter();

  return (
    <Screen scroll backgroundColor="#EAF3F1" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.card}>
          <Text style={styles.title}>Comment souhaitez-vous continuer ?</Text>
          <Text style={styles.lede}>
            Un formulaire pour le trajet et un telephone de confirmation, sans creer de profil. Paiement a
            l&apos;etape suivante.
          </Text>

          <View style={styles.infoBlue}>
            <Text style={styles.infoTitle}>Sans compte</Text>
            <Text style={styles.infoBody}>
              Le parcours le plus court : tout sur une page, puis identification pour valider la reservation.
            </Text>
          </View>

          <Pressable
            onPress={() => router.push("/(public)/pre-request/step-1" as any)}
            style={styles.primaryBtn}
          >
            <Text style={styles.primaryBtnText}>Continuer sans compte</Text>
          </Pressable>

          <View style={styles.infoNeutral}>
            <Text style={styles.infoTitle}>Creer un compte</Text>
            <Text style={styles.bodyMuted}>
              Retrouvez vos reservations, vos preferences et votre historique.
            </Text>
          </View>
          <Pressable
            onPress={() => router.push("/(public)/signup" as any)}
            style={styles.outlineBtn}
          >
            <Text style={styles.outlineBtnText}>Creer un compte</Text>
          </Pressable>
          <Pressable onPress={() => router.push("/(public)/login" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>J&apos;ai deja un compte</Text>
          </Pressable>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 24,
  },
  card: {
    gap: 14,
    borderRadius: 26,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
  },
  title: {
    fontSize: FONT_SIZE.px26,
    fontWeight: "800",
    color: "#163A34",
  },
  lede: {
    color: "#45655D",
    lineHeight: 22,
  },
  infoBlue: {
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.25)",
    backgroundColor: "rgba(10,143,122,0.06)",
    borderRadius: 14,
    padding: 14,
    gap: 6,
  },
  infoNeutral: {
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    borderRadius: 14,
    padding: 14,
    gap: 6,
  },
  infoTitle: {
    fontWeight: "700",
    color: "#163A34",
  },
  infoBody: {
    color: "#45655D",
    lineHeight: 20,
  },
  bodyMuted: {
    color: "#45655D",
    lineHeight: 20,
  },
  primaryBtn: {
    backgroundColor: "#0A8F7A",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryBtnText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: FONT_SIZE.px16,
  },
  outlineBtn: {
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
    backgroundColor: "#FFFFFF",
  },
  outlineBtnText: {
    color: "#0A8F7A",
    fontWeight: "700",
    fontSize: FONT_SIZE.px16,
  },
  linkWrap: {
    alignItems: "center",
    paddingVertical: 8,
  },
  link: {
    color: "#0A8F7A",
    fontWeight: "600",
    fontSize: FONT_SIZE.px15,
  },
});
