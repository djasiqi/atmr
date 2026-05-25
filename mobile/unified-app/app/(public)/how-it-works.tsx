import { Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

export default function HowItWorksScreen() {
  const router = useRouter();
  return (
    <Screen scroll backgroundColor="#EAF3F1" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Comment ca marche</Text>
          <Text style={styles.intro}>Lirie simplifie l&apos;organisation des transports medicaux en 5 etapes.</Text>
          <View style={styles.step}>
            <Text style={styles.stepTitle}>1. Je fais une demande</Text>
            <Text style={styles.stepBody}>
              Depart, destination, date et besoins specifiques. Vous pouvez commencer sans compte.
            </Text>
          </View>
          <View style={styles.step}>
            <Text style={styles.stepTitle}>2. Un transporteur accepte</Text>
            <Text style={styles.stepBody}>
              L&apos;ecosysteme Lirie assigne la demande selon disponibilite et contexte.
            </Text>
          </View>
          <View style={styles.step}>
            <Text style={styles.stepTitle}>3. Vous recevez la confirmation</Text>
            <Text style={styles.stepBody}>
              Notification claire de l&apos;etat: en attente, confirme, en route, termine.
            </Text>
          </View>
          <View style={styles.step}>
            <Text style={styles.stepTitle}>4. Le transport est assure</Text>
            <Text style={styles.stepBody}>
              Chauffeur, compagnie et etablissement restent synchronises selon votre contexte.
            </Text>
          </View>
          <View style={styles.step}>
            <Text style={styles.stepTitle}>5. Paiement ou facturation</Text>
            <Text style={styles.stepBody}>
              Selon votre situation: parcours patient, institutionnel ou prise en charge.
            </Text>
          </View>
          <Pressable onPress={() => router.push("/(public)/pre-request/step-1" as any)} style={styles.cta}>
            <Text style={styles.ctaText}>Demarrer une pre-demande</Text>
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
    fontSize: FONT_SIZE.px24,
    fontWeight: "800",
    color: "#163A34",
  },
  intro: {
    color: "#45655D",
    lineHeight: 22,
    marginBottom: 4,
  },
  step: {
    gap: 8,
  },
  stepTitle: {
    fontWeight: "700",
    color: "#163A34",
  },
  stepBody: {
    color: "#45655D",
    lineHeight: 22,
  },
  cta: {
    marginTop: 8,
    backgroundColor: "#0A8F7A",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
  },
  ctaText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: FONT_SIZE.px16,
  },
});
