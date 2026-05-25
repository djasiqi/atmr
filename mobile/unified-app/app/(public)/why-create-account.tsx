import { ImageBackground, Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { Screen, useAppViewport } from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");

const BENEFITS = [
  "Suivre vos transports en temps reel",
  "Modifier vos demandes rapidement",
  "Recevoir les notifications chauffeur",
  "Consulter votre historique",
  "Gerer les paiements et retours",
];

export default function WhyCreateAccountScreen() {
  const router = useRouter();
  const { topInset } = useAppViewport();

  return (
    <View style={styles.screen}>
      <ImageBackground
        source={LANDING_BACKGROUND}
        style={StyleSheet.absoluteFillObject}
        resizeMode="cover"
        imageStyle={styles.backgroundImage}
      />
      <View style={styles.overlay} />

      <Screen
        scroll
        backgroundColor="transparent"
        keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
        contentContainerStyle={styles.scrollContent}
      >
        <View style={styles.card}>
          <Pressable
            onPress={() => {
              if (router.canGoBack()) {
                router.back();
                return;
              }
              router.replace("/(public)/signup" as any);
            }}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
          >
            <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
          </Pressable>

          <Text style={styles.kicker}>Compte client</Text>
          <Text style={styles.title}>Pourquoi creer un compte</Text>
          <Text style={styles.subtitle}>
            Votre espace securise centralise vos trajets, vos infos et vos confirmations sans ressaisie.
          </Text>

          <View style={styles.benefitsBlock}>
            {BENEFITS.map((benefit) => (
              <View key={benefit} style={styles.benefitRow}>
                <View style={styles.benefitIconWrap}>
                  <Ionicons name="checkmark" size={13} color="#FFFFFF" />
                </View>
                <Text style={styles.benefitText}>{benefit}</Text>
              </View>
            ))}
          </View>

          <Pressable
            onPress={() => router.push("/(public)/signup" as any)}
            style={({ pressed }) => [styles.primaryCta, pressed ? styles.primaryCtaPressed : null]}
          >
            <Text style={styles.primaryCtaText}>Creer mon compte</Text>
          </Pressable>

          <Pressable
            onPress={() => router.push("/(public)/login" as any)}
            style={({ pressed }) => [styles.secondaryCta, pressed ? styles.secondaryCtaPressed : null]}
          >
            <Text style={styles.secondaryCtaText}>J&apos;ai deja un compte</Text>
          </Pressable>
        </View>
      </Screen>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#EAF3F1",
  },
  backgroundImage: {
    opacity: 0.08,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(234,243,241,0.88)",
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 24,
  },
  card: {
    width: "100%",
    maxWidth: 430,
    alignSelf: "center",
    borderRadius: 26,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
    ...Platform.select({
      web: { boxShadow: "0 20px 48px rgba(22,58,52,0.12)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.12,
        shadowRadius: 18,
        shadowOffset: { width: 0, height: 8 },
        elevation: 4,
      },
    }),
  },
  backButton: {
    alignSelf: "flex-start",
    paddingVertical: 6,
    paddingHorizontal: 2,
    marginBottom: 14,
  },
  kicker: {
    color: "#0A8F7A",
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  title: {
    color: "#163A34",
    fontSize: FONT_SIZE.px29,
    lineHeight: 34,
    fontWeight: "700",
  },
  subtitle: {
    marginTop: 10,
    color: "#5F7369",
    fontSize: FONT_SIZE.px15,
    lineHeight: 21,
  },
  benefitsBlock: {
    marginTop: 18,
    gap: 10,
  },
  benefitRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  benefitIconWrap: {
    width: 20,
    height: 20,
    borderRadius: 999,
    backgroundColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
  },
  benefitText: {
    flex: 1,
    color: "#1F2E2A",
    fontSize: FONT_SIZE.px14_5,
    lineHeight: 20,
    fontWeight: "500",
  },
  primaryCta: {
    marginTop: 22,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
  },
  primaryCtaPressed: {
    opacity: 0.93,
  },
  primaryCtaText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: FONT_SIZE.px16,
    letterSpacing: 0.2,
  },
  secondaryCta: {
    marginTop: 10,
    minHeight: 54,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.36)",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F7FCFA",
  },
  secondaryCtaPressed: {
    opacity: 0.9,
  },
  secondaryCtaText: {
    color: "#0A8F7A",
    fontWeight: "700",
    fontSize: FONT_SIZE.px15_5,
  },
});
