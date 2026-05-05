import {
  ImageBackground,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import * as Linking from "expo-linking";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { AppText, Screen, useAppViewport } from "../../src/design/responsive";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");

export default function ContactScreen() {
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
              router.replace("/(public)" as any);
            }}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
          >
            <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
          </Pressable>

          <Text style={styles.kicker}>Support</Text>
          <Text style={styles.title}>Contact & assistance</Text>
          <Text style={styles.intro}>
            Notre équipe répond pour les problèmes de connexion, d&apos;activation de compte et de
            réservation.
          </Text>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Établissements</Text>
            <Text style={styles.body}>
              Demandez un accès institution pour la planification patient et le suivi des transports.
            </Text>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Transporteurs</Text>
            <Text style={styles.body}>
              Rejoignez Lirie pour recevoir des missions et connecter votre flotte.
            </Text>
          </View>

          <View style={styles.contactBox}>
            <View style={styles.contactRow}>
              <Ionicons name="mail-outline" size={18} color="#5F7369" />
              <Text style={styles.contactText}>support@lirie.ch</Text>
            </View>
            <View style={styles.contactRow}>
              <Ionicons name="call-outline" size={18} color="#5F7369" />
              <Text style={styles.contactText}>+41 22 000 00 00 · 08:00–18:00</Text>
            </View>
          </View>

          <Pressable
            onPress={() => void Linking.openURL("mailto:support@lirie.ch")}
            style={({ pressed }) => [styles.primaryButton, pressed && styles.primaryPressed]}
            accessibilityRole="button"
            accessibilityLabel="Envoyer un e-mail"
          >
            <Ionicons name="mail-outline" size={20} color="#FFFFFF" />
            <AppText variant="label" style={styles.primaryButtonText}>
              Envoyer un e-mail
            </AppText>
          </Pressable>

          <Pressable
            onPress={() => void Linking.openURL("tel:+41220000000")}
            style={({ pressed }) => [styles.secondaryButton, pressed && styles.secondaryPressed]}
            accessibilityRole="button"
            accessibilityLabel="Appeler le support"
          >
            <Ionicons name="call-outline" size={20} color="#0A8F7A" />
            <AppText variant="label" style={styles.secondaryButtonText}>
              Appeler le support
            </AppText>
          </Pressable>

          <Text style={styles.footnote}>
            Si vous n&apos;avez pas accès à l&apos;application, les flux d&apos;activation et de mot de
            passe restent disponibles depuis le web.
          </Text>
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
    paddingVertical: 24,
    paddingHorizontal: 22,
  },
  card: {
    width: "100%",
    maxWidth: 420,
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
    fontSize: 13,
    fontWeight: "500",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  title: {
    color: "#163A34",
    fontSize: 28,
    lineHeight: 32,
    fontWeight: "700",
  },
  intro: {
    fontSize: 15,
    lineHeight: 22,
    color: "#5F7369",
    marginTop: 10,
  },
  section: {
    marginTop: 18,
    gap: 6,
  },
  sectionTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: "#163A34",
  },
  body: {
    fontSize: 15,
    lineHeight: 22,
    color: "#5F7369",
  },
  contactBox: {
    marginTop: 20,
    paddingVertical: 14,
    paddingHorizontal: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.38)",
    backgroundColor: "rgba(10,143,122,0.04)",
    gap: 10,
  },
  contactRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  contactText: {
    flex: 1,
    fontSize: 15,
    lineHeight: 20,
    color: "#163A34",
    fontWeight: "500",
  },
  primaryButton: {
    marginTop: 20,
    minHeight: 54,
    borderRadius: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    backgroundColor: "#0A8F7A",
  },
  primaryPressed: {
    opacity: 0.92,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    letterSpacing: 0.2,
  },
  secondaryButton: {
    marginTop: 12,
    minHeight: 54,
    borderRadius: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.45)",
    backgroundColor: "#FFFFFF",
  },
  secondaryPressed: {
    backgroundColor: "rgba(10,143,122,0.06)",
  },
  secondaryButtonText: {
    color: "#0A8F7A",
    fontWeight: "600",
    letterSpacing: 0.15,
  },
  footnote: {
    fontSize: 13,
    lineHeight: 19,
    color: "#6F857E",
    marginTop: 18,
  },
});
