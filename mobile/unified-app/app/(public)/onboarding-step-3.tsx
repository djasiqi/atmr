import {
  ImageBackground,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { Screen, useAppViewport } from "../../src/design/responsive";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");

export default function OnboardingStepThreeScreen() {
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
          <Text style={styles.stepPill}>Étape 3 sur 3</Text>

          <Text style={styles.title}>Gagnez du temps avec un compte</Text>

          <Text style={styles.subtitle}>
            Un compte LIRIE centralise vos trajets et accélère chaque prochaine demande.
          </Text>

          <View style={styles.options}>
            <View style={styles.optionCard}>
              <View style={styles.optionIconWrap}>
                <Ionicons name="albums-outline" size={22} color="#0A8F7A" />
              </View>
              <View style={styles.optionBody}>
                <Text style={styles.optionTitle}>Historique</Text>
                <Text style={styles.optionDesc}>Retrouvez vos trajets passés en un coup d&apos;œil.</Text>
              </View>
            </View>

            <View style={styles.optionCard}>
              <View style={styles.optionIconWrap}>
                <Ionicons name="options-outline" size={22} color="#0A8F7A" />
              </View>
              <View style={styles.optionBody}>
                <Text style={styles.optionTitle}>Préférences</Text>
                <Text style={styles.optionDesc}>Adresses et habitudes enregistrées pour aller plus vite.</Text>
              </View>
            </View>

            <View style={styles.optionCard}>
              <View style={styles.optionIconWrap}>
                <Ionicons name="pulse-outline" size={22} color="#0A8F7A" />
              </View>
              <View style={styles.optionBody}>
                <Text style={styles.optionTitle}>Suivi</Text>
                <Text style={styles.optionDesc}>Statut des réservations plus lisible, moins de friction.</Text>
              </View>
            </View>
          </View>

          <Pressable
            onPress={() => router.replace("/(public)/pre-request/step-1" as any)}
            style={({ pressed }) => [styles.primaryButton, pressed && styles.primaryButtonPressed]}
            accessibilityRole="button"
            accessibilityLabel="Réserver sans compte"
          >
            <Text style={styles.primaryButtonText}>Réserver sans compte</Text>
          </Pressable>

          <Pressable
            onPress={() => router.replace("/(public)/signup" as any)}
            style={({ pressed }) => [styles.secondaryButton, pressed && styles.secondaryButtonPressed]}
            accessibilityRole="button"
            accessibilityLabel="Créer un compte"
          >
            <Text style={styles.secondaryButtonText}>Créer un compte</Text>
          </Pressable>

          <Pressable
            onPress={() => router.replace("/(public)/login" as any)}
            style={({ pressed }) => [styles.tertiaryWrap, pressed && styles.tertiaryPressed]}
            accessibilityRole="button"
            accessibilityLabel="Se connecter"
          >
            <Text style={styles.tertiaryText}>Se connecter</Text>
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
  stepPill: {
    alignSelf: "flex-start",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: "#0A8F7A",
    backgroundColor: "rgba(10,143,122,0.12)",
    overflow: "hidden",
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 999,
    marginBottom: 18,
  },
  title: {
    fontFamily: "Philosopher_700Bold",
    fontSize: 28,
    lineHeight: 34,
    color: "#163A34",
    letterSpacing: -0.2,
  },
  subtitle: {
    marginTop: 12,
    fontSize: 16,
    lineHeight: 24,
    color: "#5F7369",
  },
  options: {
    marginTop: 20,
    gap: 12,
  },
  optionCard: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 14,
    padding: 16,
    borderRadius: 16,
    backgroundColor: "rgba(10,143,122,0.06)",
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.18)",
  },
  optionIconWrap: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: "rgba(10,143,122,0.14)",
    alignItems: "center",
    justifyContent: "center",
  },
  optionBody: {
    flex: 1,
  },
  optionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#163A34",
    marginBottom: 4,
  },
  optionDesc: {
    fontSize: 14,
    lineHeight: 20,
    color: "#5F7369",
  },
  primaryButton: {
    marginTop: 24,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
  },
  primaryButtonPressed: {
    opacity: 0.92,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  secondaryButton: {
    marginTop: 12,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(10,143,122,0.08)",
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
  },
  secondaryButtonPressed: {
    opacity: 0.9,
  },
  secondaryButtonText: {
    color: "#0A8F7A",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  tertiaryWrap: {
    marginTop: 12,
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 8,
  },
  tertiaryPressed: {
    opacity: 0.75,
  },
  tertiaryText: {
    color: "#0A8F7A",
    fontSize: 15,
    fontWeight: "600",
    textAlign: "center",
  },
});
