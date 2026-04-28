import {
  ImageBackground,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");

export default function OnboardingStepTwoScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.screen}>
      <ImageBackground
        source={LANDING_BACKGROUND}
        style={StyleSheet.absoluteFillObject}
        resizeMode="cover"
        imageStyle={styles.backgroundImage}
      />
      <View style={styles.overlay} />

      <ScrollView
        contentContainerStyle={[
          styles.scrollContent,
          {
            paddingTop: Math.max(insets.top, 16) + 8,
            paddingBottom: Math.max(insets.bottom, 20) + 16,
          },
        ]}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.card}>
          <Text style={styles.stepPill}>Étape 2 sur 3</Text>

          <Text style={styles.title}>Avec ou sans compte, vous choisissez</Text>

          <Text style={styles.subtitle}>
            Commencez une demande rapidement sans compte, ou créez un compte pour enregistrer vos
            informations.
          </Text>

          <View style={styles.options}>
            <View style={styles.optionCard}>
              <View style={styles.optionIconWrap}>
                <Ionicons name="flash-outline" size={22} color="#0A8F7A" />
              </View>
              <View style={styles.optionBody}>
                <Text style={styles.optionTitle}>Sans compte</Text>
                <Text style={styles.optionDesc}>
                  Pré-demande rapide, puis finalisation lorsque vous êtes prêt.
                </Text>
              </View>
            </View>

            <View style={styles.optionCard}>
              <View style={styles.optionIconWrap}>
                <Ionicons name="person-circle-outline" size={22} color="#0A8F7A" />
              </View>
              <View style={styles.optionBody}>
                <Text style={styles.optionTitle}>Avec compte</Text>
                <Text style={styles.optionDesc}>
                  Historique, suivi des trajets et reprise de vos brouillons facilitée.
                </Text>
              </View>
            </View>
          </View>

          <Pressable
            onPress={() => router.push("/(public)/onboarding-step-3" as any)}
            style={({ pressed }) => [styles.primaryButton, pressed && styles.primaryButtonPressed]}
            accessibilityRole="button"
            accessibilityLabel="Suivant"
          >
            <Text style={styles.primaryButtonText}>Suivant</Text>
          </Pressable>

          <Pressable
            onPress={() => router.replace("/(public)/choice-guest-signup" as any)}
            style={({ pressed }) => [styles.skipWrap, pressed && styles.skipPressed]}
            accessibilityRole="button"
            accessibilityLabel="Passer l'introduction"
          >
            <Text style={styles.skipText}>Passer</Text>
          </Pressable>
        </View>
      </ScrollView>
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
    paddingHorizontal: 20,
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
  skipWrap: {
    marginTop: 16,
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 8,
  },
  skipPressed: {
    opacity: 0.75,
  },
  skipText: {
    color: "#0A8F7A",
    fontSize: 15,
    fontWeight: "600",
    textAlign: "center",
  },
});
