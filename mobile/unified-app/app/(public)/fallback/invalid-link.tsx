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
import { AppText, Screen, useAppViewport } from "../../../src/design/responsive";
import { FONT_SIZE } from "../../../src/design/responsive/typographyTokens";

const LANDING_BACKGROUND = require("../../../assets/images/landing-background.png");

export default function InvalidLinkScreen() {
  const router = useRouter();
  const { topInset } = useAppViewport();

  const goHome = () => router.replace("/(public)" as any);

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
              goHome();
            }}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
          >
            <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
          </Pressable>

          <View style={styles.iconBadge} importantForAccessibility="no">
            <Ionicons name="link-outline" size={28} color="#B45309" />
          </View>

          <Text style={styles.kicker}>Lien</Text>
          <Text style={styles.title}>Lien invalide</Text>
          <Text style={styles.subtitle}>
            Le format du lien est incorrect ou incomplet. Réessayez depuis la source d&apos;origine.
          </Text>

          <Pressable onPress={goHome} style={({ pressed }) => [styles.primaryButton, pressed && styles.primaryPressed]}>
            <AppText variant="label" style={styles.primaryButtonText}>
              Retour à l&apos;accueil
            </AppText>
          </Pressable>

          <Pressable
            onPress={() => router.replace("/(public)/contact" as any)}
            style={styles.secondaryLinkWrap}
          >
            <AppText variant="label" style={styles.secondaryLink}>
              Contacter le support
            </AppText>
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
    marginBottom: 8,
  },
  iconBadge: {
    alignSelf: "center",
    width: 56,
    height: 56,
    borderRadius: 16,
    backgroundColor: "rgba(180,83,9,0.10)",
    borderWidth: 1,
    borderColor: "rgba(180,83,9,0.22)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 16,
  },
  kicker: {
    color: "#0A8F7A",
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 8,
    textAlign: "center",
  },
  title: {
    color: "#163A34",
    fontSize: FONT_SIZE.px26,
    lineHeight: 30,
    fontWeight: "700",
    textAlign: "center",
  },
  subtitle: {
    color: "#5F7369",
    fontSize: FONT_SIZE.px15,
    lineHeight: 22,
    marginTop: 12,
    textAlign: "center",
  },
  primaryButton: {
    marginTop: 22,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
  },
  primaryPressed: {
    opacity: 0.92,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    letterSpacing: 0.2,
  },
  secondaryLinkWrap: {
    marginTop: 16,
    alignItems: "center",
    paddingVertical: 6,
  },
  secondaryLink: {
    color: "#0A8F7A",
    fontWeight: "600",
  },
});
