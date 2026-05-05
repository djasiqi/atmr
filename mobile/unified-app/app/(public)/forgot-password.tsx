import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  ImageBackground,
  Keyboard,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { apiClient } from "../../src/core/api/client";
import { AppText, Screen, scrollAnchorAboveKeyboard, useAppViewport } from "../../src/design/responsive";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");

export default function ForgotPasswordScreen() {
  const router = useRouter();
  const { topInset } = useAppViewport();
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const forgotScrollRef = useRef<ScrollView | null>(null);
  const forgotScrollOffsetYRef = useRef(0);
  const emailFieldAnchorRef = useRef<View | null>(null);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [keyboardScrollPaddingBottom, setKeyboardScrollPaddingBottom] = useState(0);

  useEffect(() => {
    if (Platform.OS === "web") return;
    const show = Keyboard.addListener("keyboardDidShow", (e) => {
      const h = e.endCoordinates?.height ?? 0;
      const computed = h > 0 ? Math.round(h + 48) : 300;
      setKeyboardScrollPaddingBottom(Math.max(260, computed));
      setKeyboardVisible(true);
    });
    const hide = Keyboard.addListener("keyboardDidHide", () => {
      setKeyboardVisible(false);
      setKeyboardScrollPaddingBottom(0);
      forgotScrollRef.current?.scrollTo({ y: 0, animated: true });
      forgotScrollOffsetYRef.current = 0;
    });
    return () => {
      show.remove();
      hide.remove();
    };
  }, []);

  const submit = async () => {
    setPending(true);
    setMessage(null);
    setError(null);
    try {
      await apiClient.post("/auth/forgot-password", { email: email.trim() });
      setMessage("Si cet email existe, un lien de reinitialisation a ete envoye.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Echec de la demande de reinitialisation.");
    } finally {
      setPending(false);
    }
  };

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
        automaticallyAdjustKeyboardInsets={Platform.OS !== "web"}
        androidKeyboardFallback={Platform.OS === "android"}
        scrollViewRef={forgotScrollRef}
        onScroll={(e) => {
          forgotScrollOffsetYRef.current = e.nativeEvent.contentOffset.y;
        }}
        scrollEventThrottle={16}
        contentContainerStyle={[
          styles.scrollContent,
          Platform.OS !== "web" && keyboardVisible
            ? [styles.scrollContentWithKeyboard, { paddingBottom: keyboardScrollPaddingBottom }]
            : null,
        ]}
      >
        <View style={styles.card}>
          <Pressable
            onPress={() => {
              if (router.canGoBack()) {
                router.back();
                return;
              }
              router.replace("/(public)/login" as any);
            }}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
          >
            <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
          </Pressable>

          <Text style={styles.kicker}>Recuperation</Text>
          <Text style={styles.title}>Mot de passe oublie</Text>
          <Text style={styles.subtitle}>
            Saisissez votre adresse email pour recevoir un lien de reinitialisation.
          </Text>

          <View ref={emailFieldAnchorRef} collapsable={false} style={styles.fieldBlock}>
            <TextInput
              value={email}
              onChangeText={(value) => {
                setEmail(value);
                if (message) setMessage(null);
              }}
              placeholder="email@exemple.ch"
              placeholderTextColor="#91A59D"
              autoCapitalize="none"
              keyboardType="email-address"
              autoComplete="email"
              textContentType="emailAddress"
              returnKeyType="done"
              onSubmitEditing={() => void submit()}
              onFocus={() =>
                scrollAnchorAboveKeyboard(forgotScrollRef, forgotScrollOffsetYRef, emailFieldAnchorRef)
              }
              style={styles.fieldInput}
              {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
            />
          </View>

          <Pressable
            onPress={() => void submit()}
            disabled={pending || email.trim().length === 0}
            style={[
              styles.submitButton,
              pending || email.trim().length === 0 ? styles.submitButtonDisabled : null,
            ]}
          >
            {pending ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <AppText variant="label" style={styles.submitText}>
                Envoyer le lien
              </AppText>
            )}
          </Pressable>

          {message ? (
            <AppText variant="body" style={styles.successText}>
              {message}
            </AppText>
          ) : null}
          {error ? (
            <AppText variant="error" style={styles.errorText}>
              {error}
            </AppText>
          ) : null}

          <Pressable onPress={() => router.replace("/(public)/login" as any)} style={styles.bottomLinkWrap}>
            <AppText variant="label" style={styles.bottomLink}>
              Retour a la connexion
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
  },
  /** Natif, uniquement clavier ouvert : même logique que `login.tsx` (jeu de scroll + padding bas dynamique). */
  scrollContentWithKeyboard: {
    justifyContent: "flex-start",
    paddingTop: 28,
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
    fontSize: 30,
    lineHeight: 34,
    fontWeight: "700",
  },
  subtitle: {
    color: "#5F7369",
    fontSize: 15,
    lineHeight: 21,
    marginTop: 10,
  },
  fieldBlock: {
    marginTop: 18,
  },
  fieldInput: {
    minHeight: 50,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#91A59D",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 14,
    color: "#163A34",
    fontSize: 16,
  },
  submitButton: {
    marginTop: 20,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
  },
  submitButtonDisabled: {
    backgroundColor: "#84B7AE",
  },
  submitText: {
    color: "#FFFFFF",
    letterSpacing: 0.2,
  },
  successText: {
    marginTop: 12,
    color: "#2E7D32",
    fontWeight: "600",
  },
  errorText: {
    marginTop: 12,
    fontWeight: "600",
  },
  bottomLinkWrap: {
    marginTop: 14,
    alignItems: "center",
  },
  bottomLink: {
    color: "#0A8F7A",
    fontWeight: "600",
  },
});
