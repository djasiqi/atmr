import {
  ActivityIndicator,
  Image,
  ImageBackground,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import type { TextInput as TextInputType } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import { useSession } from "../../src/core/sessionProvider";
import { resolveInitialRoute } from "../../src/core/navigation/resolveInitialRoute";
import { useRuntimeUpdateGate } from "../../src/core/version/useRuntimeUpdateGate";
import { getLastDraftId } from "../../src/core/public/preRequestDraft";
import { queueExternalIntentResume } from "../../src/core/navigation/externalIntent";
import {
  AppNotice,
  AppSwitch,
  AppText,
  Screen,
  scrollAnchorAboveKeyboard,
  useAppViewport,
  useKeyboardHeight,
} from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

type LoginApiError = {
  message?: string;
  reason?: string;
  details?: Record<string, unknown>;
};

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");
const LIRIE_LOGO = require("../../assets/images/lirie-logo-color.png");

/** Aligné sur `forgot-password.tsx` (carte, champs, bouton primaire). */
const UI_BORDER = "#91A59D";
const UI_TEXT = "#163A34";
const UI_MUTED = "#5F7369";
const BRAND = "#0A8F7A";
const BRAND_DISABLED = "#84B7AE";

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export default function LoginScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ next?: string }>();
  const { login, error, bootstrap } = useSession();
  const [email, setEmail] = ReactRuntime.useState("");
  const [password, setPassword] = ReactRuntime.useState("");
  const [showPassword, setShowPassword] = ReactRuntime.useState(false);
  const [rememberSession, setRememberSession] = ReactRuntime.useState(true);
  const [submitting, setSubmitting] = ReactRuntime.useState(false);
  const [localError, setLocalError] = ReactRuntime.useState(null as string | null);
  const passwordInputRef = ReactRuntime.useRef(null as TextInputType | null);
  const loginScrollRef = ReactRuntime.useRef(null as ScrollView | null);
  const loginScrollOffsetYRef = ReactRuntime.useRef(0);
  const emailFieldAnchorRef = ReactRuntime.useRef(null as View | null);
  const passwordFieldAnchorRef = ReactRuntime.useRef(null as View | null);
  const updateGate = useRuntimeUpdateGate();
  const { topInset } = useAppViewport();
  /** Clavier dual : `useKeyboardHeight` remplace le doublon listeners + magic numbers (cf. plan Sprint 1). */
  const { keyboardVisible, scrollPaddingBottom: keyboardScrollPaddingBottom } = useKeyboardHeight();

  ReactRuntime.useEffect(() => {
    if (keyboardVisible) return;
    loginScrollRef.current?.scrollTo({ y: 0, animated: true });
    loginScrollOffsetYRef.current = 0;
  }, [keyboardVisible]);

  if (bootstrap?.is_authenticated) {
    return <Redirect href={resolveInitialRoute(bootstrap) as any} />;
  }

  const onSubmit = async () => {
    setLocalError(null);
    if (!email.trim() || !password.trim()) {
      setLocalError("Email et mot de passe requis.");
      return;
    }
    setSubmitting(true);
    try {
      await login(email, password);
      const draftId = await getLastDraftId();
      if (draftId) {
        await queueExternalIntentResume({ type: "pre-request-resume", draftId });
      }
      if (params.next && typeof params.next === "string" && params.next.trim()) {
        router.replace(params.next as any);
        return;
      }
      router.replace("/");
    } catch (e) {
      const typedError = (typeof e === "object" && e ? e : {}) as LoginApiError;
      const message = asString(typedError.message) || "Echec de connexion.";
      const reason = asString(typedError.reason).toLowerCase();
      const activationSessionId = asString(typedError.details?.activation_session_id);
      const maskedEmail = asString(typedError.details?.masked_email);
      const maskedPhone = asString(typedError.details?.masked_phone);
      if (reason === "account_pending_activation") {
        if (activationSessionId) {
          router.replace({
            pathname: "/(public)/activate",
            params: {
              activation_session_id: activationSessionId,
              masked_email: maskedEmail,
              masked_phone: maskedPhone,
            },
          } as any);
          return;
        }
        setLocalError("Compte en attente d'activation. Vérifiez votre email et SMS.");
        return;
      }
      if (/mfa|required|otp/i.test(message) || reason.includes("mfa")) {
        router.push({
          pathname: "/(public)/mfa",
          params: { email: email.trim() },
        } as any);
        return;
      }
      setLocalError(message);
    } finally {
      setSubmitting(false);
    }
  };

  const submitDisabled =
    submitting || !email.trim() || !password.trim() || updateGate.requiresUpdate;

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
        withHorizontalPadding={false}
        backgroundColor="transparent"
        keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
        automaticallyAdjustKeyboardInsets={Platform.OS !== "web"}
        androidKeyboardFallback={Platform.OS === "android"}
        scrollViewRef={loginScrollRef}
        onScroll={(e) => {
          loginScrollOffsetYRef.current = e.nativeEvent.contentOffset.y;
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
              router.replace("/(public)" as any);
            }}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
            hitSlop={12}
          >
            <Ionicons name="arrow-back" size={22} color={BRAND} />
          </Pressable>

          <View style={styles.logoBlock}>
            <Image
              source={LIRIE_LOGO}
              style={styles.logo}
              resizeMode="contain"
              accessibilityRole="image"
              accessibilityLabel="LIRIE"
            />
          </View>

          <Text style={styles.kicker} maxFontSizeMultiplier={1.22}>
            Espace client
          </Text>
          <Text style={styles.title} maxFontSizeMultiplier={1.28}>
            Connexion sécurisée
          </Text>
          <Text style={styles.subtitle} maxFontSizeMultiplier={1.45}>
            Suivi en temps réel · Coordination médicale · Transport accompagné.
          </Text>
          {params.next ? (
            <Text style={styles.resumeHint} maxFontSizeMultiplier={1.45}>
              Connectez-vous pour finaliser votre réservation en reprenant vos informations.
            </Text>
          ) : null}

          {updateGate.requiresUpdate ? (
            <AppNotice
              variant="danger"
              title="Mise à jour obligatoire"
              ctaLabel="Mettre à jour pour continuer"
              onCtaPress={() => void updateGate.applyUpdate()}
            >
              {`Version minimum requise: ${updateGate.minimumSupportedVersion ?? "n/a"}`}
            </AppNotice>
          ) : null}

          <View ref={emailFieldAnchorRef} collapsable={false} style={styles.fieldBlock}>
            <Text style={styles.fieldLabel}>Email</Text>
            <TextInput
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
              autoComplete="email"
              textContentType="emailAddress"
              returnKeyType="next"
              onSubmitEditing={() => passwordInputRef.current?.focus()}
              onFocus={() =>
                scrollAnchorAboveKeyboard(loginScrollRef, loginScrollOffsetYRef, emailFieldAnchorRef)
              }
              placeholder="email@exemple.ch"
              placeholderTextColor="#91A59D"
              style={styles.fieldInput}
              {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
            />
          </View>

          <View ref={passwordFieldAnchorRef} collapsable={false} style={styles.fieldBlock}>
            <Text style={styles.fieldLabel}>Mot de passe</Text>
            <View style={styles.passwordShell}>
              <TextInput
                ref={passwordInputRef}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                autoComplete="current-password"
                textContentType="password"
                returnKeyType="done"
                onSubmitEditing={() => void onSubmit()}
                onFocus={() =>
                  scrollAnchorAboveKeyboard(
                    loginScrollRef,
                    loginScrollOffsetYRef,
                    passwordFieldAnchorRef,
                  )
                }
                placeholder="Mot de passe"
                placeholderTextColor="#91A59D"
                style={styles.passwordInput}
                {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
              />
              <Pressable
                onPress={() => setShowPassword((v: boolean) => !v)}
                style={styles.passwordToggle}
                hitSlop={8}
              >
                <Ionicons
                  name={showPassword ? "eye-off-outline" : "eye-outline"}
                  size={18}
                  color="#5F7369"
                  accessibilityLabel={showPassword ? "Masquer le mot de passe" : "Afficher le mot de passe"}
                />
              </Pressable>
            </View>
          </View>

          <AppSwitch
            value={rememberSession}
            onValueChange={setRememberSession}
            accessibilityLabel="Se souvenir de moi"
            style={styles.rememberRow}
            label={
              <Text style={styles.rememberLabel} maxFontSizeMultiplier={1.35}>
                Se souvenir de moi
              </Text>
            }
          />

          {(localError || error) ? (
            <AppText variant="error" style={{ marginTop: 14 }} accessibilityRole="alert">
              {localError ?? error}
            </AppText>
          ) : null}

          {!updateGate.requiresUpdate && (updateGate.updateAvailable || updateGate.recommendedUpdate) ? (
            <AppNotice
              variant="warning"
              ctaLabel="Mettre à jour maintenant"
              onCtaPress={() => void updateGate.applyUpdate()}
              style={{ marginTop: 18 }}
            >
              {`Une mise à jour est disponible et recommandée (cible: ${updateGate.recommendedVersion ?? "latest"}).`}
            </AppNotice>
          ) : null}

          <Pressable
            onPress={() => void onSubmit()}
            disabled={submitDisabled}
            style={[styles.submitButton, submitDisabled ? styles.submitButtonDisabled : null]}
          >
            {submitting ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <AppText variant="label" style={styles.submitText}>
                Se connecter
              </AppText>
            )}
          </Pressable>

          <View style={styles.linksRow}>
            <Pressable
              onPress={() => router.push("/(public)/forgot-password" as any)}
              style={({ pressed }) => [styles.linkHit, pressed && styles.linkHitPressed]}
            >
              <AppText variant="label" style={styles.primaryLink} maxFontSizeMultiplier={1.28}>
                Mot de passe oublié ?
              </AppText>
            </Pressable>
            <Text style={styles.linksDot} accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
              ·
            </Text>
            <Pressable
              onPress={() => router.push("/(public)/signup" as any)}
              style={({ pressed }) => [styles.linkHit, pressed && styles.linkHitPressed]}
            >
              <AppText variant="label" style={styles.secondaryLink} maxFontSizeMultiplier={1.28}>
                Créer un compte
              </AppText>
            </Pressable>
          </View>
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
    paddingHorizontal: 16,
  },
  /**
   * iOS / Android : appliqué **seulement** pendant que le clavier est ouvert (`keyboardDidShow`).
   * Le `paddingBottom` est calé sur la hauteur réelle du clavier (+ marge) pour tous formats d’écran.
   * Sans clavier : `scrollContent` seul (carte centrée, pas de jeu de scroll artificiel).
   */
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
  logoBlock: {
    alignItems: "center",
    marginBottom: 12,
  },
  logo: {
    height: 26,
    width: 168,
  },
  kicker: {
    color: BRAND,
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 8,
    textAlign: "center",
  },
  title: {
    color: UI_TEXT,
    fontSize: FONT_SIZE.px30,
    lineHeight: 34,
    fontWeight: "700",
    textAlign: "center",
    alignSelf: "center",
    maxWidth: 320,
  },
  subtitle: {
    color: UI_MUTED,
    fontSize: FONT_SIZE.px15,
    lineHeight: 21,
    marginTop: 10,
    textAlign: "center",
  },
  resumeHint: {
    marginTop: 14,
    color: "#45655D",
    fontSize: FONT_SIZE.px14,
    lineHeight: 20,
    textAlign: "center",
  },
  fieldBlock: {
    marginTop: 18,
  },
  fieldLabel: {
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "600",
    color: UI_TEXT,
    marginBottom: 8,
  },
  fieldInput: {
    minHeight: 50,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: UI_BORDER,
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 14,
    color: UI_TEXT,
    fontSize: FONT_SIZE.px16,
  },
  passwordShell: {
    flexDirection: "row",
    alignItems: "center",
    minHeight: 50,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: UI_BORDER,
    backgroundColor: "#FFFFFF",
    paddingLeft: 14,
    paddingRight: 6,
  },
  passwordInput: {
    flex: 1,
    minHeight: 48,
    paddingVertical: Platform.OS === "ios" ? 12 : 10,
    fontSize: FONT_SIZE.px16,
    color: UI_TEXT,
    borderWidth: 0,
    ...Platform.select({
      web: { outlineStyle: "none" as const },
      default: {},
    }),
  },
  passwordToggle: {
    justifyContent: "center",
    paddingHorizontal: 6,
    paddingVertical: 4,
  },
  rememberRow: {
    marginTop: 18,
  },
  rememberLabel: {
    lineHeight: 20,
    color: UI_MUTED,
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
  },
  submitButton: {
    marginTop: 22,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: BRAND,
    alignSelf: "stretch",
  },
  submitButtonDisabled: {
    backgroundColor: BRAND_DISABLED,
  },
  submitText: {
    color: "#FFFFFF",
    letterSpacing: 0.2,
  },
  linksRow: {
    marginTop: 18,
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    rowGap: 10,
  },
  linkHit: {
    paddingVertical: 8,
    paddingHorizontal: 6,
    borderRadius: 10,
  },
  linkHitPressed: {
    backgroundColor: "rgba(10, 143, 122, 0.06)",
  },
  linksDot: {
    color: UI_MUTED,
    fontSize: FONT_SIZE.px13,
    fontWeight: "700",
    opacity: 0.65,
    paddingHorizontal: 2,
  },
  primaryLink: {
    color: BRAND,
    fontWeight: "600",
  },
  secondaryLink: {
    color: BRAND,
    fontWeight: "600",
    opacity: 0.92,
  },
});
