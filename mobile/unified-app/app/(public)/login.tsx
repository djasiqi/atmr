import {
  Image,
  ImageBackground,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import type { TextInput } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import { useSession } from "../../src/core/sessionProvider";
import { resolveInitialRoute } from "../../src/core/navigation/resolveInitialRoute";
import { useRuntimeUpdateGate } from "../../src/core/version/useRuntimeUpdateGate";
import { getLastDraftId } from "../../src/core/public/preRequestDraft";
import { queueExternalIntentResume } from "../../src/core/navigation/externalIntent";
import {
  AppButton,
  AppInput,
  AppNotice,
  AppSwitch,
  AppText,
  brandPrimary,
  brandText,
  brandTextMuted,
  getPublicBackButtonMetrics,
  ResponsiveContainer,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../src/design/responsive";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

type LoginApiError = {
  message?: string;
  reason?: string;
  details?: Record<string, unknown>;
};

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");
const LIRIE_LOGO = require("../../assets/images/lirie-logo-color.png");

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
  const [localError, setLocalError] = ReactRuntime.useState(
    null as string | null
  );
  const passwordInputRef = ReactRuntime.useRef(null as TextInput | null);
  const updateGate = useRuntimeUpdateGate();
  const viewport = useAppViewport();
  const { topInset } = viewport;
  const { landing: layout } = useResponsiveTokens();

  const accentLayout = ReactRuntime.useMemo(() => {
    const narrow = viewport.isTiny || viewport.isCompact;
    return {
      largeTop: viewport.isTablet ? -105 : narrow ? -98 : -92,
      largeRight: viewport.isTablet ? -48 : -32,
      smallTop: viewport.isTablet ? 95 : narrow ? 72 : 82,
      smallRight: viewport.isTablet ? 72 : 52,
    };
  }, [viewport.isCompact, viewport.isTablet, viewport.isTiny]);

  const backBtn = ReactRuntime.useMemo(
    () => getPublicBackButtonMetrics(viewport),
    [viewport.isCompact, viewport.isTiny]
  );

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

  return (
    <View style={styles.screen}>
      <ImageBackground
        source={LANDING_BACKGROUND}
        style={StyleSheet.absoluteFillObject}
        resizeMode="cover"
        imageStyle={styles.backgroundImage}
      />
      <View style={styles.overlay} />
      <View
        style={[
          styles.accentGlowLarge,
          {
            top: accentLayout.largeTop,
            right: accentLayout.largeRight,
            pointerEvents: "none",
          },
        ]}
      />
      <View
        style={[
          styles.accentGlowSmall,
          {
            top: accentLayout.smallTop,
            right: accentLayout.smallRight,
            pointerEvents: "none",
          },
        ]}
      />

      <Screen
        scroll
        withHorizontalPadding={false}
        backgroundColor="transparent"
        keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
        contentContainerStyle={styles.screenScrollContent}
      >
        <ResponsiveContainer
          style={{
            padding: layout.columnPadding,
            maxWidth: layout.contentMaxWidth,
          }}
        >
          <View
            style={[
              styles.card,
              {
                maxWidth: layout.cardMaxWidth,
                paddingHorizontal: layout.cardPadding + 6,
                paddingTop: layout.cardPadding + 10,
                paddingBottom: layout.cardPadding + 14,
                borderRadius: layout.cardRadius,
              },
            ]}
          >
          <View style={styles.cardBrandAccent} />
          <Pressable
            onPress={() => {
              if (router.canGoBack()) {
                router.back();
                return;
              }
              router.replace("/(public)" as any);
            }}
            style={({ pressed }) => [
              styles.backButtonBase,
              {
                paddingVertical: backBtn.paddingVertical,
                paddingHorizontal: backBtn.paddingHorizontal,
                marginLeft: backBtn.marginLeft,
                marginBottom: backBtn.marginBottom,
                borderRadius: backBtn.borderRadius,
                backgroundColor: pressed
                  ? backBtn.backgroundColorPressed
                  : backBtn.backgroundColorIdle,
              },
            ]}
            accessibilityRole="button"
            accessibilityLabel="Retour"
            hitSlop={12}
          >
            <Ionicons name="arrow-back" size={backBtn.iconSize} color={brandPrimary} />
          </Pressable>

          <View style={[styles.logoBlock, { marginBottom: layout.cardLineGap + 6 }]}>
            <Image
              source={LIRIE_LOGO}
              style={{ height: layout.logoHeight, width: layout.logoWidth }}
              resizeMode="contain"
              accessibilityRole="image"
              accessibilityLabel="LIRIE"
            />
          </View>

          <Text
            style={[
              styles.kicker,
              {
                fontSize: layout.secondaryFontSize,
                marginBottom: layout.cardLineGap,
                letterSpacing: viewport.isTiny ? 0.8 : 1.2,
              },
            ]}
            maxFontSizeMultiplier={1.22}
          >
            Espace client
          </Text>
          <Text
            style={[
              styles.title,
              {
                fontSize: layout.titleFontSize,
                lineHeight: layout.titleLineHeight,
                maxWidth: layout.titleMaxWidth,
              },
            ]}
            maxFontSizeMultiplier={1.28}
          >
            Connexion sécurisée
          </Text>
          <Text
            style={[
              styles.subtitle,
              {
                fontSize: layout.microProofFontSize,
                lineHeight: layout.microProofLineHeight,
                marginTop: layout.cardLineGap + 4,
              },
            ]}
            maxFontSizeMultiplier={1.45}
          >
            Suivi en temps réel · Coordination médicale · Transport accompagné.
          </Text>
          {params.next ? (
            <Text style={styles.resumeHint} maxFontSizeMultiplier={1.45}>
              Connectez-vous pour finaliser votre réservation en reprenant vos informations.
            </Text>
          ) : null}

          <View style={[styles.heroDivider, { marginTop: layout.cardBlockGap + 4 }]} />

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

          <View style={{ marginTop: layout.cardBlockGap + layout.cardLineGap }}>
            <AppInput
              label="Email"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
              autoComplete="email"
              textContentType="emailAddress"
              returnKeyType="next"
              onSubmitEditing={() => passwordInputRef.current?.focus()}
              placeholder="email@exemple.ch"
              {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
            />
          </View>

          <View style={{ marginTop: layout.cardBlockGap }}>
            <AppInput
              ref={passwordInputRef}
              label="Mot de passe"
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPassword}
              autoComplete="current-password"
              textContentType="password"
              returnKeyType="done"
              onSubmitEditing={() => void onSubmit()}
              placeholder="Mot de passe"
              {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
              rightSlot={
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
              }
            />
          </View>

          <AppSwitch
            value={rememberSession}
            onValueChange={setRememberSession}
            accessibilityLabel="Se souvenir de moi"
            style={styles.rememberRow}
            label={
              <AppText variant="bodyMuted" style={{ fontSize: layout.secondaryFontSize, fontWeight: "500" }}>
                Se souvenir de moi
              </AppText>
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

          <AppButton
            title="Se connecter"
            variant="primary"
            loading={submitting}
            disabled={updateGate.requiresUpdate}
            onPress={() => void onSubmit()}
            style={{ marginTop: layout.spaceCardToCta - 6, alignSelf: "stretch" }}
          />

          <View style={[styles.linksRow, { marginTop: layout.spaceCtaToProof + 4 }]}>
            <Pressable
              onPress={() => router.push("/(public)/forgot-password" as any)}
              style={({ pressed }) => [styles.linkHit, pressed && styles.linkHitPressed]}
            >
              <AppText variant="body" style={styles.primaryLink} maxFontSizeMultiplier={1.28}>
                Mot de passe oublié ?
              </AppText>
            </Pressable>
            <AppText
              variant="caption"
              style={[styles.linksDot, { fontSize: layout.secondaryFontSize }]}
              accessible={false}
            >
              ·
            </AppText>
            <Pressable
              onPress={() => router.push("/(public)/signup" as any)}
              style={({ pressed }) => [styles.linkHit, pressed && styles.linkHitPressed]}
            >
              <AppText
                variant="bodyMuted"
                style={[styles.secondaryLink, { fontSize: layout.secondaryFontSize }]}
                maxFontSizeMultiplier={1.28}
              >
                Créer un compte
              </AppText>
            </Pressable>
          </View>
          </View>
        </ResponsiveContainer>
      </Screen>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#F4FAF8",
  },
  backgroundImage: {
    opacity: 0.1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(244, 250, 248, 0.88)",
  },
  accentGlowLarge: {
    position: "absolute",
    width: 230,
    height: 230,
    borderRadius: 115,
    backgroundColor: "rgba(10, 143, 122, 0.11)",
    zIndex: 0,
  },
  accentGlowSmall: {
    position: "absolute",
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: "rgba(10, 143, 122, 0.08)",
    zIndex: 0,
  },
  /** Contenu formulaire : centré verticalement, scroll si clavier / grande police. */
  screenScrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 24,
  },
  card: {
    width: "100%",
    alignSelf: "center",
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.42)",
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
    ...Platform.select({
      web: { boxShadow: "0 12px 40px rgba(22, 58, 52, 0.14)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.14,
        shadowRadius: 22,
        shadowOffset: { width: 0, height: 10 },
        elevation: 6,
      },
    }),
  },
  cardBrandAccent: {
    position: "absolute",
    left: 0,
    right: 0,
    top: 0,
    height: 4,
    backgroundColor: brandPrimary,
    opacity: 0.95,
  },
  backButtonBase: {
    alignSelf: "flex-start",
  },
  logoBlock: {
    alignItems: "center",
  },
  kicker: {
    color: brandPrimary,
    fontSize: 12,
    fontWeight: "600",
    letterSpacing: 1.2,
    textTransform: "uppercase",
    marginBottom: 6,
    textAlign: "center",
  },
  title: {
    fontFamily: "Philosopher_700Bold",
    color: brandText,
    textAlign: "center",
    letterSpacing: -0.3,
  },
  subtitle: {
    color: brandTextMuted,
    textAlign: "center",
  },
  heroDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(145, 165, 157, 0.35)",
    marginBottom: 2,
    alignSelf: "stretch",
  },
  resumeHint: {
    marginTop: 14,
    color: "#45655D",
    fontSize: 14,
    lineHeight: 20,
    textAlign: "center",
  },
  passwordToggle: {
    justifyContent: "center",
    paddingHorizontal: 6,
    paddingVertical: 4,
  },
  rememberRow: {
    marginTop: 18,
  },
  errorText: {
    marginTop: 14,
    color: "#B42318",
    fontWeight: "600",
  },
  linksRow: {
    marginTop: 22,
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    rowGap: 12,
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
    color: brandTextMuted,
    fontSize: 14,
    fontWeight: "700",
    opacity: 0.65,
    paddingHorizontal: 2,
  },
  primaryLink: {
    color: brandPrimary,
    fontWeight: "700",
    fontSize: 14,
  },
  secondaryLink: {
    color: "#45655D",
    fontWeight: "600",
    fontSize: 14,
  },
});
