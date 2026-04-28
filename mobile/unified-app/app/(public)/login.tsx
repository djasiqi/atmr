import {
  ActivityIndicator,
  ImageBackground,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import { useSession } from "../../src/core/sessionProvider";
import { resolveInitialRoute } from "../../src/core/navigation/resolveInitialRoute";
import { useRuntimeUpdateGate } from "../../src/core/version/useRuntimeUpdateGate";
import { getLastDraftId } from "../../src/core/public/preRequestDraft";
import { queueExternalIntentResume } from "../../src/core/navigation/externalIntent";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

type LoginApiError = {
  message?: string;
  reason?: string;
  details?: Record<string, unknown>;
};

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");

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

      <View style={styles.centerWrap}>
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

          <Text style={styles.kicker}>Espace client</Text>
          <Text style={styles.title}>Connexion sécurisée</Text>
          <Text style={styles.subtitle}>
            Suivi en temps réel · Coordination médicale · Transport accompagné.
          </Text>
          {params.next ? (
            <Text style={styles.resumeHint}>
              Connectez-vous pour finaliser votre réservation en reprenant vos informations.
            </Text>
          ) : null}

          {updateGate.requiresUpdate ? (
            <View style={[styles.notice, styles.noticeDanger]}>
              <Text style={styles.noticeDangerTitle}>Mise à jour obligatoire</Text>
              <Text style={styles.noticeDangerText}>
                Version minimum requise: {updateGate.minimumSupportedVersion ?? "n/a"}
              </Text>
              <Pressable onPress={() => void updateGate.applyUpdate()} style={styles.noticeDangerCta}>
                <Text style={styles.noticeDangerCtaText}>Mettre à jour pour continuer</Text>
              </Pressable>
            </View>
          ) : null}

          <View style={styles.fieldBlock}>
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
              placeholder="email@exemple.ch"
              placeholderTextColor="#91A59D"
              style={styles.fieldInput}
            />
          </View>

          <View style={styles.fieldBlock}>
            <Text style={styles.fieldLabel}>Mot de passe</Text>
            <View style={styles.passwordWrap}>
              <TextInput
                ref={passwordInputRef}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                autoComplete="current-password"
                textContentType="password"
                returnKeyType="done"
                onSubmitEditing={() => void onSubmit()}
                placeholder="Mot de passe"
                placeholderTextColor="#91A59D"
                style={[styles.fieldInput, styles.passwordInput]}
              />
              <Pressable onPress={() => setShowPassword((v: boolean) => !v)} style={styles.passwordToggle}>
                <Ionicons
                  name={showPassword ? "eye-off-outline" : "eye-outline"}
                  size={20}
                  color="#5F7369"
                  accessibilityLabel={showPassword ? "Masquer le mot de passe" : "Afficher le mot de passe"}
                />
              </Pressable>
            </View>
          </View>

          <Pressable
            onPress={() => setRememberSession((v: boolean) => !v)}
            style={styles.rememberRow}
            accessibilityRole="switch"
            accessibilityState={{ checked: rememberSession }}
            accessibilityLabel="Se souvenir de moi"
          >
            <View style={[styles.rememberSwitch, rememberSession ? styles.rememberSwitchOn : null]}>
              <View style={[styles.rememberThumb, rememberSession ? styles.rememberThumbOn : null]} />
            </View>
            <Text style={styles.rememberText}>Se souvenir de moi</Text>
          </Pressable>

          {(localError || error) ? <Text style={styles.errorText}>{localError ?? error}</Text> : null}

          {!updateGate.requiresUpdate && (updateGate.updateAvailable || updateGate.recommendedUpdate) ? (
            <View style={[styles.notice, styles.noticeWarning]}>
              <Text style={styles.noticeWarningText}>
                Une mise à jour est disponible et recommandée (cible:{" "}
                {updateGate.recommendedVersion ?? "latest"}).
              </Text>
              <Pressable onPress={() => void updateGate.applyUpdate()} style={styles.noticeWarningCta}>
                <Text style={styles.noticeWarningCtaText}>Mettre à jour maintenant</Text>
              </Pressable>
            </View>
          ) : null}

          <Pressable
            onPress={() => void onSubmit()}
            disabled={submitting || updateGate.requiresUpdate}
            style={[
              styles.submitButton,
              submitting || updateGate.requiresUpdate ? styles.submitButtonDisabled : null,
            ]}
          >
            {submitting ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.submitText}>Se connecter</Text>
            )}
          </Pressable>

          <View style={styles.linksBlock}>
            <Pressable onPress={() => router.push("/(public)/forgot-password" as any)}>
              <Text style={styles.primaryLink}>Mot de passe oublié ?</Text>
            </Pressable>
            <Pressable onPress={() => router.push("/(public)/signup" as any)}>
              <Text style={styles.secondaryLink}>Créer un compte</Text>
            </Pressable>
          </View>
        </View>
      </View>
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
  centerWrap: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
    paddingVertical: 30,
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
    fontFamily: "Philosopher_700Bold",
    color: "#163A34",
    fontSize: 30,
    lineHeight: 34,
  },
  subtitle: {
    color: "#5F7369",
    fontSize: 15,
    lineHeight: 21,
    marginTop: 10,
  },
  resumeHint: {
    marginTop: 12,
    color: "#45655D",
    fontSize: 13.5,
    lineHeight: 19,
  },
  notice: {
    borderRadius: 14,
    borderWidth: 1,
    padding: 12,
    gap: 8,
    marginTop: 18,
  },
  noticeDanger: {
    borderColor: "#D92D20",
    backgroundColor: "#FFF1F1",
  },
  noticeDangerTitle: {
    color: "#7A0012",
    fontWeight: "700",
  },
  noticeDangerText: {
    color: "#7A0012",
  },
  noticeDangerCta: {
    alignItems: "center",
    justifyContent: "center",
    minHeight: 42,
    borderRadius: 12,
    backgroundColor: "#B00020",
  },
  noticeDangerCtaText: {
    color: "#FFFFFF",
    fontWeight: "700",
  },
  fieldBlock: {
    marginTop: 18,
    gap: 8,
  },
  fieldLabel: {
    color: "#5F7369",
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.2,
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
  passwordWrap: {
    position: "relative",
  },
  passwordInput: {
    paddingRight: 92,
  },
  passwordToggle: {
    position: "absolute",
    right: 12,
    top: 0,
    bottom: 0,
    justifyContent: "center",
    paddingHorizontal: 6,
  },
  rememberRow: {
    marginTop: 18,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  rememberSwitch: {
    width: 38,
    height: 22,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#B7C7C2",
    backgroundColor: "#E6EEEB",
    justifyContent: "center",
    paddingHorizontal: 2,
  },
  rememberSwitchOn: {
    backgroundColor: "#0A8F7A",
    borderColor: "#0A8F7A",
  },
  rememberThumb: {
    width: 16,
    height: 16,
    borderRadius: 999,
    backgroundColor: "#FFFFFF",
    transform: [{ translateX: 0 }],
  },
  rememberThumbOn: {
    transform: [{ translateX: 16 }],
  },
  rememberText: {
    color: "#5F7369",
    fontSize: 14,
    fontWeight: "500",
  },
  errorText: {
    marginTop: 14,
    color: "#B42318",
    fontWeight: "600",
  },
  noticeWarning: {
    borderColor: "#E0B86C",
    backgroundColor: "#FFF7E6",
  },
  noticeWarningText: {
    color: "#6A5320",
    fontWeight: "600",
    lineHeight: 20,
  },
  noticeWarningCta: {
    alignItems: "center",
    justifyContent: "center",
    minHeight: 42,
    borderRadius: 12,
    backgroundColor: "#0A8F7A",
  },
  noticeWarningCtaText: {
    color: "#FFFFFF",
    fontWeight: "700",
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
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  linksBlock: {
    marginTop: 18,
    gap: 10,
  },
  primaryLink: {
    color: "#0A8F7A",
    fontWeight: "700",
  },
  secondaryLink: {
    color: "#45655D",
    fontWeight: "600",
  },
});
