import { useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { apiClient } from "../../src/core/api/client";
import { ResponsiveContainer, Screen, useAppViewport } from "../../src/design/responsive";

type ResetApiError = {
  response?: {
    data?: {
      error?: string;
      message?: string;
      reason?: string;
    };
  };
};

export default function ResetPasswordScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ token?: string }>();
  const token = useMemo(() => String(params.token ?? "").trim(), [params.token]);
  const { topInset } = useAppViewport();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [pending, setPending] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const confirmRef = useRef<TextInput | null>(null);

  const submit = async () => {
    if (!token) {
      setError("Lien de réinitialisation invalide.");
      return;
    }
    if (password.length < 8) {
      setError("Le mot de passe doit contenir au moins 8 caractères.");
      return;
    }
    if (confirmPassword !== password) {
      setError("Les mots de passe ne correspondent pas.");
      return;
    }
    setPending(true);
    setSuccess(null);
    setError(null);
    try {
      await apiClient.post("/auth/reset-password", {
        token,
        new_password: password,
      });
      setSuccess("Mot de passe réinitialisé. Vous pouvez vous connecter.");
      setPassword("");
      setConfirmPassword("");
    } catch (rawError) {
      const apiError = rawError as ResetApiError;
      const reason = String(apiError.response?.data?.reason ?? "");
      if (reason === "password_reset_token_expired") {
        setError("Le lien a expiré. Demandez un nouveau lien.");
      } else if (reason === "password_reset_token_invalid") {
        setError("Le lien est invalide ou déjà utilisé.");
      } else {
        setError(
          apiError.response?.data?.error ??
            apiError.response?.data?.message ??
            "Réinitialisation impossible. Réessayez."
        );
      }
    } finally {
      setPending(false);
    }
  };

  return (
    <Screen
      scroll
      backgroundColor="#EAF3F1"
      keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
      contentContainerStyle={styles.scrollContent}
    >
      <ResponsiveContainer>
        <View style={styles.card}>
          <Text style={styles.title}>Nouveau mot de passe</Text>
          <Text style={styles.subtitle}>
            Saisissez votre nouveau mot de passe pour finaliser la réinitialisation.
          </Text>
          <TextInput
            value={password}
            onChangeText={setPassword}
            placeholder="Nouveau mot de passe"
            placeholderTextColor="#91A59D"
            secureTextEntry={!showPassword}
            textContentType="newPassword"
            autoComplete="new-password"
            returnKeyType="next"
            onSubmitEditing={() => confirmRef.current?.focus()}
            style={styles.input}
          />
          <TextInput
            ref={confirmRef}
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            placeholder="Confirmer le mot de passe"
            placeholderTextColor="#91A59D"
            secureTextEntry={!showPassword}
            textContentType="newPassword"
            autoComplete="new-password"
            returnKeyType="done"
            onSubmitEditing={() => void submit()}
            style={styles.input}
          />
          <Pressable onPress={() => setShowPassword((v) => !v)} style={styles.toggleWrap}>
            <Text style={styles.toggle}>
              {showPassword ? "Masquer le mot de passe" : "Afficher le mot de passe"}
            </Text>
          </Pressable>
          {success ? <Text style={styles.success}>{success}</Text> : null}
          {error ? <Text style={styles.err}>{error}</Text> : null}
          <Pressable
            onPress={() => void submit()}
            disabled={pending || !token}
            style={[styles.primaryBtn, pending || !token ? styles.primaryBtnDisabled : null]}
          >
            {pending ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryBtnText}>Valider</Text>}
          </Pressable>
          <Pressable onPress={() => router.replace("/(public)/forgot-password" as any)}>
            <Text style={styles.link}>Demander un nouveau lien</Text>
          </Pressable>
          <Pressable onPress={() => router.replace("/(public)/login" as any)}>
            <Text style={styles.link}>Retour connexion</Text>
          </Pressable>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 24,
  },
  card: {
    gap: 12,
    borderRadius: 26,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: "#163A34",
  },
  subtitle: {
    color: "#5F7369",
    lineHeight: 21,
  },
  input: {
    borderWidth: 1,
    borderColor: "#91A59D",
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
    color: "#163A34",
    minHeight: 50,
  },
  toggleWrap: {
    alignSelf: "flex-start",
  },
  toggle: {
    color: "#0A8F7A",
    fontWeight: "600",
  },
  success: {
    color: "#2e7d32",
    fontWeight: "600",
  },
  err: {
    color: "#B00020",
    fontWeight: "600",
  },
  primaryBtn: {
    marginTop: 8,
    minHeight: 48,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
  },
  primaryBtnDisabled: {
    backgroundColor: "#84B7AE",
  },
  primaryBtnText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 16,
  },
  link: {
    color: "#0A8F7A",
    fontWeight: "600",
    marginTop: 4,
  },
});
