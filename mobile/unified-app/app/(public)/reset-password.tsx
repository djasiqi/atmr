import { useMemo, useRef, useState } from "react";
import { ActivityIndicator, Pressable, Text, TextInput, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { apiClient } from "../../src/core/api/client";

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
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "700" }}>Nouveau mot de passe</Text>
      <Text style={{ color: "#555" }}>
        Saisissez votre nouveau mot de passe pour finaliser la réinitialisation.
      </Text>
      <TextInput
        value={password}
        onChangeText={setPassword}
        placeholder="Nouveau mot de passe"
        secureTextEntry={!showPassword}
        textContentType="newPassword"
        autoComplete="new-password"
        returnKeyType="next"
        onSubmitEditing={() => confirmRef.current?.focus()}
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <TextInput
        ref={confirmRef}
        value={confirmPassword}
        onChangeText={setConfirmPassword}
        placeholder="Confirmer le mot de passe"
        secureTextEntry={!showPassword}
        textContentType="newPassword"
        autoComplete="new-password"
        returnKeyType="done"
        onSubmitEditing={() => void submit()}
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <Pressable onPress={() => setShowPassword((v) => !v)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>
          {showPassword ? "Masquer le mot de passe" : "Afficher le mot de passe"}
        </Text>
      </Pressable>
      {success ? <Text style={{ color: "#2e7d32" }}>{success}</Text> : null}
      {error ? <Text style={{ color: "#B00020" }}>{error}</Text> : null}
      <Pressable
        onPress={() => void submit()}
        disabled={pending || !token}
        style={{
          backgroundColor: pending || !token ? "#91b9c6" : "#0a7ea4",
          borderRadius: 10,
          padding: 12,
          alignItems: "center",
        }}
      >
        {pending ? <ActivityIndicator color="#fff" /> : <Text style={{ color: "#fff" }}>Valider</Text>}
      </Pressable>
      <Pressable onPress={() => router.replace("/(public)/forgot-password" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Demander un nouveau lien</Text>
      </Pressable>
      <Pressable onPress={() => router.replace("/(public)/login" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Retour connexion</Text>
      </Pressable>
    </View>
  );
}
