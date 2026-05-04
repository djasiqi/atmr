import { useState } from "react";
import {
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

export default function MfaScreen() {
  const params = useLocalSearchParams<{ email?: string }>();
  const router = useRouter();
  const { topInset } = useAppViewport();
  const [email, setEmail] = useState(params.email ?? "");
  const [code, setCode] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const verify = async () => {
    setPending(true);
    setMessage(null);
    setError(null);
    try {
      await apiClient.post("/auth/mfa/verify", {
        email: email.trim(),
        code: code.trim(),
      });
      setMessage("Verification MFA reussie.");
      router.replace("/(public)/login" as any);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Echec de verification MFA.");
    } finally {
      setPending(false);
    }
  };

  return (
    <Screen
      scroll
      backgroundColor="#EAF3F1"
      keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
      contentContainerStyle={styles.scroll}
    >
      <ResponsiveContainer>
        <View style={styles.card}>
          <Text style={styles.title}>Vérification MFA</Text>
          <Text style={styles.intro}>Saisissez le code à usage unique reçu par e-mail.</Text>
          <Text style={styles.label}>E-mail</Text>
          <TextInput
            value={email}
            onChangeText={setEmail}
            placeholder="email@exemple.ch"
            placeholderTextColor="#94a3b8"
            autoCapitalize="none"
            keyboardType="email-address"
            autoComplete="email"
            textContentType="emailAddress"
            returnKeyType="next"
            style={styles.input}
          />
          <Text style={styles.label}>Code à 6 chiffres</Text>
          <TextInput
            value={code}
            onChangeText={(v) => setCode(v.replace(/[^\d]/g, "").slice(0, 6))}
            placeholder="000000"
            placeholderTextColor="#94a3b8"
            keyboardType="number-pad"
            maxLength={6}
            textContentType="oneTimeCode"
            autoComplete="one-time-code"
            returnKeyType="done"
            onSubmitEditing={() => void verify()}
            style={styles.input}
          />
          <Pressable
            onPress={() => void verify()}
            disabled={pending || email.trim().length === 0 || code.trim().length === 0}
            style={({ pressed }) => [
              styles.primaryButton,
              (pending || email.trim().length === 0 || code.trim().length === 0) &&
                styles.primaryButtonDisabled,
              pressed && styles.primaryButtonPressed,
            ]}
          >
            <Text style={styles.primaryButtonText}>
              {pending ? "Vérification…" : "Vérifier"}
            </Text>
          </Pressable>
          {message ? <Text style={styles.success}>{message}</Text> : null}
          {error ? <Text style={styles.error}>{error}</Text> : null}
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    paddingVertical: 24,
    justifyContent: "center",
  },
  card: {
    width: "100%",
    borderRadius: 22,
    padding: 22,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    gap: 12,
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#163A34",
  },
  intro: {
    fontSize: 15,
    lineHeight: 22,
    color: "#5F7369",
    marginBottom: 4,
  },
  label: {
    fontSize: 13,
    fontWeight: "600",
    color: "#5F7369",
  },
  input: {
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.55)",
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
    fontSize: 16,
    color: "#163A34",
    backgroundColor: "#F7FBFA",
  },
  primaryButton: {
    marginTop: 8,
    minHeight: 50,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
    paddingHorizontal: 20,
  },
  primaryButtonDisabled: {
    backgroundColor: "#84B7AE",
  },
  primaryButtonPressed: {
    opacity: 0.92,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "700",
  },
  success: {
    color: "#166534",
    fontSize: 14,
  },
  error: {
    color: "#B42318",
    fontSize: 14,
  },
});
