import { useState } from "react";
import { Pressable, Text, TextInput, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { apiClient } from "../../src/core/api/client";

export default function MfaScreen() {
  const params = useLocalSearchParams<{ email?: string }>();
  const router = useRouter();
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
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 10 }}>
      <Text style={{ fontSize: 22, fontWeight: "700" }}>MFA</Text>
      <TextInput
        value={email}
        onChangeText={setEmail}
        placeholder="Email"
        autoCapitalize="none"
        keyboardType="email-address"
        autoComplete="email"
        textContentType="emailAddress"
        returnKeyType="next"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <TextInput
        value={code}
        onChangeText={(v) => setCode(v.replace(/[^\d]/g, "").slice(0, 6))}
        placeholder="Code MFA"
        keyboardType="number-pad"
        maxLength={6}
        textContentType="oneTimeCode"
        autoComplete="one-time-code"
        returnKeyType="done"
        onSubmitEditing={() => void verify()}
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <Pressable
        onPress={() => void verify()}
        disabled={pending || email.trim().length === 0 || code.trim().length === 0}
        style={{
          borderWidth: 1,
          borderColor: "#0a7ea4",
          backgroundColor: "#0a7ea4",
          borderRadius: 10,
          padding: 12,
        }}
      >
        <Text style={{ color: "#fff", fontWeight: "600" }}>
          {pending ? "Verification..." : "Verifier"}
        </Text>
      </Pressable>
      {message ? <Text style={{ color: "#2e7d32" }}>{message}</Text> : null}
      {error ? <Text style={{ color: "#B00020" }}>{error}</Text> : null}
    </View>
  );
}
