import { useState } from "react";
import { Pressable, ScrollView, Text, TextInput } from "react-native";
import { useRouter } from "expo-router";
import {
  requestPasswordlessOtp,
  setAuthToken,
  verifyPasswordlessOtp,
} from "../../src/core/api/client";

export default function PasswordlessOtpScreen() {
  const router = useRouter();
  const [channel, setChannel] = useState<"email" | "phone">("email");
  const [identifier, setIdentifier] = useState("");
  const [otpSessionId, setOtpSessionId] = useState("");
  const [code, setCode] = useState("");
  const [debugCode, setDebugCode] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const requestCode = async () => {
    if (!identifier.trim()) {
      setError("Identifiant requis.");
      return;
    }
    setPending(true);
    setError(null);
    setMessage(null);
    try {
      const response = await requestPasswordlessOtp({
        channel,
        identifier: identifier.trim(),
      });
      setOtpSessionId(response.otp_session_id);
      setDebugCode(response.debug_code ?? null);
      setMessage(
        `Code envoye sur ${response.masked_identifier}. Saisissez-le pour vous connecter.`
      );
    } catch (e: any) {
      setError(e?.message ?? "Impossible de demander un code.");
    } finally {
      setPending(false);
    }
  };

  const verifyCode = async () => {
    if (!otpSessionId.trim() || code.trim().length !== 6) {
      setError("Session OTP ou code invalide.");
      return;
    }
    setPending(true);
    setError(null);
    setMessage(null);
    try {
      const tokens = await verifyPasswordlessOtp({
        otp_session_id: otpSessionId.trim(),
        code: code.trim(),
      });
      setAuthToken(tokens.access_token);
      setMessage("Connexion reussie.");
      router.replace("/");
    } catch (e: any) {
      setError(e?.message ?? "Code invalide ou expire.");
    } finally {
      setPending(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "700" }}>Connexion par code (beta)</Text>
      <Text style={{ color: "#475569" }}>
        Cette methode est separee de l&apos;activation de compte classique.
      </Text>
      <Pressable
        onPress={() => setChannel((value) => (value === "email" ? "phone" : "email"))}
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      >
        <Text>Canal: {channel === "email" ? "Email" : "Telephone"} (appuyer pour changer)</Text>
      </Pressable>
      <TextInput
        value={identifier}
        onChangeText={setIdentifier}
        placeholder={channel === "email" ? "Email" : "Telephone"}
        autoCapitalize="none"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <Pressable
        onPress={() => void requestCode()}
        disabled={pending}
        style={{ borderRadius: 10, backgroundColor: pending ? "#9cb7c1" : "#0a7ea4", padding: 12, alignItems: "center" }}
      >
        <Text style={{ color: "#fff", fontWeight: "700" }}>{pending ? "Envoi..." : "Recevoir un code"}</Text>
      </Pressable>

      <TextInput
        value={otpSessionId}
        onChangeText={setOtpSessionId}
        placeholder="OTP session id"
        autoCapitalize="none"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <TextInput
        value={code}
        onChangeText={(value) => setCode(value.replace(/[^\d]/g, "").slice(0, 6))}
        placeholder="Code a 6 chiffres"
        keyboardType="number-pad"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <Pressable
        onPress={() => void verifyCode()}
        disabled={pending}
        style={{ borderRadius: 10, borderWidth: 1, borderColor: "#0a7ea4", padding: 12, alignItems: "center" }}
      >
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>{pending ? "Verification..." : "Verifier le code"}</Text>
      </Pressable>
      {debugCode ? <Text style={{ color: "#92400e" }}>Code dev: {debugCode}</Text> : null}
      {message ? <Text style={{ color: "#0f5132" }}>{message}</Text> : null}
      {error ? <Text style={{ color: "#b91c1c" }}>{error}</Text> : null}
    </ScrollView>
  );
}
