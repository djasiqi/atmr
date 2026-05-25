import { useState } from "react";
import {
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import {
  requestPasswordlessOtp,
  setAuthToken,
  verifyPasswordlessOtp,
} from "../../src/core/api/client";
import { ResponsiveContainer, Screen, useAppViewport } from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

export default function PasswordlessOtpScreen() {
  const router = useRouter();
  const { topInset } = useAppViewport();
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
    <Screen
      scroll
      backgroundColor="#EAF3F1"
      keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
      contentContainerStyle={styles.scroll}
    >
      <ResponsiveContainer>
        <View style={styles.card}>
          <Text style={styles.title}>Connexion par code (beta)</Text>
          <Text style={styles.lede}>
            Cette methode est separee de l&apos;activation de compte classique.
          </Text>
          <Pressable
            onPress={() => setChannel((value) => (value === "email" ? "phone" : "email"))}
            style={styles.channelBtn}
          >
            <Text style={styles.channelBtnText}>
              Canal: {channel === "email" ? "Email" : "Telephone"} (appuyer pour changer)
            </Text>
          </Pressable>
          <TextInput
            value={identifier}
            onChangeText={setIdentifier}
            placeholder={channel === "email" ? "Email" : "Telephone"}
            placeholderTextColor="#91A59D"
            autoCapitalize="none"
            style={styles.input}
          />
          <Pressable
            onPress={() => void requestCode()}
            disabled={pending}
            style={[styles.primaryBtn, pending && styles.primaryBtnDisabled]}
          >
            <Text style={styles.primaryBtnText}>{pending ? "Envoi..." : "Recevoir un code"}</Text>
          </Pressable>

          <TextInput
            value={otpSessionId}
            onChangeText={setOtpSessionId}
            placeholder="OTP session id"
            placeholderTextColor="#91A59D"
            autoCapitalize="none"
            style={styles.input}
          />
          <TextInput
            value={code}
            onChangeText={(value) => setCode(value.replace(/[^\d]/g, "").slice(0, 6))}
            placeholder="Code a 6 chiffres"
            placeholderTextColor="#91A59D"
            keyboardType="number-pad"
            style={styles.input}
          />
          <Pressable
            onPress={() => void verifyCode()}
            disabled={pending}
            style={styles.outlineBtn}
          >
            <Text style={styles.outlineBtnText}>{pending ? "Verification..." : "Verifier le code"}</Text>
          </Pressable>
          {debugCode ? <Text style={styles.debug}>Code dev: {debugCode}</Text> : null}
          {message ? <Text style={styles.success}>{message}</Text> : null}
          {error ? <Text style={styles.err}>{error}</Text> : null}
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
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
    fontSize: FONT_SIZE.px22,
    fontWeight: "700",
    color: "#163A34",
  },
  lede: {
    color: "#5F7369",
    lineHeight: 22,
  },
  channelBtn: {
    borderWidth: 1,
    borderColor: "#91A59D",
    borderRadius: 14,
    padding: 12,
  },
  channelBtnText: {
    color: "#163A34",
    fontSize: FONT_SIZE.px14,
  },
  input: {
    borderWidth: 1,
    borderColor: "#91A59D",
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: FONT_SIZE.px16,
    color: "#163A34",
    minHeight: 48,
  },
  primaryBtn: {
    borderRadius: 14,
    backgroundColor: "#0A8F7A",
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryBtnDisabled: {
    backgroundColor: "#84B7AE",
  },
  primaryBtnText: {
    color: "#FFFFFF",
    fontWeight: "700",
  },
  outlineBtn: {
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    paddingVertical: 14,
    alignItems: "center",
  },
  outlineBtnText: {
    color: "#0A8F7A",
    fontWeight: "700",
  },
  debug: {
    color: "#92400e",
    fontSize: FONT_SIZE.px13,
  },
  success: {
    color: "#0f5132",
    fontWeight: "600",
  },
  err: {
    color: "#b91c1c",
    fontWeight: "600",
  },
});
