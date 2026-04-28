import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { apiClient } from "../../src/core/api/client";

const DEFAULT_COOLDOWN = 60;

type ActivationStatus = {
  email_verified: boolean;
  phone_verified: boolean;
  is_complete: boolean;
  is_finalized: boolean;
};

function parseErrorMessage(e: unknown): string {
  const ax = e as any;
  return (
    ax?.response?.data?.error_message ||
    ax?.response?.data?.message ||
    ax?.response?.data?.error ||
    (e instanceof Error ? e.message : null) ||
    "Une erreur s'est produite."
  );
}

export default function ActivateScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    activation_session_id?: string;
    masked_email?: string;
    masked_phone?: string;
    token?: string;
    next?: string;
  }>();

  const [sessionId, setSessionId] = useState((params.activation_session_id ?? "").trim());
  const nextRoute = (params.next ?? "").trim();
  const activationToken = (params.token ?? "").trim();
  const [maskedEmail, setMaskedEmail] = useState(params.masked_email ?? "");
  const [maskedPhone, setMaskedPhone] = useState(params.masked_phone ?? "");

  const [status, setStatus] = useState<ActivationStatus>({
    email_verified: false,
    phone_verified: false,
    is_complete: false,
    is_finalized: false,
  });

  const [smsCode, setSmsCode] = useState("");
  const [newPhone, setNewPhone] = useState("");
  const [showPhoneEdit, setShowPhoneEdit] = useState(false);

  const [loading, setLoading] = useState(false);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [debugSmsCode, setDebugSmsCode] = useState<string | null>(null);
  const [debugActivationLink, setDebugActivationLink] = useState<string | null>(null);

  const [emailCooldown, setEmailCooldown] = useState(0);
  const [smsCooldown, setSmsCooldown] = useState(0);

  const cooldownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tokenHandledRef = useRef(false);

  useEffect(() => {
    if (emailCooldown > 0 || smsCooldown > 0) {
      if (cooldownRef.current) clearInterval(cooldownRef.current);
      cooldownRef.current = setInterval(() => {
        setEmailCooldown((v) => (v > 0 ? v - 1 : 0));
        setSmsCooldown((v) => (v > 0 ? v - 1 : 0));
      }, 1000);
    } else if (cooldownRef.current) {
      clearInterval(cooldownRef.current);
    }
    return () => {
      if (cooldownRef.current) clearInterval(cooldownRef.current);
    };
  }, [emailCooldown, smsCooldown]);

  const clearMessages = () => {
    setInfoMsg(null);
    setErrorMsg(null);
    setDebugSmsCode(null);
    setDebugActivationLink(null);
  };

  const applyStatus = (raw: any) => {
    if (!raw || typeof raw !== "object") return;
    setStatus({
      email_verified: Boolean(raw.email_verified),
      phone_verified: Boolean(raw.phone_verified),
      is_complete: Boolean(raw.is_complete),
      is_finalized: Boolean(raw.is_finalized),
    });
  };

  const fetchStatus = async (overrideSessionId?: string) => {
    const targetSessionId = (overrideSessionId ?? sessionId).trim();
    if (!targetSessionId) return;
    const res = await apiClient.get(
      `/auth/activation/status?activation_session_id=${encodeURIComponent(targetSessionId)}`
    );
    if (res.data?.activation_session_id) {
      setSessionId(String(res.data.activation_session_id));
    }
    if (res.data?.masked_email) setMaskedEmail(res.data.masked_email);
    if (res.data?.masked_phone) setMaskedPhone(res.data.masked_phone);
    applyStatus(res.data?.activation_status);
  };

  const verifyEmailToken = async () => {
    if (!activationToken) return;
    const res = await apiClient.post("/auth/activation/verify-email", {
      token: activationToken,
    });
    const nextSessionId = String(res.data?.activation_session_id ?? "").trim();
    if (nextSessionId) {
      setSessionId(nextSessionId);
    }
    if (res.data?.masked_email) setMaskedEmail(res.data.masked_email);
    if (res.data?.masked_phone) setMaskedPhone(res.data.masked_phone);
    applyStatus(res.data?.activation_status);
  };

  // Charge le statut initial
  useEffect(() => {
    if (!sessionId && !activationToken) return;
    setLoading(true);
    (async () => {
      if (activationToken && !tokenHandledRef.current) {
        tokenHandledRef.current = true;
        await verifyEmailToken();
      }
      await fetchStatus();
    })()
      .catch((e) => setErrorMsg(parseErrorMessage(e)))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, activationToken]);

  const handleResendEmail = async () => {
    if (!sessionId) return;
    clearMessages();
    setLoading(true);
    try {
      const res = await apiClient.post("/auth/activation/resend-email", {
        activation_session_id: sessionId,
      });
      setEmailCooldown(DEFAULT_COOLDOWN);
      const link = String(res.data?.debug_activation_link ?? "").trim();
      if (link) {
        setDebugActivationLink(link);
        setInfoMsg("Service email indisponible (dev). Utilisez le lien ci-dessous.");
      } else {
        setInfoMsg("Email renvoyé. Vérifiez votre boîte de réception.");
      }
    } catch (e: unknown) {
      const retryAfter = (e as any)?.response?.data?.details?.retry_after_seconds;
      if (retryAfter > 0) setEmailCooldown(Number(retryAfter));
      setErrorMsg(parseErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const handleResendSms = async () => {
    if (!sessionId) return;
    clearMessages();
    setLoading(true);
    try {
      const res = await apiClient.post("/auth/activation/resend-sms", {
        activation_session_id: sessionId,
      });
      setSmsCooldown(DEFAULT_COOLDOWN);
      const code = String(res.data?.debug_sms_code ?? "").trim();
      if (code) {
        setDebugSmsCode(code);
        setInfoMsg("Service SMS indisponible (dev). Code de secours ci-dessous.");
      } else {
        setInfoMsg("Code SMS renvoyé.");
      }
    } catch (e: unknown) {
      const retryAfter = (e as any)?.response?.data?.details?.retry_after_seconds;
      if (retryAfter > 0) setSmsCooldown(Number(retryAfter));
      setErrorMsg(parseErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const handleVerifySms = async () => {
    if (!sessionId) return;
    const code = smsCode.trim();
    if (!/^\d{6}$/.test(code)) {
      setErrorMsg("Entrez un code à 6 chiffres.");
      return;
    }
    clearMessages();
    setLoading(true);
    try {
      const res = await apiClient.post("/auth/activation/verify-sms", {
        activation_session_id: sessionId,
        code,
      });
      applyStatus(res.data?.activation_status);
      setInfoMsg("Téléphone confirmé.");
      setSmsCode("");
    } catch (e: unknown) {
      setErrorMsg(parseErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const handleUpdatePhone = async () => {
    if (!sessionId) return;
    const phone = newPhone.trim();
    if (phone.length < 7) {
      setErrorMsg("Numéro invalide (min. 7 caractères).");
      return;
    }
    clearMessages();
    setLoading(true);
    try {
      const res = await apiClient.post("/auth/activation/update-phone", {
        activation_session_id: sessionId,
        phone,
      });
      if (res.data?.masked_phone) setMaskedPhone(res.data.masked_phone);
      applyStatus(res.data?.activation_status);
      setShowPhoneEdit(false);
      setNewPhone("");
      setSmsCooldown(DEFAULT_COOLDOWN);
      const code = String(res.data?.debug_sms_code ?? "").trim();
      if (code) {
        setDebugSmsCode(code);
        setInfoMsg("Numéro mis à jour. Code de secours (dev) ci-dessous.");
      } else {
        setInfoMsg(res.data?.message ?? "Numéro mis à jour. Nouveau code envoyé.");
      }
    } catch (e: unknown) {
      const retryAfter = (e as any)?.response?.data?.details?.retry_after_seconds;
      if (retryAfter > 0) setSmsCooldown(Number(retryAfter));
      setErrorMsg(parseErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const handleFinalize = async () => {
    if (!sessionId) return;
    clearMessages();
    setLoading(true);
    try {
      const res = await apiClient.post("/auth/activation/finalize", {
        activation_session_id: sessionId,
      });
      applyStatus(res.data?.activation_status);
      setInfoMsg("Compte activé ! Vous pouvez maintenant vous connecter.");
    } catch (e: unknown) {
      setErrorMsg(parseErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId && !activationToken) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", padding: 24, gap: 12 }}>
        <Text style={{ fontSize: 16, color: "#B00020", textAlign: "center" }}>
          Session d&apos;activation introuvable. Recommencez l&apos;inscription.
        </Text>
        <Pressable onPress={() => router.replace("/(public)/signup" as any)}>
          <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Retour à l&apos;inscription</Text>
        </Pressable>
      </View>
    );
  }

  if (status.is_finalized) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", padding: 24, gap: 16 }}>
        <Text style={{ fontSize: 22, fontWeight: "700" }}>Compte activé</Text>
        <Text style={{ color: "#2e7d32", textAlign: "center" }}>
          Votre compte est actif. Vous pouvez vous connecter.
        </Text>
        <Pressable
          onPress={() =>
            router.replace({
              pathname: "/(public)/login",
              ...(nextRoute ? { params: { next: nextRoute } } : {}),
            } as any)
          }
          style={{
            backgroundColor: "#0a7ea4",
            borderRadius: 10,
            paddingVertical: 14,
            paddingHorizontal: 32,
          }}
        >
          <Text style={{ color: "#fff", fontWeight: "700", fontSize: 15 }}>Se connecter</Text>
        </Pressable>
      </View>
    );
  }

  const finalizeEnabled = status.email_verified && status.phone_verified && !status.is_finalized;

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 24 }}>
      <View style={{ gap: 20 }}>
        <Text style={{ fontSize: 22, fontWeight: "700" }}>Activation du compte</Text>
        <Text style={{ color: "#555" }}>
          Confirmez votre email et votre téléphone pour activer votre compte.
        </Text>

        {/* Messages */}
        {infoMsg ? (
          <View style={{ backgroundColor: "#e8f5e9", borderRadius: 8, padding: 12 }}>
            <Text style={{ color: "#1b5e20", fontSize: 13 }}>{infoMsg}</Text>
          </View>
        ) : null}
        {debugActivationLink ? (
          <View style={{ backgroundColor: "#fff3e0", borderRadius: 8, padding: 12 }}>
            <Text style={{ color: "#e65100", fontSize: 13, fontWeight: "600" }}>Lien d&apos;activation (dev) :</Text>
            <Text style={{ color: "#e65100", fontSize: 12, marginTop: 4 }} selectable>
              {debugActivationLink}
            </Text>
          </View>
        ) : null}
        {debugSmsCode ? (
          <View style={{ backgroundColor: "#fff3e0", borderRadius: 8, padding: 12 }}>
            <Text style={{ color: "#e65100", fontSize: 13 }}>
              Code SMS de secours (dev) : <Text style={{ fontWeight: "700" }}>{debugSmsCode}</Text>
            </Text>
          </View>
        ) : null}
        {errorMsg ? (
          <View style={{ backgroundColor: "#ffebee", borderRadius: 8, padding: 12 }}>
            <Text style={{ color: "#B00020", fontSize: 13 }}>{errorMsg}</Text>
          </View>
        ) : null}

        {/* Section Email */}
        <View style={{ gap: 10 }}>
          <Text style={{ fontSize: 16, fontWeight: "600" }}>Email</Text>
          <Text style={{ color: "#555", fontSize: 13 }}>
            Un lien a été envoyé à {maskedEmail || "votre adresse email"}.
          </Text>
          <View
            style={{
              backgroundColor: status.email_verified ? "#e8f5e9" : "#fff8e1",
              borderRadius: 6,
              paddingHorizontal: 10,
              paddingVertical: 4,
              alignSelf: "flex-start",
            }}
          >
            <Text style={{ color: status.email_verified ? "#2e7d32" : "#f57f17", fontWeight: "600", fontSize: 12 }}>
              {status.email_verified ? "Email confirmé" : "En attente de confirmation"}
            </Text>
          </View>
          <Pressable
            onPress={() => void handleResendEmail()}
            disabled={loading || status.email_verified || emailCooldown > 0}
            style={{
              borderWidth: 1,
              borderColor: "#0a7ea4",
              borderRadius: 8,
              padding: 10,
              alignItems: "center",
              opacity: loading || status.email_verified || emailCooldown > 0 ? 0.5 : 1,
            }}
          >
            <Text style={{ color: "#0a7ea4", fontWeight: "600", fontSize: 13 }}>
              {emailCooldown > 0 ? `Renvoyer l'email (${emailCooldown}s)` : "Renvoyer l'email"}
            </Text>
          </Pressable>
        </View>

        {/* Séparateur */}
        <View style={{ height: 1, backgroundColor: "#e0e0e0" }} />

        {/* Section SMS */}
        <View style={{ gap: 10 }}>
          <Text style={{ fontSize: 16, fontWeight: "600" }}>Téléphone (SMS)</Text>
          <Text style={{ color: "#555", fontSize: 13 }}>
            Saisissez le code reçu sur {maskedPhone || "votre téléphone"}.
          </Text>

          {/* Modifier numéro */}
          {!status.phone_verified ? (
            <Pressable onPress={() => { setShowPhoneEdit((v) => !v); clearMessages(); }}>
              <Text style={{ color: "#0a7ea4", fontSize: 13 }}>
                {showPhoneEdit ? "Annuler" : "Mauvais numéro ?"}
              </Text>
            </Pressable>
          ) : null}
          {showPhoneEdit && !status.phone_verified ? (
            <View style={{ flexDirection: "row", gap: 8 }}>
              <TextInput
                value={newPhone}
                onChangeText={setNewPhone}
                placeholder="+41..."
                keyboardType="phone-pad"
                style={{
                  flex: 1,
                  borderWidth: 1,
                  borderColor: "#ddd",
                  borderRadius: 8,
                  padding: 10,
                  fontSize: 14,
                }}
              />
              <Pressable
                onPress={() => void handleUpdatePhone()}
                disabled={loading}
                style={{
                  backgroundColor: "#0a7ea4",
                  borderRadius: 8,
                  paddingHorizontal: 14,
                  justifyContent: "center",
                  opacity: loading ? 0.5 : 1,
                }}
              >
                <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>Mettre à jour</Text>
              </Pressable>
            </View>
          ) : null}

          <View
            style={{
              backgroundColor: status.phone_verified ? "#e8f5e9" : "#fff8e1",
              borderRadius: 6,
              paddingHorizontal: 10,
              paddingVertical: 4,
              alignSelf: "flex-start",
            }}
          >
            <Text style={{ color: status.phone_verified ? "#2e7d32" : "#f57f17", fontWeight: "600", fontSize: 12 }}>
              {status.phone_verified ? "Téléphone confirmé" : "En attente de confirmation"}
            </Text>
          </View>

          {/* Saisie code */}
          {!status.phone_verified ? (
            <View style={{ flexDirection: "row", gap: 8 }}>
              <TextInput
                value={smsCode}
                onChangeText={(v) => setSmsCode(v.replace(/[^\d]/g, "").slice(0, 6))}
                placeholder="Code 6 chiffres"
                keyboardType="number-pad"
                maxLength={6}
                style={{
                  flex: 1,
                  borderWidth: 1,
                  borderColor: "#ddd",
                  borderRadius: 8,
                  padding: 10,
                  fontSize: 18,
                  letterSpacing: 4,
                  textAlign: "center",
                }}
              />
              <Pressable
                onPress={() => void handleVerifySms()}
                disabled={loading || smsCode.length < 6}
                style={{
                  backgroundColor: "#0a7ea4",
                  borderRadius: 8,
                  paddingHorizontal: 14,
                  justifyContent: "center",
                  opacity: loading || smsCode.length < 6 ? 0.5 : 1,
                }}
              >
                <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>Valider</Text>
              </Pressable>
            </View>
          ) : null}

          {!status.phone_verified ? (
            <Pressable
              onPress={() => void handleResendSms()}
              disabled={loading || smsCooldown > 0}
              style={{
                borderWidth: 1,
                borderColor: "#0a7ea4",
                borderRadius: 8,
                padding: 10,
                alignItems: "center",
                opacity: loading || smsCooldown > 0 ? 0.5 : 1,
              }}
            >
              <Text style={{ color: "#0a7ea4", fontWeight: "600", fontSize: 13 }}>
                {smsCooldown > 0 ? `Renvoyer le code (${smsCooldown}s)` : "Renvoyer le code"}
              </Text>
            </Pressable>
          ) : null}
        </View>

        {/* Séparateur */}
        <View style={{ height: 1, backgroundColor: "#e0e0e0" }} />

        {/* Bouton Finaliser */}
        <Pressable
          onPress={() => void handleFinalize()}
          disabled={!finalizeEnabled || loading}
          style={{
            backgroundColor: finalizeEnabled ? "#0a7ea4" : "#bbb",
            borderRadius: 10,
            padding: 14,
            alignItems: "center",
          }}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={{ color: "#fff", fontWeight: "700", fontSize: 15 }}>
              Activer mon compte
            </Text>
          )}
        </Pressable>

        <Pressable
          onPress={() =>
            router.replace({
              pathname: "/(public)/login",
              ...(nextRoute ? { params: { next: nextRoute } } : {}),
            } as any)
          }
        >
          <Text style={{ color: "#999", textAlign: "center", fontSize: 13 }}>
            Déjà activé ? Se connecter
          </Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}
