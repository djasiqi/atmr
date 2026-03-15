import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { router } from "expo-router";
import Ionicons from "react-native-vector-icons/Ionicons";

import { useAuth } from "@/hooks/useAuth";
import { Loader } from "@/components/ui/Loader";
import { getLoginStyles } from "@/styles/loginStyles";
import {
  getRememberMe,
  setRememberMe as persistRememberMe,
  getRememberedCredentials,
  clearRememberedCredentials,
} from "@/utils/rememberMeStorage";
import { consumeLogoutMarker } from "@/services/logoutMarker";
import { SessionExpiredBanner } from "@/components/common/SessionExpiredBanner";
import { useAppAlert } from "@/contexts/AppAlertContext";

export default function EnterpriseLoginScreen() {
  const { loginEnterprise, enterpriseLoading, setMode } = useAuth();
  const appAlert = useAppAlert();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [sessionExpiredMarker, setSessionExpiredMarker] = useState<{
    reason: string;
    ts: number;
  } | null>(null);
  const { styles, palette } = useMemo(() => getLoginStyles("enterprise"), []);

  useEffect(() => {
    let cancelled = false;
    consumeLogoutMarker("enterprise").then((marker) => {
      if (!cancelled && marker) {
        setSessionExpiredMarker({ reason: marker.reason, ts: marker.ts });
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        const rm = await getRememberMe("enterprise");
        if (!isMounted) return;
        if (rm) {
          const creds = await getRememberedCredentials("enterprise");
          if (!isMounted) return;
          if (creds?.email && creds?.password) {
            setEmail(creds.email);
            setPassword(creds.password);
            setRememberMe(true);
          } else {
            await persistRememberMe(false, "enterprise");
            await clearRememberedCredentials("enterprise");
            setRememberMe(false);
          }
        } else {
          setRememberMe(false);
        }
      } catch {
        if (isMounted) setRememberMe(false);
        void clearRememberedCredentials("enterprise").catch(() => {});
      } finally {
        if (isMounted) setHydrated(true);
      }
    })();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleToggleRememberMe = async () => {
    const next = !rememberMe;
    if (next) {
      try {
        await persistRememberMe(true, "enterprise");
        setRememberMe(true);
      } catch {
        setRememberMe(false);
        appAlert.showAlert("", "Impossible d'enregistrer sur cet appareil.");
      }
    } else {
      try {
        await persistRememberMe(false, "enterprise");
        setRememberMe(false);
      } catch {
        setRememberMe(false);
      }
    }
  };

  const handleSubmit = async () => {
    if (!email || !password) {
      appAlert.showAlert(
        "Information manquante",
        "Email et mot de passe sont requis."
      );
      return;
    }
    try {
      const result = await loginEnterprise({
        method: "password",
        email,
        password,
        rememberMe,
      });
      if (result.mfaRequired) {
        const isReviewerAccount = /review|reviewer/i.test(email);
        if (isReviewerAccount) {
          appAlert.showAlert(
            "Configuration reviewer attendue",
            "Ce compte reviewer ne devrait pas demander MFA. Vérifiez la configuration de la review company (mobile_mfa.required=false). Vous pouvez continuer vers MFA pour QA si nécessaire."
          );
        }
        router.replace({
          pathname: "/(enterprise-auth)/mfa",
          params: { challengeId: result.challenge.challengeId },
        } as any);
      } else {
        router.replace("/(enterprise)/dashboard" as any);
      }
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Connexion impossible. Vérifiez vos identifiants.";
      appAlert.showAlert("Échec connexion", message);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
      >
        <View style={styles.card}>
          <View style={styles.header}>
            <Text style={styles.kicker}>Espace Entreprise</Text>
            <Text style={styles.title}>Supervision des Courses</Text>
            <Text style={styles.subtitle}>
              Affectez, suivez et optimisez vos courses en toute simplicité.
            </Text>
          </View>

          <View style={styles.form}>
            {sessionExpiredMarker && (
              <SessionExpiredBanner marker={sessionExpiredMarker} />
            )}
            <View style={styles.inputBlock}>
              <Text style={styles.label}>Email Entreprise</Text>
              <TextInput
                style={styles.input}
                placeholder="entreprise@liri.ch"
                placeholderTextColor={palette.placeholder}
                keyboardType="email-address"
                autoCapitalize="none"
                value={email}
                onChangeText={setEmail}
              />
            </View>

            <View style={styles.inputBlock}>
              <Text style={styles.label}>Mot de passe</Text>
              <View style={styles.passwordField}>
                <TextInput
                  style={styles.input}
                  placeholder="Mot de passe"
                  placeholderTextColor={palette.placeholder}
                  secureTextEntry={!showPassword}
                  value={password}
                  onChangeText={setPassword}
                />
                <TouchableOpacity
                  style={styles.eyeButton}
                  onPress={() => setShowPassword((v) => !v)}
                >
                  <Ionicons
                    name={showPassword ? "eye" : "eye-off"}
                    size={22}
                    color={palette.secondary}
                  />
                </TouchableOpacity>
              </View>
            </View>

            <TouchableOpacity
              style={styles.rememberMeContainer}
              onPress={handleToggleRememberMe}
              activeOpacity={0.7}
              disabled={!hydrated}
            >
              <View style={[styles.checkbox, rememberMe && styles.checkboxChecked]}>
                {rememberMe && (
                  <Ionicons name="checkmark" size={14} color="#FFFFFF" />
                )}
              </View>
              <Text style={styles.checkboxLabel}>Se souvenir de moi</Text>
            </TouchableOpacity>
            <Text style={styles.rememberMeHint}>
              Stocké de manière sécurisée sur cet appareil (Keychain/Keystore).
            </Text>

            <TouchableOpacity
              style={styles.helperLink}
              onPress={() => router.push("/(enterprise-auth)/forgot-password")}
            >
              <Text style={styles.helperLinkText}>Mot de passe oublié ?</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.primaryButton}
              onPress={handleSubmit}
              disabled={enterpriseLoading}
            >
              {enterpriseLoading ? (
                <Loader />
              ) : (
                <Text style={styles.primaryButtonText}>Se connecter</Text>
              )}
            </TouchableOpacity>

            <View style={styles.switchRow}>
              <Text style={styles.switchPrompt}>Vous êtes chauffeur ?</Text>
              <TouchableOpacity
                onPress={async () => {
                  await setMode("driver");
                  router.replace("/(auth)/login");
                }}
              >
                <Text style={styles.switchLink}>
                  Accéder à l’espace chauffeur
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
