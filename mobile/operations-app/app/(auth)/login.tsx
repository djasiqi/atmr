// src/app/(auth)/login.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  Alert,
  TextInput,
  ActivityIndicator,
} from "react-native";
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
  RememberMeStorageError,
} from "@/utils/rememberMeStorage";
import { consumeLogoutMarker } from "@/services/logoutMarker";
import { SessionExpiredBanner } from "@/components/common/SessionExpiredBanner";

export default function LoginScreen() {
  const { login, loading, setMode } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [sessionExpiredMarker, setSessionExpiredMarker] = useState<{
    reason: string;
    ts: number;
  } | null>(null);
  const { styles, palette } = useMemo(() => getLoginStyles("driver"), []);

  useEffect(() => {
    let cancelled = false;
    consumeLogoutMarker("driver").then((marker) => {
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
        const rm = await getRememberMe();
        if (!isMounted) return;
        if (rm) {
          const creds = await getRememberedCredentials();
          if (!isMounted) return;
          if (creds?.email && creds?.password) {
            setEmail(creds.email);
            setPassword(creds.password);
            setRememberMe(true);
          } else {
            await persistRememberMe(false);
            await clearRememberedCredentials();
            setRememberMe(false);
          }
        } else {
          setRememberMe(false);
        }
      } catch {
        if (isMounted) setRememberMe(false);
        void clearRememberedCredentials().catch(() => { });
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
        await persistRememberMe(true);
        setRememberMe(true);
      } catch {
        setRememberMe(false);
        Alert.alert("", "Impossible d'enregistrer sur cet appareil.");
      }
    } else {
      try {
        await persistRememberMe(false);
        setRememberMe(false);
      } catch {
        setRememberMe(false);
      }
    }
  };

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert(
        "Erreur",
        "Veuillez entrer votre email et votre mot de passe."
      );
      return;
    }
    try {
      await login(email, password, rememberMe);
      router.replace("/(tabs)/mission");
    } catch (e) {
      if (e instanceof RememberMeStorageError) {
        Alert.alert("", "Impossible d'enregistrer sur cet appareil.");
        router.replace("/(tabs)/mission");
        return;
      }
      Alert.alert("Connexion échouée", "Email ou mot de passe incorrect.");
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
            <Text style={styles.kicker}>Espace Chauffeur</Text>
            <Text style={styles.title}>{"Missions en\nTemps Réel"}</Text>
            <Text style={styles.subtitle}>
              Pilotez votre journée : missions, disponibilité et communication.
            </Text>
          </View>

          <View style={styles.form}>
            {sessionExpiredMarker && (
              <SessionExpiredBanner marker={sessionExpiredMarker} />
            )}
            {!hydrated && (
              <View style={{ paddingVertical: 8, alignItems: "center" }}>
                <ActivityIndicator size="small" color={palette.secondary} />
              </View>
            )}
            <View style={styles.inputBlock}>
              <Text style={styles.label}>Email Chauffeur</Text>
              <TextInput
                style={styles.input}
                placeholder="chauffeur@liri.ch"
                placeholderTextColor={palette.placeholder}
                keyboardType="email-address"
                autoCapitalize="none"
                value={email}
                onChangeText={setEmail}
                editable={hydrated}
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
                  editable={hydrated}
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

            {/* ✅ PHASE 1 : Checkbox "Se souvenir de moi" */}
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
              onPress={() => router.push("/(auth)/forgot-password")}
            >
              <Text style={styles.helperLinkText}>Mot de passe oublié ?</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.primaryButton}
              onPress={handleLogin}
              disabled={loading}
            >
              {loading ? (
                <Loader />
              ) : (
                <Text style={styles.primaryButtonText}>Se connecter</Text>
              )}
            </TouchableOpacity>

            <View style={styles.switchRow}>
              <Text style={styles.switchPrompt}>
                Besoin du dispatch mobile ?
              </Text>
              <TouchableOpacity
                onPress={async () => {
                  await setMode("enterprise");
                  router.replace("/(enterprise-auth)/login" as any);
                }}
              >
                <Text style={styles.switchLink}>
                  Accéder à l’espace entreprise
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
