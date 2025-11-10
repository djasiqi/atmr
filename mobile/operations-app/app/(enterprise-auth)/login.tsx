import React, { useMemo, useState } from "react";
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { router } from "expo-router";
import Ionicons from "react-native-vector-icons/Ionicons";

import { useAuth } from "@/hooks/useAuth";
import { Loader } from "@/components/ui/Loader";
import { getLoginStyles } from "@/styles/loginStyles";

export default function EnterpriseLoginScreen() {
  const { loginEnterprise, enterpriseLoading, setMode } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const { styles, palette } = useMemo(() => getLoginStyles("enterprise"), []);

  const handleSubmit = async () => {
    if (!email || !password) {
      Alert.alert(
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
      });
      if (result.mfaRequired) {
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
      Alert.alert("Échec connexion", message);
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
              style={styles.helperLink}
              onPress={() => router.push("/(auth)/forgot-password")}
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
