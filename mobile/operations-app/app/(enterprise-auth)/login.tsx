import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { router } from "expo-router";

import { useAuth } from "@/hooks/useAuth";

export default function EnterpriseLoginScreen() {
  const { loginEnterprise, enterpriseLoading, setMode } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

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
        <View style={styles.header}>
          <Text style={styles.title}>Espace Enterprise Dispatch</Text>
          <Text style={styles.subtitle}>
            Accédez au pilotage des assignations et au monitoring en mobilité.
          </Text>
        </View>

        <View style={styles.form}>
          <Text style={styles.label}>Email entreprise</Text>
          <TextInput
            style={styles.input}
            placeholder="prenom.nom@entreprise.ch"
            placeholderTextColor="#9AA3BC"
            keyboardType="email-address"
            autoCapitalize="none"
            value={email}
            onChangeText={setEmail}
          />

          <Text style={styles.label}>Mot de passe</Text>
          <TextInput
            style={styles.input}
            placeholder="Mot de passe"
            placeholderTextColor="#9AA3BC"
            secureTextEntry
            value={password}
            onChangeText={setPassword}
          />

          <TouchableOpacity
            style={styles.primaryButton}
            onPress={handleSubmit}
            disabled={enterpriseLoading}
          >
            {enterpriseLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.primaryButtonText}>Se connecter</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={async () => {
              await setMode("driver");
              router.replace("/(auth)/login");
            }}
          >
            <Text style={styles.secondaryButtonText}>
              Retour à l’espace chauffeur
            </Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#0B1736",
  },
  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingVertical: 32,
    justifyContent: "center",
  },
  header: {
    marginBottom: 36,
  },
  title: {
    fontSize: 30,
    fontWeight: "700",
    color: "#FFFFFF",
    marginBottom: 12,
  },
  subtitle: {
    fontSize: 16,
    color: "#CED6FF",
    lineHeight: 22,
  },
  form: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 20,
    padding: 24,
  },
  label: {
    color: "#E3E9FF",
    fontSize: 14,
    fontWeight: "600",
    marginBottom: 8,
  },
  input: {
    height: 48,
    backgroundColor: "rgba(9,23,62,0.85)",
    borderRadius: 12,
    paddingHorizontal: 16,
    color: "#FFFFFF",
    marginBottom: 18,
  },
  primaryButton: {
    backgroundColor: "#4D6BFE",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  secondaryButton: {
    marginTop: 20,
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "#AAB6FF",
    fontSize: 15,
    fontWeight: "500",
    textDecorationLine: "underline",
  },
});
