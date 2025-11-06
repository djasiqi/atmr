import React, { useEffect, useState } from "react";
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
import { router, useLocalSearchParams } from "expo-router";

import { useAuth } from "@/hooks/useAuth";

export default function EnterpriseMfaScreen() {
  const { challengeId } = useLocalSearchParams<{ challengeId?: string }>();
  const { pendingEnterpriseMfa, verifyEnterpriseMfa, enterpriseLoading } =
    useAuth();
  const [code, setCode] = useState("");
  const [challenge, setChallenge] = useState<string | null>(null);

  useEffect(() => {
    if (challengeId && typeof challengeId === "string") {
      setChallenge(challengeId);
      return;
    }
    if (pendingEnterpriseMfa?.challengeId) {
      setChallenge(pendingEnterpriseMfa.challengeId);
    }
  }, [challengeId, pendingEnterpriseMfa]);

  const submitCode = async () => {
    if (!challenge) {
      Alert.alert(
        "Challenge expiré",
        "Veuillez relancer la connexion depuis l'écran précédent."
      );
      router.replace("/(enterprise-auth)/login" as any);
      return;
    }
    if (!code || code.length < 4) {
      Alert.alert("Code invalide", "Veuillez saisir le code MFA reçu.");
      return;
    }
    try {
      await verifyEnterpriseMfa(code, challenge);
      router.replace("/(enterprise)/dashboard" as any);
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de vérifier le code MFA.";
      Alert.alert("Erreur MFA", message);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
      >
        <View style={styles.header}>
          <Text style={styles.title}>Vérification MFA</Text>
          <Text style={styles.subtitle}>
            Saisissez le code à 6 chiffres généré par votre application
            d’authentification.
          </Text>
        </View>

        <View style={styles.form}>
          <TextInput
            style={styles.input}
            placeholder="••••••"
            keyboardType="number-pad"
            maxLength={6}
            value={code}
            onChangeText={setCode}
            placeholderTextColor="#808AA9"
          />

          <TouchableOpacity
            style={styles.primaryButton}
            onPress={submitCode}
            disabled={enterpriseLoading}
          >
            {enterpriseLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.primaryButtonText}>Valider le code</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={() => router.replace("/(enterprise-auth)/login" as any)}
          >
            <Text style={styles.secondaryButtonText}>
              Retour à l’écran de connexion
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
    fontSize: 28,
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
  input: {
    height: 60,
    backgroundColor: "rgba(9,23,62,0.85)",
    borderRadius: 12,
    paddingHorizontal: 20,
    color: "#FFFFFF",
    fontSize: 26,
    letterSpacing: 12,
    textAlign: "center",
    marginBottom: 24,
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
