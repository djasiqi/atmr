// src/app/(auth)/login.tsx
import React, { useMemo, useState } from "react";
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  Alert,
  TextInput,
} from "react-native";
import { router } from "expo-router";
import Ionicons from "react-native-vector-icons/Ionicons";

import { useAuth } from "@/hooks/useAuth";
import { Loader } from "@/components/ui/Loader";
import { getLoginStyles } from "@/styles/loginStyles";

export default function LoginScreen() {
  const { login, loading, setMode } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const { styles, palette } = useMemo(() => getLoginStyles("driver"), []);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert(
        "Erreur",
        "Veuillez entrer votre email et votre mot de passe."
      );
      return;
    }
    try {
      await login(email, password);
      router.replace("/(tabs)/mission");
    } catch {
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
