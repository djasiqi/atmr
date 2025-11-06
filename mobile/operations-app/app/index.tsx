import React, { useEffect } from "react";
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { useAuth } from "@/hooks/useAuth";

export default function IndexScreen() {
  const router = useRouter();
  const {
    loading,
    mode,
    setMode,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
  } = useAuth();

  useEffect(() => {
    if (loading) return;
    if (mode === "driver" && isDriverAuthenticated) {
      router.replace("/(tabs)/mission");
    } else if (mode === "enterprise" && isEnterpriseAuthenticated) {
      router.replace("/(enterprise)/dashboard" as any);
    }
  }, [loading, mode, isDriverAuthenticated, isEnterpriseAuthenticated, router]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.info}>Initialisation…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Bienvenue</Text>
      <Text style={styles.subtitle}>
        Choisissez l’espace auquel vous souhaitez accéder.
      </Text>

      <TouchableOpacity
        style={styles.card}
        onPress={async () => {
          await setMode("driver");
          router.replace("/(auth)/login");
        }}
      >
        <Text style={styles.cardTitle}>Espace Chauffeur</Text>
        <Text style={styles.cardDescription}>
          Suivez vos missions, mettez à jour votre statut et accédez aux détails
          clients.
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.card}
        onPress={async () => {
          await setMode("enterprise");
          router.replace("/(enterprise-auth)/login" as any);
        }}
      >
        <Text style={styles.cardTitle}>Espace Enterprise Dispatch</Text>
        <Text style={styles.cardDescription}>
          Pilotez les assignations et surveillez l’activité en temps réel depuis
          votre mobile.
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 24,
    backgroundColor: "#0B1736",
    justifyContent: "center",
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0B1736",
  },
  info: {
    marginTop: 12,
    color: "#fff",
    fontSize: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 12,
  },
  subtitle: {
    fontSize: 18,
    color: "#d1d8ff",
    marginBottom: 32,
  },
  card: {
    backgroundColor: "rgba(255,255,255,0.1)",
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
  },
  cardTitle: {
    fontSize: 22,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 16,
    color: "#dbe1ff",
    lineHeight: 22,
  },
});
