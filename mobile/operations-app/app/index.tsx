import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import Ionicons from "react-native-vector-icons/Ionicons";
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
  const [selectedMode, setSelectedMode] = useState(mode ?? "driver");
  const toggleAnim = useRef(new Animated.Value(mode === "enterprise" ? 1 : 0)).current;
  const [toggleWidth, setToggleWidth] = useState(0);

  useEffect(() => {
    if (loading) return;
    if (mode === "driver" && isDriverAuthenticated) {
      router.replace("/(tabs)/mission");
    } else if (mode === "enterprise" && isEnterpriseAuthenticated) {
      router.replace("/(enterprise)/dashboard" as any);
    }
  }, [loading, mode, isDriverAuthenticated, isEnterpriseAuthenticated, router]);

  useEffect(() => {
    setSelectedMode(mode ?? "driver");
    toggleAnim.setValue(mode === "enterprise" ? 1 : 0);
  }, [mode, toggleAnim]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.info}>Initialisation…</Text>
      </View>
    );
  }

  const handleSelect = (target: "driver" | "enterprise") => {
    if (selectedMode === target) return;
    setSelectedMode(target);
    Animated.timing(toggleAnim, {
      toValue: target === "enterprise" ? 1 : 0,
      duration: 260,
      useNativeDriver: true,
    }).start();
  };

  const handleContinue = async () => {
    await setMode(selectedMode);
    if (selectedMode === "driver") {
      router.replace("/(auth)/login");
    } else {
      router.replace("/(enterprise-auth)/login" as any);
    }
  };

  const indicatorTranslate = toggleAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, toggleWidth > 0 ? toggleWidth / 2 : 0],
  });

  const description =
    selectedMode === "enterprise"
      ? {
          title: "Espace Enterprise Dispatch",
          text: "Pilotez les assignations, suivez les alertes critiques et ajustez les paramètres en temps réel depuis votre téléphone.",
        }
      : {
          title: "Espace Chauffeur",
          text: "Consultez vos missions quotidiennes, mettez à jour votre disponibilité et gardez le contact avec l’équipe.",
        };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={["#06100C", "#10261A", "#06100C"]}
        style={StyleSheet.absoluteFill}
      />

      <View style={styles.content}>
        <Text style={styles.brand}>Bienvenue</Text>
        <Text style={styles.headline}>Choisissez votre espace</Text>
        <Text style={styles.subtitle}>
          Accédez aux outils dédiés à la conduite ou au dispatch en un seul geste.
        </Text>

        <View
          style={styles.toggleWrapper}
          onLayout={(event) => setToggleWidth(event.nativeEvent.layout.width)}
        >
          <Animated.View
            pointerEvents="none"
            style={[
              styles.toggleIndicator,
              {
                width:
                  toggleWidth > 0 ? Math.max(toggleWidth / 2 - 8, 0) : 0,
                transform: [{ translateX: indicatorTranslate }],
              },
            ]}
          />
          <TouchableOpacity
            style={styles.toggleOption}
            onPress={() => handleSelect("driver")}
            activeOpacity={0.85}
          >
            <Ionicons
              name="car-outline"
              size={22}
              style={
                selectedMode === "driver"
                  ? styles.toggleIconActive
                  : styles.toggleIcon
              }
            />
            <Text
              style={
                selectedMode === "driver"
                  ? styles.toggleLabelActive
                  : styles.toggleLabel
              }
            >
              Chauffeur
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.toggleOption}
            onPress={() => handleSelect("enterprise")}
            activeOpacity={0.85}
          >
            <Ionicons
              name="business-outline"
              size={22}
              style={
                selectedMode === "enterprise"
                  ? styles.toggleIconActive
                  : styles.toggleIcon
              }
            />
            <Text
              style={
                selectedMode === "enterprise"
                  ? styles.toggleLabelActive
                  : styles.toggleLabel
              }
            >
              Entreprise
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>{description.title}</Text>
          <Text style={styles.cardDescription}>{description.text}</Text>
        </View>

        <TouchableOpacity
          style={styles.primaryButton}
          activeOpacity={0.9}
          onPress={handleContinue}
        >
          <Text style={styles.primaryButtonText}>Accéder à cet espace</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#06100C",
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#06100C",
  },
  info: {
    marginTop: 12,
    color: "#fff",
    fontSize: 16,
  },
  content: {
    flex: 1,
    padding: 28,
    justifyContent: "center",
  },
  brand: {
    color: "rgba(255,255,255,0.55)",
    textTransform: "uppercase",
    letterSpacing: 4,
    fontSize: 13,
    marginBottom: 12,
  },
  headline: {
    fontSize: 32,
    fontWeight: "700",
    color: "#F4FFFA",
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: "#8AA295",
    marginBottom: 28,
  },
  card: {
    backgroundColor: "rgba(244,255,250,0.05)",
    borderRadius: 18,
    padding: 22,
    marginBottom: 28,
    borderWidth: 1,
    borderColor: "rgba(121,197,156,0.18)",
  },
  cardTitle: {
    fontSize: 22,
    fontWeight: "600",
    color: "#E6F2EA",
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 16,
    color: "rgba(244,255,250,0.78)",
    lineHeight: 23,
  },
  toggleWrapper: {
    flexDirection: "row",
    backgroundColor: "rgba(244,255,250,0.08)",
    borderRadius: 18,
    padding: 4,
    marginBottom: 28,
    position: "relative",
    overflow: "hidden",
  },
  toggleIndicator: {
    position: "absolute",
    top: 4,
    bottom: 4,
    borderRadius: 14,
    backgroundColor: "rgba(10,122,77,0.4)",
  },
  toggleOption: {
    flex: 1,
    alignItems: "center",
    paddingVertical: 14,
    zIndex: 1,
    gap: 4,
  },
  toggleIcon: {
    color: "rgba(244,255,250,0.45)",
  },
  toggleIconActive: {
    color: "#F4FFFA",
  },
  toggleLabel: {
    fontSize: 15,
    color: "rgba(244,255,250,0.6)",
    fontWeight: "600",
    letterSpacing: 0.2,
  },
  toggleLabelActive: {
    fontSize: 15,
    color: "#F4FFFA",
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  primaryButton: {
    backgroundColor: "#00796B",
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: "center",
    shadowColor: "#00796B",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.25,
    shadowRadius: 12,
    elevation: 6,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
    letterSpacing: 0.4,
  },
});
