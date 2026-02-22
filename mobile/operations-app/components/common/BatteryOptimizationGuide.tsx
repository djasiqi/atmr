import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  Platform,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  isSamsungDevice,
  requestBatteryOptimizationExemption,
  openSamsungBatterySettings,
} from "../../services/batteryOptimization";

const DISMISS_KEY = "battery_guide_dismissed_v1";

interface Props {
  visible: boolean;
  onDismiss: () => void;
}

export function BatteryOptimizationGuide({ visible, onDismiss }: Props) {
  const [samsung, setSamsung] = useState(false);
  const [permanentlyDismissed, setPermanentlyDismissed] = useState(true);

  useEffect(() => {
    setSamsung(isSamsungDevice());
    AsyncStorage.getItem(DISMISS_KEY).then((val) => {
      setPermanentlyDismissed(val === "true");
    });
  }, []);

  if (Platform.OS !== "android" || permanentlyDismissed) return null;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onDismiss}
    >
      <View style={styles.overlay}>
        <View style={styles.card}>
          <Text style={styles.title}>Optimisation de la batterie</Text>
          <Text style={styles.body}>
            Pour recevoir les notifications de mission même quand l'application
            est en arrière-plan, veuillez désactiver l'optimisation de batterie
            pour Liri Opérations.
          </Text>

          {samsung && (
            <Text style={styles.body}>
              Sur Samsung, allez aussi dans les paramètres de batterie et
              désactivez « Mise en veille de l'application » pour Liri.
            </Text>
          )}

          <TouchableOpacity
            style={styles.primaryBtn}
            onPress={async () => {
              await requestBatteryOptimizationExemption();
              onDismiss();
            }}
          >
            <Text style={styles.primaryBtnText}>Autoriser</Text>
          </TouchableOpacity>

          {samsung && (
            <TouchableOpacity
              style={styles.secondaryBtn}
              onPress={async () => {
                await openSamsungBatterySettings();
              }}
            >
              <Text style={styles.secondaryBtnText}>
                Ouvrir paramètres Samsung
              </Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity style={styles.dismissBtn} onPress={onDismiss}>
            <Text style={styles.dismissText}>Plus tard</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.dismissBtn}
            onPress={async () => {
              await AsyncStorage.setItem(DISMISS_KEY, "true");
              setPermanentlyDismissed(true);
              onDismiss();
            }}
          >
            <Text style={styles.neverText}>Ne plus afficher</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 24,
    width: "100%",
    maxWidth: 360,
    elevation: 8,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
  },
  title: {
    fontSize: 18,
    fontWeight: "700",
    color: "#1F2937",
    marginBottom: 12,
    textAlign: "center",
  },
  body: {
    fontSize: 14,
    color: "#4B5563",
    lineHeight: 20,
    marginBottom: 12,
  },
  primaryBtn: {
    backgroundColor: "#2563EB",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 8,
  },
  primaryBtnText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "600",
  },
  secondaryBtn: {
    backgroundColor: "#F3F4F6",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 8,
  },
  secondaryBtnText: {
    color: "#374151",
    fontSize: 14,
    fontWeight: "500",
  },
  dismissBtn: {
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 4,
  },
  dismissText: {
    color: "#9CA3AF",
    fontSize: 13,
  },
  neverText: {
    color: "#D1D5DB",
    fontSize: 12,
    textDecorationLine: "underline" as const,
  },
});
