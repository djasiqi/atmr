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

export const BATTERY_GUIDE_DISMISSED_KEY = "battery_guide_dismissed_v1";

interface Props {
  visible: boolean;
  onDismiss: () => void;
}

export function BatteryOptimizationGuide({ visible, onDismiss }: Props) {
  const [samsung, setSamsung] = useState(false);
  const [permanentlyDismissed, setPermanentlyDismissed] = useState(true);

  useEffect(() => {
    setSamsung(isSamsungDevice());
    AsyncStorage.getItem(BATTERY_GUIDE_DISMISSED_KEY).then((val) => {
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
          <Text style={styles.title}>Paramètres batterie</Text>
          <Text style={styles.body}>
            Désactivez l'optimisation de batterie pour recevoir les notifications
            en arrière-plan.
          </Text>

          {samsung && (
            <Text style={styles.body}>
              Samsung : désactivez « Mise en veille » pour Liri dans les
              paramètres.
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
                Paramètres Samsung
              </Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity style={styles.dismissBtn} onPress={onDismiss}>
            <Text style={styles.dismissText}>Plus tard</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.dismissBtn}
            onPress={async () => {
              await AsyncStorage.setItem(BATTERY_GUIDE_DISMISSED_KEY, "true");
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
    backgroundColor: "rgba(0,0,0,0.52)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 14,
    padding: 22,
    width: "100%",
    maxWidth: 340,
    elevation: 6,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
  },
  title: {
    fontSize: 17,
    fontWeight: "600",
    color: "#1A1A1A",
    marginBottom: 10,
    textAlign: "center",
    letterSpacing: -0.2,
  },
  body: {
    fontSize: 14,
    color: "#525252",
    lineHeight: 20,
    marginBottom: 10,
  },
  primaryBtn: {
    backgroundColor: "#00796B",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 6,
  },
  primaryBtnText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "600",
  },
  secondaryBtn: {
    backgroundColor: "#F5F5F5",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 6,
  },
  secondaryBtnText: {
    color: "#404040",
    fontSize: 14,
    fontWeight: "500",
  },
  dismissBtn: {
    paddingVertical: 10,
    alignItems: "center",
    marginTop: 2,
  },
  dismissText: {
    color: "#737373",
    fontSize: 13,
  },
  neverText: {
    color: "#A3A3A3",
    fontSize: 12,
    textDecorationLine: "underline" as const,
  },
});
