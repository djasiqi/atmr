// components/common/EnterpriseReconnectBanner.tsx
// P2.1.2 — Affiche un bouton "Reconnecter" quand reconnect_failed (enterprise)
// P2.1.2b — Subscribe au lieu de polling (perf, batterie)

import React, { useState, useEffect } from "react";
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from "react-native";
import { subscribeSocketStatus, reconnectSocketManually } from "@/services/socket";

export function EnterpriseReconnectBanner() {
  const [reconnectExhausted, setReconnectExhausted] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);

  useEffect(() => {
    const unsubscribe = subscribeSocketStatus((payload) => {
      setReconnectExhausted(
        payload.reconnectExhausted && payload.role === "enterprise"
      );
    });
    return unsubscribe;
  }, []);

  const handleReconnect = async () => {
    if (reconnecting) return;
    setReconnecting(true);
    try {
      await reconnectSocketManually("enterprise");
    } finally {
      setReconnecting(false);
    }
  };

  if (!reconnectExhausted) return null;

  return (
    <View style={styles.banner}>
      <Text style={styles.text}>
        Connexion temps réel indisponible. Réessayer ?
      </Text>
      <TouchableOpacity
        style={[styles.button, reconnecting && styles.buttonDisabled]}
        onPress={handleReconnect}
        disabled={reconnecting}
        accessibilityLabel="Reconnecter le socket"
        accessibilityRole="button"
      >
        {reconnecting ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Reconnecter</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: "#FEF3C7",
    borderBottomWidth: 1,
    borderBottomColor: "#F59E0B",
  },
  text: {
    flex: 1,
    fontSize: 14,
    color: "#92400E",
    marginRight: 12,
  },
  button: {
    backgroundColor: "#0A7F59",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  buttonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
});
