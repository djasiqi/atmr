/**
 * P1.C — Bannière "Hors ligne" globale.
 * Affichée quand isConnected === false ou isInternetReachable === false.
 */

import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { useNetworkBanner } from "@/hooks/useNetworkBanner";

export function OfflineBanner() {
  const isOffline = useNetworkBanner();

  if (!isOffline) return null;

  return (
    <View style={styles.banner}>
      <Text style={styles.text}>Hors ligne — reconnexion automatique.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    backgroundColor: "#FEE2E2",
    paddingVertical: 8,
    paddingHorizontal: 16,
    alignItems: "center",
    justifyContent: "center",
    borderBottomWidth: 1,
    borderBottomColor: "#FECACA",
  },
  text: {
    color: "#991B1B",
    fontSize: 13,
    fontWeight: "500",
  },
});
