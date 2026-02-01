/**
 * P1.C — Bannière "Session expirée" sur écrans de login.
 * Optionnel : lien "Détails" affiche reason + ts relatif (support).
 */

import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

type SessionExpiredMarker = { reason: string; ts: number };

function formatRelativeTime(ts: number): string {
  const diffMs = Date.now() - ts;
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return "à l'instant";
  if (diffMin === 1) return "il y a 1 min";
  if (diffMin < 60) return `il y a ${diffMin} min`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH === 1) return "il y a 1 h";
  return `il y a ${diffH} h`;
}

export function SessionExpiredBanner({ marker }: { marker: SessionExpiredMarker }) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <View style={styles.banner}>
      <Text style={styles.mainText}>Session expirée — reconnectez-vous.</Text>
      <TouchableOpacity
        onPress={() => setShowDetails((v) => !v)}
        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
      >
        <Text style={styles.detailsLink}>{showDetails ? "Masquer" : "Détails"}</Text>
      </TouchableOpacity>
      {showDetails && (
        <Text style={styles.detailsText}>
          {marker.reason} · {formatRelativeTime(marker.ts)}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    backgroundColor: "#FEF3C7",
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: "#F59E0B",
  },
  mainText: {
    color: "#92400E",
    fontSize: 14,
    fontWeight: "500",
  },
  detailsLink: {
    color: "#B45309",
    fontSize: 12,
    marginTop: 6,
    textDecorationLine: "underline",
  },
  detailsText: {
    color: "#92400E",
    fontSize: 11,
    marginTop: 4,
    opacity: 0.9,
  },
});
