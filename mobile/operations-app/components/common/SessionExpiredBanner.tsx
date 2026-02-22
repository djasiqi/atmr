/**
 * P1.C — Bannière "Session expirée" / "Compte désactivé" sur écrans de login.
 * Optionnel : lien "Détails" affiche reason + ts relatif (support).
 */

import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { isAccountDisabledReason } from "@/services/authLogoutReasons";

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
  const isDisabled = isAccountDisabledReason(marker.reason);

  const bannerStyle = isDisabled ? styles.bannerDisabled : styles.banner;
  const mainMessage = isDisabled
    ? "Votre compte a été désactivé. Contactez votre entreprise."
    : "Session expirée — reconnectez-vous.";

  return (
    <View style={bannerStyle}>
      <Text style={isDisabled ? styles.mainTextDisabled : styles.mainText}>
        {mainMessage}
      </Text>
      <TouchableOpacity
        onPress={() => setShowDetails((v) => !v)}
        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
      >
        <Text style={isDisabled ? styles.detailsLinkDisabled : styles.detailsLink}>
          {showDetails ? "Masquer" : "Détails"}
        </Text>
      </TouchableOpacity>
      {showDetails && (
        <Text style={isDisabled ? styles.detailsTextDisabled : styles.detailsText}>
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
  bannerDisabled: {
    backgroundColor: "#FEE2E2",
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: "#EF4444",
  },
  mainText: {
    color: "#92400E",
    fontSize: 14,
    fontWeight: "500",
  },
  mainTextDisabled: {
    color: "#991B1B",
    fontSize: 14,
    fontWeight: "500",
  },
  detailsLink: {
    color: "#B45309",
    fontSize: 12,
    marginTop: 6,
    textDecorationLine: "underline",
  },
  detailsLinkDisabled: {
    color: "#DC2626",
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
  detailsTextDisabled: {
    color: "#991B1B",
    fontSize: 11,
    marginTop: 4,
    opacity: 0.9,
  },
});
