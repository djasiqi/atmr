/**
 * Bannière d'état du suivi GPS — Plan GPS background.
 * Affiche : Suivi actif / limité / désactivé / permission requise.
 */

import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, Linking, Platform } from "react-native";
import type { TrackingDisplayState } from "@/hooks/useTrackingState";

interface TrackingStateBannerProps {
  displayState: TrackingDisplayState;
  onRequestPermission?: () => Promise<void>;
}

const MESSAGES: Record<TrackingDisplayState, string | null> = {
  active: null,
  disabled: null,
  limited:
    "Suivi limité au premier plan — activez la localisation en arrière-plan pour un suivi continu pendant la mission.",
  permission_required:
    "Autorisation de localisation requise pour le suivi en mission.",
};

export function TrackingStateBanner({
  displayState,
  onRequestPermission,
}: TrackingStateBannerProps) {
  if (displayState === "active" || displayState === "disabled") return null;

  const message = MESSAGES[displayState];
  if (!message) return null;

  const handleOpenSettings = () => {
    if (Platform.OS === "ios") {
      Linking.openURL("app-settings:");
    } else {
      Linking.sendIntent("android.settings.LOCATION_SOURCE_SETTINGS").catch(
        () => Linking.openSettings()
      );
    }
  };

  const handleRequestPermission = async () => {
    if (displayState === "permission_required" && onRequestPermission) {
      await onRequestPermission();
    } else {
      handleOpenSettings();
    }
  };

  const isLimited = displayState === "limited";
  const isPermissionRequired = displayState === "permission_required";

  return (
    <View style={[styles.banner, isLimited ? styles.bannerLimited : styles.bannerPermission]}>
      <Text style={[styles.text, isLimited ? styles.textLimited : styles.textPermission]}>
        {message}
      </Text>
      <TouchableOpacity
        onPress={isPermissionRequired ? handleRequestPermission : handleOpenSettings}
        style={styles.button}
      >
        <Text style={styles.buttonText}>
          {isPermissionRequired ? "Autoriser" : "Paramètres"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
  },
  bannerLimited: {
    backgroundColor: "#EFF6FF",
    borderBottomColor: "#BFDBFE",
  },
  bannerPermission: {
    backgroundColor: "#FEF3C7",
    borderBottomColor: "#FDE68A",
  },
  text: {
    fontSize: 13,
    fontWeight: "500",
    flex: 1,
  },
  textLimited: {
    color: "#1E40AF",
  },
  textPermission: {
    color: "#92400E",
  },
  button: {
    backgroundColor: "#3B82F6",
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 6,
    marginLeft: 8,
  },
  buttonText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#FFFFFF",
  },
});
