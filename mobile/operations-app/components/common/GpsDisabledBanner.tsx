import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, TouchableOpacity, Linking, Platform } from "react-native";
import {
  getAdaptiveLocationTracker,
  type GpsStatus,
} from "@/services/locationTracker";

export function GpsDisabledBanner() {
  const [gpsStatus, setGpsStatus] = useState<GpsStatus>("unknown");

  useEffect(() => {
    const tracker = getAdaptiveLocationTracker();
    const unsubscribe = tracker.onGpsStatusChange(setGpsStatus);
    return unsubscribe;
  }, []);

  if (gpsStatus === "active" || gpsStatus === "unknown") return null;

  const message =
    gpsStatus === "disabled"
      ? "GPS désactivé — activez la localisation pour être visible."
      : "Position indisponible — vérifiez votre GPS.";

  const openSettings = () => {
    if (Platform.OS === "ios") {
      Linking.openURL("app-settings:");
    } else {
      Linking.sendIntent("android.settings.LOCATION_SOURCE_SETTINGS").catch(
        () => Linking.openSettings()
      );
    }
  };

  return (
    <View style={styles.banner}>
      <Text style={styles.text}>{message}</Text>
      <TouchableOpacity onPress={openSettings} style={styles.button}>
        <Text style={styles.buttonText}>Activer</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    backgroundColor: "#FEF3C7",
    paddingVertical: 8,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "#FDE68A",
  },
  text: {
    color: "#92400E",
    fontSize: 13,
    fontWeight: "500",
    flex: 1,
  },
  button: {
    backgroundColor: "#F59E0B",
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 6,
    marginLeft: 8,
  },
  buttonText: {
    color: "#FFFFFF",
    fontSize: 12,
    fontWeight: "600",
  },
});
