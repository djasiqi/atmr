import React from "react";
import { Linking, Platform, Pressable, StyleSheet, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import type { DriverBackgroundTrackingUiState } from "../hooks/useDriverBackgroundTrackingUi";

type Props = {
  ui: DriverBackgroundTrackingUiState;
  onRequestPermission?: () => void;
};

export function DriverTrackingBanner({ ui, onRequestPermission }: Props) {
  if (!ui.showBanner) return null;

  const isPermission = ui.bannerKind === "permission_required";
  const message = isPermission
    ? 'Autorisation « Toujours autoriser » requise pour le suivi en mission.'
    : ui.lastNativeStartError?.includes("startup_timeout")
      ? "Suivi arrière-plan indisponible (délai de démarrage dépassé)."
      : "Suivi arrière-plan indisponible.";

  const openSettings = () => {
    if (Platform.OS === "ios") {
      void Linking.openURL("app-settings:");
    } else {
      void Linking.openSettings();
    }
  };

  return (
    <View style={[styles.banner, isPermission ? styles.bannerPermission : styles.bannerError]}>
      <AppText variant="body" style={styles.text}>
        {message}
      </AppText>
      <Pressable
        onPress={isPermission && onRequestPermission ? onRequestPermission : openSettings}
        style={styles.button}
      >
        <AppText variant="label" style={styles.buttonText}>
          {isPermission ? "Autoriser" : "Paramètres"}
        </AppText>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 12,
    borderRadius: 8,
    borderWidth: 1,
  },
  bannerError: {
    backgroundColor: "#FEF2F2",
    borderColor: "#FECACA",
  },
  bannerPermission: {
    backgroundColor: "#FEF2F2",
    borderColor: "#FCA5A5",
  },
  text: {
    flex: 1,
    color: "#991B1B",
    fontSize: 13,
  },
  button: {
    marginLeft: 8,
    backgroundColor: "#DC2626",
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  buttonText: {
    color: "#FFFFFF",
    fontSize: 12,
  },
});
