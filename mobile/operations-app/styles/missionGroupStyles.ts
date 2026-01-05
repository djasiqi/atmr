// styles/missionGroupStyles.ts
import { StyleSheet, Platform } from "react-native";

const palette = {
  warning: "#FF6B35",
  warningBg: "rgba(255, 107, 53, 0.1)",
  warningBorder: "rgba(255, 107, 53, 0.3)",
  text: "#15362B",
  secondary: "#5F7369",
};

const groupHeaderShadow = Platform.OS === "web"
  ? { boxShadow: "0 2px 8px rgba(255, 107, 53, 0.15)" }
  : {
      shadowColor: palette.warning,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.15,
      shadowRadius: 8,
      elevation: 4,
    };

export const styles = StyleSheet.create({
  // En-tête de groupe
  groupHeaderContainer: {
    backgroundColor: palette.warningBg,
    borderLeftWidth: 4,
    borderLeftColor: palette.warning,
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 20,
    marginTop: 12,
    marginBottom: 8,
    ...groupHeaderShadow,
  },

  groupHeaderContent: {
    flexDirection: "row",
    alignItems: "center",
  },

  groupHeaderIcon: {
    marginRight: 12,
  },

  groupHeaderTextContainer: {
    flex: 1,
  },

  groupHeaderLabel: {
    fontSize: 14,
    fontWeight: "700",
    color: palette.warning,
    marginBottom: 4,
    letterSpacing: 0.2,
  },

  groupHeaderLocation: {
    fontSize: 13,
    color: palette.text,
    fontWeight: "600",
    lineHeight: 18,
  },

  // Badge numéroté pour les missions
  missionNumberBadge: {
    position: "absolute",
    top: -8,
    right: -8,
    backgroundColor: palette.warning,
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#FFFFFF",
    zIndex: 10,
  },

  missionNumberText: {
    color: "#FFFFFF",
    fontSize: 12,
    fontWeight: "700",
  },

  // Bordure pour les missions groupées
  groupedCardBorder: {
    borderWidth: 2,
    borderColor: palette.warningBorder,
  },
});

