import { Platform, ViewStyle, TextStyle } from "react-native";

type ViewStyleStrict = ViewStyle & {
  alignItems?: "flex-start" | "flex-end" | "center" | "stretch" | "baseline";
  justifyContent?:
    | "flex-start"
    | "flex-end"
    | "center"
    | "space-between"
    | "space-around"
    | "space-evenly";
};

// ✅ Palette épurée et élégante (cohérente avec le login, mission, courses, chat et profile)
const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  barSurface: "#FFFFFF",
  border: "rgba(15,54,43,0.08)",
  shadow: "rgba(16,39,30,0.08)",
  indicator: "rgba(10,127,89,0.12)",
  label: "#0A7F59",
  labelInactive: "#5F7369",
};

export const tabBarStyles = {
  tabBarStyle: {
    position: "absolute",
    backgroundColor: palette.barSurface,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderTopWidth: 1,
    borderColor: palette.border,
    elevation: 8,
    shadowColor: palette.shadow,
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: -4 },
    shadowRadius: 12,
    height: Platform.OS === "ios" ? 78 : 68,
    paddingBottom: Platform.OS === "ios" ? 22 : 14,
    paddingTop: 0, // ✅ Réduit pour monter un peu le contenu
    paddingHorizontal: 12,
  } as ViewStyle,

  tabBarItemStyle: {
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 4,
    borderRadius: 14,
    paddingVertical: 4,
    paddingHorizontal: 12,
  } as ViewStyleStrict,

  // ✅ Style pour le conteneur des icônes (marge en dessous des icônes)
  tabBarIconContainer: {
    marginBottom: 0, // ✅ Marge réduite entre icône et label pour monter un peu
  } as ViewStyle,

  tabBarLabelStyle: {
    fontSize: 12,
    fontWeight: "700",
    marginTop: 0, // ✅ Pas de marge top, la marge est gérée par marginBottom sur l'icône
    marginBottom: 10, // ✅ Marge en dessous du label
    letterSpacing: 0.3,
    color: palette.label,
  } as TextStyle,

  palette,
};
