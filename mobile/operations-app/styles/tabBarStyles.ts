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

export const tabBarStyles = {
  tabBarStyle: {
    position: "absolute",
    backgroundColor: "#004D40", // Même couleur que le header
    borderTopLeftRadius: 20, // Coins arrondis comme le header
    borderTopRightRadius: 20,
    borderTopWidth: 0, // Supprimer la bordure pour un look plus moderne
    elevation: 8,
    shadowColor: "#000",
    shadowOpacity: 0.2, // Ombre plus prononcée
    shadowOffset: { width: 0, height: -4 },
    shadowRadius: 8,
    height: Platform.OS === "ios" ? 75 : 65,
    paddingBottom: Platform.OS === "ios" ? 20 : 12,
    paddingTop: 0,
    paddingHorizontal: 8,
  } as ViewStyle,

  tabBarItemStyle: {
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 6,
    borderRadius: 12, // Coins arrondis pour les items
    paddingVertical: 4,
    paddingHorizontal: 8,
  } as ViewStyleStrict,

  tabBarLabelStyle: {
    fontSize: 12,
    fontWeight: "700", // Plus gras comme le header
    marginTop: 0, // Supprimer le marginTop pour centrer le texte
    letterSpacing: 0.3, // Espacement des lettres
  } as TextStyle,
};
