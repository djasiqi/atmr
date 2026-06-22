import type { ViewStyle } from "react-native";
import { createShadow } from "../../../../styles/shadowStyles";

/**
 * Empilement des listes de suggestions (client / adresse).
 * Même comportement iOS, Android et web : zIndex + elevation + overflow visible.
 */

/** Champ actif : dropdown au-dessus des champs voisins dans la section. */
export const suggestionOverlayFieldStyle: ViewStyle = {
  position: "relative",
  zIndex: 120,
  elevation: 120,
  overflow: "visible",
};

/** Rangée client : fond opaque pour masquer le contenu sous-jacent. */
export const suggestionOverlayFieldWithBgStyle: ViewStyle = {
  ...suggestionOverlayFieldStyle,
  backgroundColor: "#FFFFFF",
};

/** Section formulaire au-dessus des sections suivantes (ex. modale création). */
export const suggestionOverlaySectionStyle: ViewStyle = {
  position: "relative",
  zIndex: 100,
  elevation: 100,
  overflow: "visible",
  backgroundColor: "#FFFFFF",
};

/** Contenu masqué pendant une suggestion ouverte (évite les chevauchements visuels). */
export const sectionsHiddenDuringSuggestionStyle: ViewStyle = {
  opacity: 0,
  pointerEvents: "none",
};

/** Conteneur relatif du champ pendant l’ouverture de la liste. */
export const suggestionFieldOpenStyle: ViewStyle = {
  position: "relative",
  zIndex: 200,
  elevation: 200,
  overflow: "visible",
};

/** Ancrage absolu du panneau de suggestions sous le champ. */
export const suggestionDropdownAnchorStyle: ViewStyle = {
  position: "absolute",
  left: 0,
  right: 0,
  top: "100%",
  marginTop: 4,
  zIndex: 201,
  elevation: 201,
  overflow: "visible",
};

export const suggestionDropdownPanelShadow = createShadow({
  shadowColor: "#0F172A",
  shadowOffset: { width: 0, height: 8 },
  shadowOpacity: 0.12,
  shadowRadius: 16,
  elevation: 16,
});

export const suggestionDropdownPanelStyle: ViewStyle = {
  backgroundColor: "#FFFFFF",
  overflow: "hidden",
  ...suggestionDropdownPanelShadow,
};
