import type { ViewStyle } from "react-native";
import { createShadow } from "../../../../styles/shadowStyles";

/**
 * Empilement des listes de suggestions (client / adresse).
 *
 * Important (web + RN) : le z-index du dropdown ne compte que dans le stacking context
 * de son ancêtre positionné. Il faut donc monter le z-index sur le **frère** du bloc
 * qui suit (formGroup client / colonne adresse), pas seulement sur le champ interne —
 * sinon les champs suivants (adresses, etc.) se peignent par-dessus.
 */

/**
 * Contenu sous une suggestion ouverte : couche basse + ne capture pas les clics
 * (reste visible, mais laisse passer les presses vers le dropdown absolu au-dessus).
 */
export const suggestionContentBelowOverlayStyle: ViewStyle = {
  position: "relative",
  zIndex: 1,
  elevation: 1,
  pointerEvents: "none",
};

/** Conteneur frère (formGroup / colonne) du champ dont la liste est ouverte. */
export const suggestionOverlayFieldStyle: ViewStyle = {
  position: "relative",
  zIndex: 40,
  elevation: 40,
  overflow: "visible",
};

/** Rangée du champ actif : fond opaque pour lisibilité du dropdown. */
export const suggestionOverlayFieldWithBgStyle: ViewStyle = {
  ...suggestionOverlayFieldStyle,
  backgroundColor: "#FFFFFF",
};

/** Section formulaire au-dessus des sections 2/3. */
export const suggestionOverlaySectionStyle: ViewStyle = {
  position: "relative",
  zIndex: 30,
  elevation: 30,
  overflow: "visible",
  backgroundColor: "#FFFFFF",
};

/** @deprecated No-op — ne plus masquer le contenu sous les suggestions. */
export const sectionsHiddenDuringSuggestionStyle: ViewStyle = {};

/** Conteneur relatif du champ pendant l’ouverture de la liste. */
export const suggestionFieldOpenStyle: ViewStyle = {
  position: "relative",
  zIndex: 50,
  elevation: 50,
  overflow: "visible",
};

/** Ancrage absolu du panneau de suggestions sous le champ. */
export const suggestionDropdownAnchorStyle: ViewStyle = {
  position: "absolute",
  left: 0,
  right: 0,
  top: "100%",
  marginTop: 4,
  zIndex: 60,
  elevation: 60,
  overflow: "visible",
  pointerEvents: "box-none",
};

/** Panneau cliquable (réactive les events annulés par le parent box-none). */
export const suggestionDropdownHitAreaStyle: ViewStyle = {
  pointerEvents: "auto",
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
