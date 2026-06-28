import { Platform, type TextStyle, type ViewStyle, StyleSheet } from "react-native";
import {
  brandPrimary,
  brandSurfaceSoft,
  brandText,
  brandTextMuted,
  surfaceCard,
} from "../../../design/responsive/colors";
import { createShadow } from "../../../styles/shadowStyles";
import { CHAT_BUBBLE_OWN, CHAT_BUBBLE_OWN_STRONG } from "../chatPalette";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

const MAX_W_FORM = 512;
const ACTION_SIZE = 50;
const MENU_FLOAT_TOP = 120;
/** Bouton envoi / micro : 44pt (guidelines accessibilité, zone de touche). */
const SEND_DIAM = 44;

/** Ombre très légère — alignée cartes dashboard / courses chauffeur. */
const composerSoftShadow = createShadow({
  shadowColor: "#0F172A",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.035,
  shadowRadius: 14,
  elevation: 1,
});

const composerFocusShadow = createShadow({
  shadowColor: "#0F172A",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.05,
  shadowRadius: 16,
  elevation: 2,
});
export const C_FIELD_TEXT = brandText;
export const C_FIELD_PLACEHOLDER = brandTextMuted;
/** Icône pièce jointe — ardoise pour contraste sur fond blanc. */
export const C_FIELD_ICON = "#64748B";
/** Chips menu pièces jointes (fond blanc). */
const C_MUTED = "#94A3B8";
export const C_BRAND = CHAT_BUBBLE_OWN;
const C_BRAND_STRONG = CHAT_BUBBLE_OWN_STRONG;
/** Micro désactivé (cercle gris clair) — contraste renforcé. */
export const C_ICON_DISABLED = "#64748B";

const C_FIELD_BG = surfaceCard;

/** Hors StyleSheet : évite que les props web-only polluent l’inférence des autres styles. */
export const textInputWebFix = { outlineStyle: "none" as const, outlineWidth: 0 } as unknown as TextStyle;

/** Anneau de focus discret sur le champ (web). */
export function webFieldShellFocusOutline(focused: boolean): ViewStyle {
  if (Platform.OS !== "web") return {};
  if (!focused) {
    return { outlineWidth: 0, outlineStyle: "none" as const };
  }
  return {
    outlineWidth: 2,
    outlineColor: `${brandPrimary}73`,
    outlineStyle: "solid",
    outlineOffset: 2,
  };
}

export { C_MUTED, composerSoftShadow as cardShadow, MENU_FLOAT_TOP };

export const styles = StyleSheet.create({
  root: {
    width: "100%",
    maxWidth: MAX_W_FORM,
    alignSelf: "center",
    zIndex: 1,
    pointerEvents: "box-none",
  },
  mainRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    overflow: "visible",
  },
  fieldShell: {
    flex: 1,
    minWidth: 0,
    minHeight: 48,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: C_FIELD_BG,
    borderRadius: 999,
    borderWidth: 0,
    paddingLeft: 16,
    paddingRight: 6,
    overflow: "visible",
    zIndex: 2,
    ...composerSoftShadow,
  },
  fieldShellFocused: {
    backgroundColor: brandSurfaceSoft,
    ...composerFocusShadow,
  },
  dialMenu: {
    position: "absolute",
    right: 6,
    top: -MENU_FLOAT_TOP,
    flexDirection: "column",
    gap: 6,
    alignItems: "flex-end",
    zIndex: 5,
  },
  textInput: {
    flex: 1,
    minWidth: 0,
    minHeight: 44,
    paddingVertical: 10,
    paddingLeft: 0,
    paddingRight: 4,
    fontSize: FONT_SIZE.px15,
    lineHeight: 22,
    color: C_FIELD_TEXT,
  },
  textInputWithAttach: {
    paddingRight: 2,
  },
  attachBtn: {
    width: 44,
    height: 44,
    minWidth: 44,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
  },
  attachBtnPressed: {
    backgroundColor: "rgba(22, 58, 52, 0.08)",
  },
  actionChip: {
    width: ACTION_SIZE,
    minHeight: ACTION_SIZE,
    paddingVertical: 4,
    paddingHorizontal: 2,
    borderRadius: 12,
    backgroundColor: "#fff",
    borderWidth: 0,
    alignItems: "center",
    justifyContent: "center",
    ...composerSoftShadow,
  },
  /** Typo via `AppText` ; espacement chip uniquement. */
  actionLabel: {
    marginTop: 1,
    textAlign: "center" as const,
  },
  pressedOp: {
    opacity: 0.88,
  },
  sendCircle: {
    width: SEND_DIAM,
    height: SEND_DIAM,
    borderRadius: SEND_DIAM / 2,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 0,
    zIndex: 4,
    ...composerSoftShadow,
  },
  sendCirclePressed: {
    backgroundColor: C_BRAND_STRONG,
  },
  sendCircleBusy: {
    opacity: 0.75,
  },
  sendCircleDisabled: {
    backgroundColor: "#f1f5f9",
    opacity: 1,
    pointerEvents: "none",
    ...Platform.select({
      web: {
        boxShadow: "none",
      },
      ios: {
        shadowOpacity: 0,
        shadowRadius: 0,
        shadowOffset: { width: 0, height: 0 },
      },
      default: {
        elevation: 0,
      },
    }),
  },
});

export const C_RECORDING = "#b91c1c";
export { SEND_DIAM, MAX_W_FORM };
