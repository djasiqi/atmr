import { Platform, type TextStyle, type ViewStyle, StyleSheet } from "react-native";
import {
  borderDefault,
  borderStrong,
  brandPrimary,
  brandSurfaceSoft,
  brandText,
  brandTextMuted,
  surfaceCard,
} from "../../../design/responsive/colors";
import { CHAT_BUBBLE_OWN, CHAT_BUBBLE_OWN_STRONG } from "../chatPalette";

const MAX_W_FORM = 512;
const ACTION_SIZE = 50;
const MENU_FLOAT_TOP = 120;
/** Bouton envoi / micro : 44pt (guidelines accessibilité, zone de touche). */
const SEND_DIAM = 44;

/** Champ saisie : coquille claire alignée `AppInput` / tokens shell. */
const C_FIELD_BG = surfaceCard;
const C_FIELD_BORDER = borderDefault;
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

const cardShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 1px 2px 0 rgba(0,0,0,0.08)" as const }
    : {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
        elevation: 3,
      };

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

export { C_MUTED, cardShadow, MENU_FLOAT_TOP };

export const styles = StyleSheet.create({
  root: {
    width: "100%",
    maxWidth: MAX_W_FORM,
    alignSelf: "center",
    zIndex: 1,
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
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: C_FIELD_BORDER,
    paddingLeft: 14,
    paddingRight: 4,
    overflow: "visible",
    zIndex: 2,
    ...Platform.select({
      web: {
        boxShadow: "0 1px 2px rgba(15, 23, 42, 0.05), 0 4px 14px rgba(15, 23, 42, 0.05)",
      },
      ios: {
        shadowColor: "#0f172a",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.07,
        shadowRadius: 6,
      },
      default: {
        elevation: 2,
      },
    }),
  },
  fieldShellFocused: {
    backgroundColor: brandSurfaceSoft,
    borderColor: borderStrong,
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
    fontSize: 15,
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
    borderRadius: 8,
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: C_FIELD_BORDER,
    alignItems: "center",
    justifyContent: "center",
    ...cardShadow,
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
    borderWidth: 1,
    ...Platform.select({
      web: {
        boxShadow: "0 1px 2px 0 rgba(0,0,0,0.08)" as const,
      },
      default: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.12,
        shadowRadius: 2,
        elevation: 3,
      },
    }),
  },
  sendCirclePressed: {
    backgroundColor: C_BRAND_STRONG,
    borderColor: C_BRAND_STRONG,
  },
  sendCircleDisabled: {
    backgroundColor: "#f1f5f9",
    borderColor: "rgba(148, 163, 184, 0.55)",
    opacity: 1,
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
