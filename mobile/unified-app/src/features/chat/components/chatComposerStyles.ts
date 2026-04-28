import { Platform, type TextStyle, StyleSheet } from "react-native";

const MAX_W_FORM = 512;
const ACTION_SIZE = 50;
const MENU_FLOAT_TOP = 112;
/** Bouton envoi / micro : 44pt (guidelines accessibilité, zone de touche). */
const SEND_DIAM = 44;

const C_FIELD_BG = "#f3f4f6";
const C_FIELD_BORDER = "#e5e7eb";
const C_TEXT = "#111827";
const C_MUTED = "#6b7280";
export const C_BRAND = "#0d9488";
const C_BRAND_STRONG = "#0f766e";

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

export { C_TEXT, C_MUTED, cardShadow, MENU_FLOAT_TOP };

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
    gap: 8,
    overflow: "visible",
  },
  fieldShell: {
    flex: 1,
    minWidth: 0,
    minHeight: 44,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: C_FIELD_BG,
    borderRadius: 8,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: C_FIELD_BORDER,
    paddingLeft: 12,
    paddingRight: 2,
    overflow: "visible",
    zIndex: 2,
  },
  fieldShellFocused: {
    backgroundColor: "#e8eaed",
    borderColor: "rgba(0,0,0,0.06)",
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
    minHeight: 40,
    paddingVertical: 10,
    paddingLeft: 0,
    paddingRight: 4,
    fontSize: 14,
    lineHeight: 20,
    color: C_TEXT,
  },
  textInputWithAttach: {
    paddingRight: 2,
  },
  attachBtn: {
    minWidth: 40,
    minHeight: 40,
    paddingHorizontal: 6,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 6,
  },
  attachBtnPressed: {
    backgroundColor: "rgba(0,0,0,0.05)",
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
  actionLabel: {
    fontSize: 9,
    fontWeight: "600",
    color: C_TEXT,
    marginTop: 1,
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
    backgroundColor: C_MUTED,
    borderColor: C_MUTED,
    opacity: 0.9,
  },
});

export const C_RECORDING = "#b91c1c";
export { SEND_DIAM, MAX_W_FORM };
