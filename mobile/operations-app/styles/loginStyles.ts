// src/styles/loginStyles.ts
import { StyleSheet } from "react-native";

export type LoginMode = "driver" | "enterprise";

const palettes: Record<
  LoginMode,
  {
    background: string;
    card: string;
    text: string;
    secondary: string;
    accent: string;
    border: string;
    placeholder: string;
  }
> = {
  driver: {
    background: "#F5F7F6",
    card: "#FFFFFF",
    text: "#15362B",
    secondary: "#5F7369",
    accent: "#0A7F59",
    border: "rgba(15,54,43,0.08)",
    placeholder: "#91A59D",
  },
  enterprise: {
    background: "#0B1510",
    card: "rgba(16,27,22,0.95)",
    text: "#E6F2EA",
    secondary: "#8AA295",
    accent: "#0A7A4D",
    border: "rgba(26,164,112,0.22)",
    placeholder: "#6F8E81",
  },
};

export const getLoginStyles = (mode: LoginMode) => {
  const palette = palettes[mode];

  const styles = StyleSheet.create({
    safeArea: {
      flex: 1,
      backgroundColor: palette.background,
    },
    container: {
      flex: 1,
      paddingHorizontal: 28,
      paddingVertical: 32,
      justifyContent: "center",
    },
    card: {
      backgroundColor: palette.card,
      borderRadius: 24,
      padding: 28,
      borderWidth: 1,
      borderColor: palette.border,
      shadowColor:
        mode === "driver" ? "rgba(16,39,30,0.12)" : "rgba(0,0,0,0.45)",
      shadowOffset: { width: 0, height: 28 },
      shadowOpacity: mode === "driver" ? 0.14 : 0.32,
      shadowRadius: 40,
      elevation: mode === "driver" ? 8 : 14,
    },
    header: {
      marginBottom: 28,
    },
    kicker: {
      color: palette.accent,
      fontSize: 12,
      letterSpacing: 3,
      textTransform: "uppercase",
      fontWeight: "600",
      marginBottom: 10,
    },
    title: {
      fontSize: 28,
      fontWeight: "700",
      color: palette.text,
    },
    subtitle: {
      fontSize: 15,
      color: palette.secondary,
      marginTop: 10,
      lineHeight: 22,
    },
    form: {
      marginTop: 12,
    },
    inputBlock: {
      marginBottom: 16,
    },
    label: {
      fontSize: 13,
      fontWeight: "600",
      letterSpacing: 0.2,
      color: palette.secondary,
      marginBottom: 6,
    },
    input: {
      height: 50,
      borderRadius: 14,
      borderWidth: 1,
      borderColor: palette.border,
      backgroundColor: mode === "driver" ? "#FFFFFF" : "rgba(255,255,255,0.03)",
      paddingHorizontal: 18,
      paddingRight: 48,
      fontSize: 16,
      color: palette.text,
    },
    passwordField: {
      position: "relative",
    },
    eyeButton: {
      position: "absolute",
      right: 5,
      top: 0,
      bottom: 0,
      width: 44,
      alignItems: "center",
      justifyContent: "center",
    },
    helperLink: {
      alignSelf: "flex-end",
      marginBottom: 26,
    },
    helperLinkText: {
      color: palette.accent,
      fontWeight: "600",
      letterSpacing: 0.2,
    },
    primaryButton: {
      backgroundColor: "#00796B",
      borderRadius: 16,
      paddingVertical: 16,
      alignItems: "center",
      shadowColor: "#00796B",
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.24,
      shadowRadius: 18,
      elevation: 6,
      marginBottom: 22,
    },
    primaryButtonText: {
      color: "#FFFFFF",
      fontSize: 16,
      fontWeight: "600",
      letterSpacing: 0.3,
    },
    switchRow: {
      marginTop: 8,
      alignItems: "center",
    },
    switchPrompt: {
      color: palette.secondary,
      fontSize: 14,
      marginBottom: 6,
      textAlign: "center",
    },
    switchLink: {
      color: palette.accent,
      fontWeight: "600",
      fontSize: 14,
      textAlign: "center",
    },
  });

  return { styles, palette };
};
