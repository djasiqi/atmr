import { StyleSheet, Platform } from "react-native";

export const palette = {
  brand: "#00796b",
  brandDark: "#00695c",
  brandLight: "#e0f2f1",
  card: "#FFFFFF",
  text: "#0f172a",
  secondary: "#6b7280",
  accent: "#0A7F59",
  border: "#e5e7eb",
  background: "#f4f7fc",
  connected: "#16a34a",
  reconnecting: "#f59e0b",
  disconnected: "#ef4444",
};

export const styles = StyleSheet.create({
  container: {
    backgroundColor: palette.card,
    paddingTop: Platform.OS === "ios" ? 52 : 40,
    paddingBottom: 12,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },

  topRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },

  greeting: {
    fontSize: 20,
    fontWeight: "700",
    color: palette.text,
    letterSpacing: -0.3,
  },

  greetingName: {
    color: palette.brand,
  },

  dateText: {
    fontSize: 12,
    color: palette.secondary,
    marginTop: 1,
  },

  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 14,
    gap: 5,
  },

  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },

  statusLabel: {
    fontSize: 11,
    fontWeight: "600",
    letterSpacing: 0.1,
  },

  summaryRow: {
    flexDirection: "row",
    gap: 8,
  },

  summaryCard: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: palette.background,
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: palette.border,
  },

  summaryIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 7,
    alignItems: "center",
    justifyContent: "center",
  },

  summaryValue: {
    fontSize: 15,
    fontWeight: "700",
    color: palette.text,
    letterSpacing: -0.2,
  },

  summaryLabel: {
    fontSize: 10,
    color: palette.secondary,
    letterSpacing: 0.1,
  },
});
