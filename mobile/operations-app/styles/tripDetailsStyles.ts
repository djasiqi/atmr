import { StyleSheet, Platform } from "react-native";

const spacing = { xs: 4, s: 8, m: 12, l: 16, xl: 20 } as const;
const radius = { s: 8, m: 12, l: 16 } as const;

export const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
  placeholder: "#91A59D",
  alertBg: "rgba(255,193,7,0.06)",
  alertBorder: "rgba(255,193,7,0.18)",
  alertText: "#92400E",
} as const;

const sheetShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 -4px 20px rgba(0,0,0,0.12)" }
    : {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.12,
        shadowRadius: 20,
        elevation: 12,
      };

export const styles = StyleSheet.create({
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.4)",
  },

  sheet: {
    backgroundColor: palette.card,
    borderTopLeftRadius: radius.l,
    borderTopRightRadius: radius.l,
    maxHeight: "88%",
    ...sheetShadow,
  },

  handle: {
    alignItems: "center",
    paddingVertical: spacing.s,
  },
  handleBar: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(15,54,43,0.12)",
  },

  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: spacing.l,
    paddingBottom: spacing.m,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: palette.text,
    letterSpacing: -0.3,
  },
  headerClose: {
    padding: spacing.xs,
  },

  scrollContent: {
    paddingHorizontal: spacing.l,
    paddingTop: spacing.m,
    paddingBottom: 40,
  },

  // ——— Sections ———
  section: {
    marginBottom: spacing.l,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: "700",
    color: palette.secondary,
    textTransform: "uppercase",
    letterSpacing: 0.4,
    marginBottom: spacing.s,
  },
  sectionCard: {
    backgroundColor: "rgba(15,54,43,0.025)",
    borderRadius: radius.m,
    borderWidth: 1,
    borderColor: palette.border,
    padding: spacing.m,
  },

  // ——— Rows ———
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    paddingVertical: spacing.xs,
  },
  rowIcon: {
    width: 20,
    alignItems: "center",
    marginRight: spacing.s,
    marginTop: 1,
  },
  rowContent: {
    flex: 1,
  },
  rowLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: palette.secondary,
    letterSpacing: 0.2,
    marginBottom: 1,
  },
  rowValue: {
    fontSize: 14,
    fontWeight: "500",
    color: palette.text,
    lineHeight: 20,
  },
  rowValueSecondary: {
    fontSize: 13,
    color: palette.secondary,
    lineHeight: 18,
    marginTop: 1,
  },

  // ——— Route block ———
  routeBlock: {
    backgroundColor: "rgba(15,54,43,0.025)",
    borderRadius: radius.m,
    borderWidth: 1,
    borderColor: palette.border,
    padding: spacing.m,
  },
  routeRow: {
    flexDirection: "row",
    alignItems: "flex-start",
  },
  routeIndicator: {
    width: 16,
    alignItems: "center",
    marginRight: spacing.s,
    paddingTop: 2,
  },
  routeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  routeDotSquare: {
    width: 8,
    height: 8,
    borderRadius: 2,
  },
  routeLine: {
    width: 1.5,
    height: 14,
    backgroundColor: "rgba(15,54,43,0.15)",
    alignSelf: "center",
    marginVertical: 3,
  },
  routeTextWrap: {
    flex: 1,
    paddingBottom: 2,
  },

  // ——— Access notes (subtle highlight) ———
  accessRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginTop: spacing.xs,
    paddingLeft: 24,
  },
  accessText: {
    fontSize: 12,
    fontWeight: "500",
    color: palette.accent,
    lineHeight: 17,
    fontStyle: "italic",
  },

  // ——— Alert blocks (wheelchair, etc.) ———
  alertBlock: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: palette.alertBg,
    borderRadius: radius.s,
    borderWidth: 1,
    borderColor: palette.alertBorder,
    paddingVertical: spacing.s,
    paddingHorizontal: spacing.m,
    gap: spacing.s,
    marginBottom: spacing.s,
  },
  alertText: {
    fontSize: 13,
    fontWeight: "600",
    color: palette.alertText,
    flex: 1,
  },

  // ——— Notes ———
  notesText: {
    fontSize: 13,
    color: palette.secondary,
    fontStyle: "italic",
    lineHeight: 19,
  },

  // ——— Status badge inline ———
  statusInline: {
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },

  // ——— Divider ———
  divider: {
    height: 1,
    backgroundColor: palette.border,
    marginVertical: spacing.m,
  },

  // ——— Backward compat ———
  container: { backgroundColor: "#FFFFFF", borderRadius: 16, padding: 18, marginHorizontal: 12, marginVertical: 12 },
  title: { fontWeight: "700", fontSize: 18, color: palette.text, marginBottom: 12 },
  rowBetween: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginVertical: 6 },
  label: { fontSize: 14, fontWeight: "600", color: palette.secondary },
  value: { flex: 1, fontSize: 14, color: palette.text, textAlign: "right" },
  sectionHeader: { fontSize: 15, fontWeight: "700", color: "#333", marginBottom: 4 },
  metaText: { fontSize: 13, color: "#616161", marginTop: 2 },
  badge: { backgroundColor: "rgba(10,127,89,0.12)", paddingVertical: 4, paddingHorizontal: 10, borderRadius: 12 },
  badgeText: { fontSize: 12, fontWeight: "700", color: palette.accent },
  actionsRow: { flexDirection: "row", justifyContent: "space-between", flexWrap: "wrap", marginTop: 16, gap: 8 },
  actionButton: { flexGrow: 1, backgroundColor: palette.accent, borderRadius: 12, paddingVertical: 10, paddingHorizontal: 12, alignItems: "center", marginVertical: 4 },
  actionButtonText: { fontSize: 13, color: "#FFFFFF", fontWeight: "600" },
});
