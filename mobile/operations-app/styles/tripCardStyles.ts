import { StyleSheet, Platform } from "react-native";

const spacing = { xs: 4, s: 6, m: 10, l: 14 } as const;
const radius = { s: 8, m: 12 } as const;

export const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
  placeholder: "#91A59D",

  statusAssigned: "#0A7F59",
  statusEnRoute: "#D97706",
  statusInProgress: "#2563EB",
  statusCompleted: "#5F7369",
  statusCanceled: "#DC2626",
} as const;

const cardShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 2px 8px rgba(16,39,30,0.06)" }
    : {
        shadowColor: "rgba(16,39,30,0.08)",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08,
        shadowRadius: 8,
        elevation: 2,
      };

export const tripCardStyles = StyleSheet.create({
  cardContainer: {
    backgroundColor: palette.card,
    borderRadius: radius.m,
    marginHorizontal: spacing.l,
    marginVertical: 4,
    borderWidth: 1,
    borderColor: palette.border,
    overflow: "hidden",
    ...cardShadow,
  },
  cardInner: {
    flexDirection: "row",
  },
  statusBar: {
    width: 3,
  },
  cardContent: {
    flex: 1,
    paddingVertical: spacing.m,
    paddingHorizontal: spacing.l,
  },

  topRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.s,
  },
  timeText: {
    fontSize: 13,
    fontWeight: "700",
    color: palette.text,
    letterSpacing: -0.2,
    marginRight: spacing.m,
    minWidth: 40,
  },
  clientName: {
    fontSize: 14,
    fontWeight: "600",
    color: palette.text,
    flex: 1,
    letterSpacing: -0.2,
  },
  statusLabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },

  routeRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  routeIndicator: {
    width: 14,
    alignItems: "center",
    marginRight: spacing.s,
  },
  dotPickup: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: palette.accent,
  },
  dotDropoff: {
    width: 6,
    height: 6,
    borderRadius: 1.5,
    backgroundColor: palette.text,
  },
  routeAddress: {
    fontSize: 13,
    fontWeight: "500",
    color: palette.secondary,
    flex: 1,
    lineHeight: 18,
  },

  routeSpacer: {
    height: 3,
  },

  bottomRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: spacing.s,
    gap: spacing.s,
  },
  badge: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 2,
    paddingHorizontal: spacing.s,
    borderRadius: 5,
    borderWidth: 1,
    gap: 3,
  },
  badgeText: {
    fontSize: 10,
    fontWeight: "600",
    letterSpacing: 0.2,
  },
  distanceText: {
    fontSize: 11,
    fontWeight: "500",
    color: palette.placeholder,
    marginLeft: "auto",
  },

  sectionHeader: {
    fontSize: 11,
    fontWeight: "700",
    paddingTop: spacing.l,
    paddingBottom: spacing.xs,
    marginHorizontal: spacing.l + 4,
    color: palette.secondary,
    letterSpacing: 0.4,
    textTransform: "uppercase",
  },

  emptyText: {
    marginTop: 20,
    marginHorizontal: spacing.l,
    color: palette.secondary,
    fontSize: 14,
    textAlign: "center",
    lineHeight: 20,
  },
  placeholderCard: {
    backgroundColor: palette.card,
    borderRadius: radius.m,
    paddingVertical: 20,
    paddingHorizontal: spacing.l,
    marginHorizontal: spacing.l,
    marginVertical: 4,
    alignItems: "center",
    borderWidth: 1,
    borderColor: palette.border,
    borderStyle: "dashed",
  },
  placeholderTitle: {
    fontSize: 14,
    color: palette.text,
    textAlign: "center",
    fontWeight: "600",
  },
  placeholderSubtitle: {
    fontSize: 12,
    color: palette.secondary,
    textAlign: "center",
    marginTop: spacing.xs,
    lineHeight: 18,
  },

  // Backward compat
  routeSection: { marginLeft: 0, marginBottom: 8, marginTop: 4, fontSize: 15, color: palette.text, lineHeight: 22 },
  routeText: { fontSize: 16, fontWeight: "600", color: palette.text, marginBottom: 10, lineHeight: 24, letterSpacing: -0.2 },
  statusText: { fontSize: 14, fontWeight: "600", marginTop: 8, color: palette.secondary, letterSpacing: 0.1 },
  statusBadge: { backgroundColor: "rgba(10,127,89,0.12)", color: palette.accent, paddingVertical: 6, paddingHorizontal: 14, borderRadius: 16, fontSize: 13, fontWeight: "700", alignSelf: "flex-start" as const, marginTop: 12, borderWidth: 1, borderColor: "rgba(10,127,89,0.2)", letterSpacing: 0.2 },
  timeEnhanced: { fontSize: 15, fontWeight: "600", marginTop: 8, color: palette.text, letterSpacing: 0.1 },
});
