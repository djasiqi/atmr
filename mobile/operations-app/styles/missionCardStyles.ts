import { StyleSheet, Platform } from "react-native";

// ——— Design system : terrain, lisible en ≤3s ———
const spacing = { s: 8, m: 12, l: 16, xl: 24 } as const;
const radius = { m: 12, l: 16 } as const;
export const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
  placeholder: "#91A59D",
  danger: "#dc3545",
  dangerBorder: "rgba(220,53,69,0.2)",
  secondaryAction: "#6c757d",
  secondaryActionBorder: "rgba(108,117,125,0.2)",
  timingDeparture: "#2C3E50",
  timingArrival: "#7F8C8D",
} as const;

const containerShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 8px 24px rgba(16,39,30,0.12)" }
    : {
        shadowColor: "rgba(16,39,30,0.12)",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.14,
        shadowRadius: 24,
        elevation: 8,
      };

const actionShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 4px 8px rgba(10,127,89,0.2)" }
    : {
        shadowColor: palette.accent,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.2,
        shadowRadius: 8,
        elevation: 4,
      };

export const styles = StyleSheet.create({
  // ——— Container ———
  containerEnhanced: {
    backgroundColor: palette.card,
    borderRadius: radius.l,
    padding: spacing.xl,
    marginHorizontal: 20,
    marginVertical: spacing.l,
    marginBottom: 75,
    ...containerShadow,
    borderWidth: 1,
    borderColor: palette.border,
  },
  /** Web : largeur fixe 380px pour aligner card + map */
  containerWebFixed: Platform.OS === "web" ? { width: 380, alignSelf: "center" as const, marginHorizontal: 0 } : {},

  // ——— 1. MissionCardHeader : identité client + statut ———
  headerRowEnhanced: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.l,
  },
  headerClientWrap: {
    flex: 1,
    marginRight: spacing.m,
  },
  clientCivility: {
    fontSize: 12,
    color: palette.secondary,
    marginBottom: 2,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  clientName: {
    fontWeight: "700",
    fontSize: 18,
    color: palette.text,
    letterSpacing: -0.3,
  },
  clientBirthDate: {
    fontSize: 12,
    marginTop: spacing.s,
    color: palette.secondary,
  },
  statusBadgeContainer: {
    backgroundColor: "rgba(10,127,89,0.12)",
    paddingVertical: spacing.s,
    paddingHorizontal: spacing.m,
    borderRadius: radius.m,
    minWidth: 90,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(10,127,89,0.2)",
  },
  statusBadgeText: {
    color: palette.accent,
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.2,
  },

  // ——— 2. MissionTimingBlock : départ / arrivée estimée ———
  timingSection: {
    marginTop: spacing.m,
    marginBottom: spacing.m,
    gap: spacing.s,
  },
  timingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.s,
  },
  timingDepartureColor: {
    color: "#2C3E50",
  },
  timingArrivalColor: {
    color: "#7F8C8D",
  },
  timingDeparture: {
    fontSize: 15,
    fontWeight: "600",
    color: "#2C3E50",
  },
  timingArrival: {
    fontSize: 15,
    fontWeight: "400",
    color: "#7F8C8D",
  },
  timingUnavailable: {
    color: palette.placeholder,
  },

  // ——— 3. MissionRouteBlock : départ → destination (icônes) ———
  routeSection: {
    marginTop: spacing.m,
    marginBottom: spacing.m,
    paddingVertical: spacing.m,
    paddingHorizontal: spacing.l,
    backgroundColor: "rgba(15,54,43,0.03)",
    borderRadius: radius.m,
    borderWidth: 1,
    borderColor: palette.border,
  },
  routeRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: spacing.s,
    gap: spacing.m,
  },
  routeRowLast: {
    marginBottom: 0,
  },
  routeIconWrap: {
    width: 24,
    alignItems: "center",
  },
  routeLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: palette.secondary,
    marginBottom: 2,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  routeAddress: {
    fontSize: 15,
    fontWeight: "500",
    color: palette.text,
    lineHeight: 22,
    flex: 1,
    maxHeight: 44,
  },
  routeContentWrap: {
    flex: 1,
  },
  addressLine: {
    fontSize: 15,
    color: palette.text,
    marginTop: 6,
    lineHeight: 22,
    flexShrink: 1,
    maxWidth: "100%",
  },

  // ——— 4. MissionHintsSection : accès contextuel ———
  metaInfoSection: {
    marginTop: spacing.m,
    marginBottom: spacing.m,
    gap: spacing.s,
  },
  hintsSection: {
    marginTop: spacing.m,
    paddingVertical: spacing.m,
    paddingHorizontal: spacing.l,
    borderRadius: radius.m,
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: "rgba(10,127,89,0.04)",
  },
  hintsSectionTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: palette.text,
    marginBottom: spacing.m,
    letterSpacing: 0.2,
  },
  hintRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: spacing.s,
    gap: spacing.m,
  },
  hintRowLast: {
    marginBottom: 0,
  },
  hintIconWrap: {
    width: 20,
    alignItems: "center",
  },
  hintLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: palette.secondary,
    marginBottom: 2,
  },
  hintValue: {
    fontSize: 14,
    color: palette.text,
    lineHeight: 20,
    flex: 1,
  },

  // ——— 5. MissionNotes ———
  notesBlock: {
    marginTop: spacing.m,
  },
  notesEnhanced: {
    fontSize: 14,
    color: palette.secondary,
    fontStyle: "italic",
    lineHeight: 20,
  },
  notesSeeMoreButton: {
    marginTop: spacing.s,
    minHeight: 44,
    justifyContent: "center",
  },
  notesSeeMoreText: {
    fontSize: 13,
    color: palette.accent,
    fontWeight: "600",
  },
  notesModalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    padding: spacing.xl,
  },
  notesModalCard: {
    backgroundColor: palette.card,
    borderRadius: radius.l,
    padding: spacing.xl,
    maxHeight: "80%",
  },
  notesModalTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: palette.text,
    marginBottom: spacing.m,
    letterSpacing: 0.2,
  },
  notesModalScroll: {
    maxHeight: 300,
  },
  notesModalBody: {
    marginTop: spacing.s,
  },
  notesModalCloseButton: {
    marginTop: spacing.l,
    alignSelf: "flex-end",
    minHeight: 44,
    justifyContent: "center",
    paddingVertical: spacing.s,
    paddingHorizontal: spacing.l,
  },
  notesModalCloseText: {
    fontSize: 15,
    color: palette.accent,
    fontWeight: "600",
  },

  // ——— 6. MissionActionsPrimary ———
  actionsRowEnhanced: {
    flexDirection: "row",
    flexWrap: "nowrap",
    justifyContent: "flex-start",
    alignItems: "stretch",
    marginTop: spacing.xl,
    gap: spacing.m,
  },
  actionItemEnhanced: {
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.accent,
    borderRadius: radius.m,
    paddingVertical: spacing.m,
    paddingHorizontal: spacing.m,
    minHeight: 44,
    flex: 1,
    flexBasis: 0,
    flexGrow: 1,
    flexShrink: 1,
    ...actionShadow,
  },
  actionLabel: {
    fontSize: 12,
    color: "#FFFFFF",
    marginTop: 4,
    textAlign: "center",
    fontWeight: "600",
    letterSpacing: 0.2,
  },
  actionItemMore: {
    alignItems: "center",
    justifyContent: "center",
    minWidth: 44,
    minHeight: 44,
    borderRadius: radius.m,
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: palette.card,
  },

  // ——— Sheet Plus (Détails) : slide depuis le bas, backdrop plus sombre ———
  detailsSheetBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.6)",
    justifyContent: "flex-end",
    padding: 0,
  },
  detailsSheetCard: {
    backgroundColor: palette.card,
    borderTopLeftRadius: radius.l,
    borderTopRightRadius: radius.l,
    padding: spacing.xl,
    paddingBottom: spacing.xl + 24,
    marginHorizontal: 0,
    maxWidth: "100%",
    width: "100%",
  },
  detailsSheetTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: palette.text,
    marginBottom: spacing.m,
    letterSpacing: 0.2,
  },
  detailsSheetItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: spacing.m,
    paddingHorizontal: spacing.l,
    borderRadius: radius.m,
    backgroundColor: "rgba(15,54,43,0.03)",
    marginBottom: spacing.s,
    gap: spacing.m,
  },
  detailsSheetItemText: {
    fontSize: 15,
    fontWeight: "600",
    color: palette.text,
  },
  detailsSheetScroll: {
    maxHeight: 360,
    marginBottom: spacing.m,
  },
  detailsSheetSection: {
    marginBottom: spacing.m,
  },
  detailsSheetSectionTitle: {
    fontSize: 12,
    fontWeight: "700",
    color: palette.secondary,
    marginBottom: spacing.s,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  detailsSheetLine: {
    fontSize: 14,
    color: palette.text,
    marginBottom: 4,
    lineHeight: 20,
  },
  detailsSheetLineLabel: {
    fontWeight: "600",
    color: palette.secondary,
  },

  // ——— Compact mode (réduction hauteur ~25–35%) ———
  containerCompact: {
    paddingVertical: spacing.l,
    paddingHorizontal: spacing.l,
  },
  headerRowCompact: {
    marginBottom: 10,
  },
  timingSectionCompact: {
    marginTop: 6,
    marginBottom: 8,
    gap: 4,
  },
  timingRowCompact: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  timingTextCompact: {
    fontSize: 13,
    fontWeight: "500",
    color: palette.text,
  },
  timingTextSecondaryCompact: {
    fontSize: 13,
    color: palette.secondary,
  },
  routeSectionCompact: {
    marginTop: 6,
    marginBottom: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
  },
  routeRowCompact: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 4,
    gap: 6,
  },
  routeRowCompactLast: {
    marginBottom: 0,
  },
  routeAddressCompact: {
    fontSize: 13,
    fontWeight: "500",
    color: palette.text,
    flex: 1,
    maxHeight: 20,
  },
  metaInfoSectionCompact: {
    marginTop: 6,
    marginBottom: 8,
    gap: 6,
  },
  hintsSectionCompact: {
    marginTop: 4,
    paddingVertical: 6,
    paddingHorizontal: 10,
    marginBottom: 0,
  },
  hintsSectionTitleCompact: {
    fontSize: 12,
    fontWeight: "700",
    color: palette.text,
    marginBottom: 6,
  },
  hintRowCompact: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 2,
    gap: 6,
  },
  hintRowCompactLast: {
    marginBottom: 0,
  },
  hintLineCompact: {
    fontSize: 12,
    color: palette.text,
    flex: 1,
  },
  notesBlockCompact: {
    marginTop: 6,
  },
  notesEnhancedCompact: {
    fontSize: 12,
    lineHeight: 18,
  },
  notesSeeMoreButtonCompact: {
    marginTop: 4,
    minHeight: 44,
  },
  actionsRowCompact: {
    marginTop: 10,
    marginBottom: 0,
  },
  actionsRowSecondaryCompact: {
    marginTop: 8,
  },

  // ——— 7. MissionActionsDanger ———
  actionsRowSecondary: {
    flexDirection: "row",
    flexWrap: "nowrap",
    justifyContent: "center",
    alignItems: "stretch",
    marginTop: spacing.m,
    gap: spacing.m,
  },
  actionItemSecondary: {
    flex: 1,
    maxWidth: "48%",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.secondaryAction,
    borderRadius: radius.m,
    paddingVertical: spacing.m,
    paddingHorizontal: spacing.m,
    minHeight: 44,
    ...(Platform.OS === "web"
      ? { boxShadow: "0 2px 6px rgba(108,117,125,0.25)" }
      : { elevation: 3, shadowColor: palette.secondaryAction, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.25, shadowRadius: 6 }),
  },
  actionItemDanger: {
    flex: 1,
    maxWidth: "48%",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.danger,
    borderRadius: radius.m,
    paddingVertical: spacing.m,
    paddingHorizontal: spacing.m,
    minHeight: 44,
    ...(Platform.OS === "web"
      ? { boxShadow: "0 2px 6px rgba(220,53,69,0.25)" }
      : { elevation: 3, shadowColor: palette.danger, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.25, shadowRadius: 6 }),
  },

  // ——— Legacy / compat ———
  detailText: {
    fontSize: 15,
    color: palette.text,
    marginLeft: 0,
    marginBottom: 6,
    marginTop: 4,
    flexShrink: 1,
    lineHeight: 22,
  },
  infoEnhanced: {
    fontSize: 13,
    fontWeight: "600",
    color: palette.secondary,
    marginTop: 4,
    letterSpacing: 0.2,
  },
  routeSectionLegacy: {
    marginTop: spacing.s,
    marginBottom: spacing.s,
    paddingVertical: spacing.s,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: "#E0E0E0",
  },
  timeRow: {
    flexDirection: "row",
    alignItems: "center",
    marginLeft: 12,
    minWidth: 60,
  },
  timeEnhanced: {
    fontWeight: "600",
    fontSize: 15,
    color: palette.text,
    marginLeft: 4,
    letterSpacing: 0.1,
  },
  medicalInfoSection: {
    backgroundColor: "rgba(10,127,89,0.06)",
    borderLeftWidth: 3,
    borderLeftColor: palette.accent,
    borderRadius: radius.m,
    padding: spacing.l,
    marginTop: spacing.m,
    marginBottom: spacing.s,
    borderWidth: 1,
    borderColor: palette.border,
  },
  medicalTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: palette.text,
    marginBottom: spacing.s,
    letterSpacing: 0.2,
  },
  medicalDetail: {
    fontSize: 14,
    color: palette.secondary,
    marginLeft: 0,
    marginBottom: 4,
    lineHeight: 20,
  },
  wheelchairSection: {
    backgroundColor: "rgba(255,193,7,0.08)",
    borderLeftWidth: 3,
    borderLeftColor: "#FFC107",
    borderRadius: radius.m,
    padding: spacing.l,
    marginTop: spacing.m,
    marginBottom: spacing.s,
    borderWidth: 1,
    borderColor: "rgba(255,193,7,0.15)",
  },
  wheelchairAlert: {
    fontSize: 14,
    fontWeight: "700",
    color: "#8B6914",
    marginBottom: 4,
    letterSpacing: 0.1,
  },
  notesHint: {
    fontSize: 13,
    color: palette.placeholder,
    marginTop: spacing.s,
    lineHeight: 20,
  },

  // ——— EmptyState ———
  emptyStateContainer: {
    backgroundColor: palette.card,
    borderRadius: radius.l,
    padding: spacing.xl,
    marginHorizontal: 20,
    marginVertical: spacing.l,
    marginBottom: 75,
    ...containerShadow,
    borderWidth: 1,
    borderColor: palette.border,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyStateWebFixed: Platform.OS === "web" ? { width: 380, alignSelf: "center" as const, marginHorizontal: 0 } : {},
  emptyStateTitle: {
    fontSize: 18,
    textAlign: "center",
    color: palette.text,
    fontWeight: "600",
    letterSpacing: 0.2,
  },
  emptyStateSubtitle: {
    fontSize: 15,
    textAlign: "center",
    color: palette.secondary,
    marginTop: spacing.m,
    lineHeight: 22,
  },
});
