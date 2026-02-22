import { StyleSheet, Platform } from "react-native";

const spacing = { s: 8, m: 12, l: 16, xl: 24 } as const;
const radius = { m: 12, l: 16 } as const;
export const palette = {
  background: "#f4f7fc",
  card: "#FFFFFF",
  text: "#1E293B",
  secondary: "#64748B",
  accent: "#00796B",
  accentDark: "#00695C",
  accentLight: "#26a69a",
  border: "rgba(0,121,107,0.08)",
  placeholder: "#94A3B8",
  danger: "#dc3545",
  dangerBorder: "rgba(220,53,69,0.2)",
  secondaryAction: "#6c757d",
  secondaryActionBorder: "rgba(108,117,125,0.2)",
  timingDeparture: "#1E293B",
  timingArrival: "#64748B",
  routePickup: "#00796B",
  routeDropoff: "#1E293B",
} as const;

const containerShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 4px 16px rgba(0,0,0,0.06)" }
    : {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
      };

const actionShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 4px 8px rgba(0,121,107,0.2)" }
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
    flexWrap: "wrap",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: spacing.l,
  },
  headerClientWrap: {
    flex: 1,
    flexBasis: 0,
    minWidth: 120,
    marginRight: spacing.m,
    overflow: "visible",
  },
  clientCivility: {
    fontSize: 11,
    color: palette.secondary,
    marginBottom: 2,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    fontWeight: "600",
  },
  clientName: {
    fontWeight: "700",
    fontSize: 17,
    color: palette.text,
    letterSpacing: -0.2,
  },
  clientBirthDate: {
    fontSize: 12,
    color: palette.secondary,
  },
  headerBadgesWrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.s,
    flexWrap: "wrap",
    flexShrink: 1,
    minWidth: 0,
    marginLeft: "auto",
  },
  statusBadgeContainer: {
    backgroundColor: "rgba(0,121,107,0.08)",
    paddingVertical: 6,
    paddingHorizontal: spacing.m,
    borderRadius: 8,
    minWidth: 70,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(0,121,107,0.15)",
  },
  statusBadgeText: {
    color: palette.accent,
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  /** Badge type mission : Livraison (couleur distincte) */
  deliveryTypeBadge: {
    backgroundColor: "rgba(245,158,11,0.15)",
    borderColor: "rgba(245,158,11,0.35)",
  },
  deliveryTypeBadgeText: {
    color: "#B45309",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  /** Ligne description livraison (sous le header) */
  deliveryDescRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.m,
    paddingVertical: spacing.s,
    paddingHorizontal: spacing.m,
    backgroundColor: "rgba(245,158,11,0.06)",
    borderRadius: radius.m,
    borderWidth: 1,
    borderColor: "rgba(245,158,11,0.15)",
  },
  deliveryDescRowCompact: {
    marginBottom: spacing.s,
    paddingVertical: spacing.s,
    paddingHorizontal: spacing.m,
  },
  deliveryDescLabel: {
    fontSize: 13,
    fontWeight: "600",
    color: "#B45309",
  },
  deliveryDescText: {
    fontSize: 13,
    color: palette.text,
    flex: 1,
    flexShrink: 1,
  },

  // ——— 2. MissionTimingBlock ———
  timingSection: {
    marginTop: spacing.s,
    marginBottom: spacing.m,
    gap: 6,
  },
  timingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.s,
  },
  timingDepartureColor: {
    color: palette.timingDeparture,
  },
  timingArrivalColor: {
    color: palette.timingArrival,
  },
  timingDeparture: {
    fontSize: 14,
    fontWeight: "600",
    color: palette.timingDeparture,
  },
  timingArrival: {
    fontSize: 14,
    fontWeight: "400",
    color: palette.timingArrival,
  },
  timingUnavailable: {
    color: palette.placeholder,
  },

  // ——— 3. MissionRouteBlock : départ → destination (timeline) ———
  routeSection: {
    marginTop: spacing.m,
    marginBottom: spacing.m,
    paddingVertical: 14,
    paddingHorizontal: spacing.l,
    backgroundColor: "rgba(0,121,107,0.03)",
    borderRadius: radius.m,
    borderWidth: 1,
    borderColor: "rgba(0,121,107,0.08)",
  },
  routeRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: spacing.m,
  },
  routeRowLast: {
    // no margin needed — connector provides spacing
  },
  routeTimelineWrap: {
    alignItems: "center",
    width: 24,
  },
  routeDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2.5,
    borderColor: palette.routePickup,
    backgroundColor: "#fff",
  },
  routeDotDropoff: {
    borderColor: palette.routeDropoff,
    backgroundColor: palette.routeDropoff,
  },
  routeConnector: {
    width: 2,
    height: 28,
    backgroundColor: "rgba(0,121,107,0.15)",
    marginVertical: 4,
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
    letterSpacing: 0.5,
  },
  routeAddress: {
    fontSize: 14,
    fontWeight: "500",
    color: palette.text,
    lineHeight: 20,
    flex: 1,
    maxHeight: 40,
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
    borderColor: "rgba(0,121,107,0.08)",
    backgroundColor: "rgba(0,121,107,0.03)",
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
    marginTop: spacing.l,
    gap: spacing.s,
  },
  actionItemEnhanced: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.accent,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    minHeight: 40,
    flex: 1,
    flexBasis: 0,
    flexGrow: 1,
    flexShrink: 1,
    gap: 6,
    ...actionShadow,
  },
  actionLabel: {
    fontSize: 12,
    color: "#FFFFFF",
    textAlign: "center",
    fontWeight: "600",
    letterSpacing: 0.2,
  },
  actionItemMore: {
    alignItems: "center",
    justifyContent: "center",
    minWidth: 40,
    minHeight: 40,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: palette.card,
  },

  // Legacy compat — kept for notes modal
  detailsSheetBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.35)",
    justifyContent: "flex-end",
    padding: 0,
  },
  detailsSheetCard: {
    backgroundColor: palette.card,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: spacing.xl,
    paddingBottom: spacing.xl + 24,
    maxWidth: "100%",
    width: "100%",
  },

  // ——— Compact mode (réduction hauteur ~25–35%) ———
  containerCompact: {
    paddingVertical: spacing.l,
    paddingHorizontal: spacing.l,
    marginTop: spacing.s,
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
    marginTop: spacing.s,
    gap: spacing.s,
  },
  actionItemSecondary: {
    flex: 1,
    maxWidth: "48%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.secondaryAction,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    minHeight: 40,
    gap: 6,
    ...(Platform.OS === "web"
      ? { boxShadow: "0 2px 4px rgba(108,117,125,0.2)" }
      : { elevation: 2, shadowColor: palette.secondaryAction, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.2, shadowRadius: 4 }),
  },
  actionItemDanger: {
    flex: 1,
    maxWidth: "48%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.danger,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    minHeight: 40,
    gap: 6,
    ...(Platform.OS === "web"
      ? { boxShadow: "0 2px 4px rgba(220,53,69,0.2)" }
      : { elevation: 2, shadowColor: palette.danger, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.2, shadowRadius: 4 }),
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
    backgroundColor: "rgba(0,121,107,0.04)",
    borderLeftWidth: 3,
    borderLeftColor: palette.accent,
    borderRadius: radius.m,
    padding: spacing.l,
    marginTop: spacing.m,
    marginBottom: spacing.s,
    borderWidth: 1,
    borderColor: "rgba(0,121,107,0.08)",
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
    padding: 32,
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
  emptyStateIconWrap: {
    width: 64,
    height: 64,
    borderRadius: 18,
    backgroundColor: "rgba(0,121,107,0.06)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 20,
  },
  emptyStateTitle: {
    fontSize: 18,
    textAlign: "center",
    color: palette.text,
    fontWeight: "700",
    letterSpacing: -0.2,
  },
  emptyStateSubtitle: {
    fontSize: 14,
    textAlign: "center",
    color: palette.secondary,
    marginTop: spacing.s,
    lineHeight: 22,
    maxWidth: 260,
  },
  emptyStateBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    marginTop: spacing.l,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    backgroundColor: "rgba(22,163,74,0.08)",
  },
  emptyStateBadgeText: {
    fontSize: 13,
    fontWeight: "600",
    color: "#16a34a",
    letterSpacing: 0.1,
  },
});

// ——— Detail Sheet — Bottom Sheet séparé pour éviter la limite TS de StyleSheet.create ———
const sheetShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 -4px 24px rgba(0,0,0,0.12)" }
    : { shadowColor: "#000", shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.12, shadowRadius: 24, elevation: 12 };

export const dsStyles = StyleSheet.create({
  dsRoot: { flex: 1, justifyContent: "flex-end" },
  dsOverlay: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.35)" },
  dsSheet: { backgroundColor: palette.card, borderTopLeftRadius: 20, borderTopRightRadius: 20, height: "65%", overflow: "hidden", ...sheetShadow },
  dsHandle: { width: 36, height: 4, borderRadius: 2, backgroundColor: "#D1D5DB", alignSelf: "center", marginTop: 10, marginBottom: 6 },
  dsHeaderBar: { flexDirection: "row", alignItems: "center", gap: 10, paddingHorizontal: 20, paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: palette.border },
  dsHeaderIcon: { width: 36, height: 36, borderRadius: 10, backgroundColor: "rgba(0,121,107,0.08)", alignItems: "center", justifyContent: "center" },
  dsHeaderTitle: { fontSize: 16, fontWeight: "700", color: palette.text },
  dsHeaderSub: { fontSize: 12, color: palette.secondary, marginTop: 2 },
  dsStatusBadge: { flexDirection: "row", alignItems: "center", gap: 5, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8, backgroundColor: "rgba(0,121,107,0.08)" },
  dsStatusText: { fontSize: 11, fontWeight: "600", color: palette.accent },
  dsScroll: { flex: 1 },
  dsScrollContent: { paddingHorizontal: 20, paddingTop: 14, paddingBottom: 20 },
  dsFooter: { flexDirection: "row", gap: 10, paddingHorizontal: 20, paddingVertical: 14, borderTopWidth: 1, borderTopColor: palette.border },
  dsFooterBtn: { flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, paddingVertical: 13, borderRadius: 12, borderWidth: 1, borderColor: palette.border, backgroundColor: palette.card },
  dsFooterBtnText: { fontSize: 14, fontWeight: "600", color: palette.accent },
  dsCard: { backgroundColor: "rgba(0,121,107,0.02)", borderRadius: radius.m, borderWidth: 1, borderColor: palette.border, marginBottom: 10, overflow: "hidden" },
  dsCardHeader: { flexDirection: "row", alignItems: "center", gap: 6, paddingHorizontal: spacing.m, paddingVertical: 8, backgroundColor: "rgba(0,121,107,0.05)", borderBottomWidth: 1, borderBottomColor: palette.border },
  dsCardTitle: { fontSize: 11, fontWeight: "700", color: palette.secondary, textTransform: "uppercase", letterSpacing: 0.4 },
  dsCardBody: { paddingHorizontal: 14, paddingVertical: 10 },
  dsMainText: { fontSize: 13, fontWeight: "500", color: palette.text, lineHeight: 19 },
  dsSecText: { fontSize: 12, color: palette.secondary, lineHeight: 17 },
  dsPhoneText: { fontSize: 13, fontWeight: "600", color: palette.accent },
  dsChipRow: { flexDirection: "row", alignItems: "center", gap: 4, marginTop: 4 },
  dsMetricChip: { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "rgba(0,121,107,0.06)", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
  dsMetricText: { fontSize: 12, fontWeight: "600", color: palette.accent },
  dsReturnChip: { flexDirection: "row", alignItems: "center", gap: 3, backgroundColor: "rgba(245,158,11,0.10)", paddingHorizontal: 7, paddingVertical: 3, borderRadius: 6 },
  dsRouteLabel: { fontSize: 10, fontWeight: "600", color: palette.placeholder, textTransform: "uppercase", letterSpacing: 0.3, marginBottom: 2 },
  dsInfoRow: { flexDirection: "row", alignItems: "flex-start", gap: 8, marginBottom: 8 },
  dsInfoLabel: { fontSize: 11, fontWeight: "600", color: palette.placeholder, textTransform: "uppercase", letterSpacing: 0.2, marginBottom: 1 },
});
