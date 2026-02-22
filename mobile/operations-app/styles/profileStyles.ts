import { StyleSheet, Platform } from "react-native";

const BRAND = "#00796B";
const BRAND_DARK = "#00695C";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const BG = "#f4f7fc";
const CARD = "#FFFFFF";
const DANGER = "#dc3545";

const cardShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 2px 8px rgba(0,0,0,0.04)" }
    : {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.04,
        shadowRadius: 8,
        elevation: 2,
      };

const btnShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 2px 6px rgba(0,121,107,0.2)" }
    : {
        shadowColor: BRAND,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 6,
        elevation: 3,
      };

const sheetShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 -4px 24px rgba(0,0,0,0.12)" }
    : {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.1,
        shadowRadius: 16,
        elevation: 12,
      };

export const profileStyles = StyleSheet.create({
  // ——— Layout ———
  container: {
    flex: 1,
    backgroundColor: BG,
  },
  scrollContainer: {
    flex: 1,
  },

  // ——— Header ———
  headerGradient: {
    backgroundColor: CARD,
    paddingHorizontal: 20,
    paddingTop: Platform.OS === "ios" ? 56 : 44,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  headerContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerText: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: "700",
    color: TEXT,
    letterSpacing: -0.3,
  },
  headerSubtitle: {
    fontSize: 13,
    color: TEXT_SEC,
    marginTop: 4,
  },
  headerPhotoContainer: {
    position: "relative",
    marginLeft: 16,
  },
  headerPhoto: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: BORDER,
    borderWidth: 2.5,
    borderColor: BRAND,
  },
  headerPhotoOverlay: {
    position: "absolute",
    bottom: -2,
    right: -2,
    backgroundColor: BRAND,
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: CARD,
  },

  // ——— Section cards ———
  cardContainer: {
    backgroundColor: CARD,
    marginHorizontal: 16,
    marginTop: 12,
    borderRadius: 14,
    padding: 16,
    ...cardShadow,
    borderWidth: 1,
    borderColor: BORDER,
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 14,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: TEXT,
    marginLeft: 10,
    flex: 1,
    letterSpacing: -0.1,
  },
  cardDesc: {
    fontSize: 13,
    color: TEXT_SEC,
    marginTop: 2,
    lineHeight: 18,
  },

  // ——— Read-only info rows ———
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(0,121,107,0.05)",
  },
  infoRowLast: {
    borderBottomWidth: 0,
  },
  infoLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: TEXT_SEC,
    letterSpacing: 0.2,
    textTransform: "uppercase",
    flex: 1,
  },
  infoValue: {
    fontSize: 14,
    fontWeight: "500",
    color: TEXT,
    textAlign: "right",
    flex: 1.5,
  },
  infoValueMuted: {
    color: TEXT_MUTED,
    fontStyle: "italic",
  },
  infoBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    backgroundColor: "rgba(0,121,107,0.06)",
  },
  infoBadgeText: {
    fontSize: 12,
    fontWeight: "600",
    color: BRAND,
  },
  infoBadgeWarn: {
    backgroundColor: "rgba(245,158,11,0.1)",
  },
  infoBadgeWarnText: {
    color: "#92400e",
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    justifyContent: "flex-end",
    flex: 1.5,
  },
  chip: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    backgroundColor: "rgba(0,121,107,0.06)",
  },
  chipText: {
    fontSize: 12,
    fontWeight: "600",
    color: BRAND,
  },

  // ——— Actions ———
  actionsContainer: {
    paddingHorizontal: 16,
    marginTop: 16,
    gap: 10,
  },
  saveButton: {
    backgroundColor: BRAND,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    gap: 8,
    ...btnShadow,
  },
  saveButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
  logoutButton: {
    backgroundColor: DANGER,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    gap: 8,
    ...(Platform.OS === "web"
      ? { boxShadow: "0 2px 6px rgba(220,53,69,0.2)" }
      : {
          shadowColor: DANGER,
          shadowOffset: { width: 0, height: 2 },
          shadowOpacity: 0.2,
          shadowRadius: 6,
          elevation: 3,
        }),
  },
  logoutButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },

  bottomSpacing: {
    height: 80,
  },

  // ——— Modal backdrop (shared) ———
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.45)",
    justifyContent: "flex-end",
  },

  // ——— Photo modal (bottom sheet) ———
  modalContainer: {
    backgroundColor: CARD,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingBottom: Platform.OS === "ios" ? 32 : 16,
    ...sheetShadow,
  },
  modalHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: TEXT,
    letterSpacing: -0.2,
  },
  modalCloseButton: {
    width: 28,
    height: 28,
    borderRadius: 6,
    alignItems: "center",
    justifyContent: "center",
  },
  modalContent: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 4,
  },
  modalOption: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 14,
    paddingHorizontal: 14,
    borderRadius: 12,
    marginBottom: 8,
    backgroundColor: BG,
    borderWidth: 1,
    borderColor: BORDER,
  },
  modalOptionIcon: {
    width: 42,
    height: 42,
    borderRadius: 12,
    backgroundColor: "rgba(0,121,107,0.06)",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 14,
  },
  modalOptionText: {
    fontSize: 14,
    fontWeight: "600",
    color: TEXT,
  },
  modalOptionSubtext: {
    fontSize: 12,
    color: TEXT_SEC,
    marginTop: 2,
    lineHeight: 17,
  },

  // ——— Logout / Switch modal (bottom sheet) ———
  logoutModalContainer: {
    backgroundColor: CARD,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: Platform.OS === "ios" ? 36 : 20,
    alignItems: "center",
    ...sheetShadow,
  },
  logoutIconContainer: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: "rgba(220,53,69,0.08)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
  },
  logoutModalTitle: {
    fontSize: 17,
    fontWeight: "600",
    color: TEXT,
    textAlign: "center",
    letterSpacing: -0.2,
  },
  logoutModalMessage: {
    fontSize: 13,
    color: TEXT_SEC,
    textAlign: "center",
    lineHeight: 19,
    marginTop: 8,
    maxWidth: 300,
  },
  logoutModalActions: {
    flexDirection: "row",
    gap: 10,
    width: "100%",
    marginTop: 20,
  },
  logoutCancelButton: {
    flex: 1,
    height: 42,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: BG,
    borderWidth: 1,
    borderColor: "rgba(0,0,0,0.08)",
  },
  logoutCancelButtonText: {
    fontSize: 14,
    fontWeight: "500",
    color: TEXT_SEC,
  },
  logoutConfirmButton: {
    flex: 1.2,
    height: 42,
    borderRadius: 10,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: DANGER,
    gap: 6,
  },
  logoutConfirmButtonText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#fff",
  },
});
