// styles/profileStyles.ts
import { StyleSheet } from "react-native";

// ✅ Palette épurée et élégante (cohérente avec le login, mission, courses et chat)
const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
  placeholder: "#91A59D",
  error: "#D32F2F",
  errorLight: "#FFEBEE",
};

export const profileStyles = StyleSheet.create({
  // ✅ Container principal
  container: {
    flex: 1,
    backgroundColor: palette.background,
  },

  scrollContainer: {
    flex: 1,
  },

  // ✅ Header élégant avec photo
  headerGradient: {
    backgroundColor: palette.card,
    paddingHorizontal: 28,
    paddingTop: 32,
    paddingBottom: 24,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
    shadowColor: "rgba(16,39,30,0.06)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 2,
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
    fontSize: 28,
    fontWeight: "700",
    color: palette.text,
    marginBottom: 4,
    letterSpacing: -0.5,
  },

  headerPhotoContainer: {
    position: "relative",
    marginLeft: 16,
  },

  headerPhoto: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: palette.border,
    borderWidth: 3,
    borderColor: palette.accent,
    shadowColor: "rgba(10,127,89,0.2)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.16,
    shadowRadius: 6,
    elevation: 3,
  },

  headerPhotoOverlay: {
    position: "absolute",
    bottom: -2,
    right: -2,
    backgroundColor: palette.accent,
    width: 26,
    height: 26,
    borderRadius: 13,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: palette.card,
    shadowColor: "rgba(10,127,89,0.3)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },

  // ✅ Cartes de contenu élégantes
  cardContainer: {
    backgroundColor: palette.card,
    marginHorizontal: 20,
    marginVertical: 10,
    borderRadius: 18,
    padding: 20,
    shadowColor: "rgba(16,39,30,0.06)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
    borderWidth: 1,
    borderColor: palette.border,
  },

  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 20,
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },

  cardTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: palette.text,
    marginLeft: 12,
    flex: 1,
    letterSpacing: -0.3,
  },

  // ✅ Actions
  actionsContainer: {
    paddingHorizontal: 20,
    marginVertical: 16,
  },

  saveButton: {
    backgroundColor: palette.accent,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 14,
    shadowColor: "rgba(10,127,89,0.3)",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4,
    marginBottom: 12,
  },

  saveButtonText: {
    color: palette.card,
    fontSize: 16,
    fontWeight: "600",
    marginLeft: 8,
    letterSpacing: 0.2,
  },

  // ✅ Bouton de déconnexion
  logoutButton: {
    backgroundColor: palette.error,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 14,
    shadowColor: "rgba(211,47,47,0.3)",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4,
  },

  logoutButtonText: {
    color: palette.card,
    fontSize: 16,
    fontWeight: "600",
    marginLeft: 8,
    letterSpacing: 0.2,
  },

  // ✅ Espacement final
  bottomSpacing: {
    height: 80,
  },

  // ✅ Modal de sélection photo
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(15,54,43,0.5)",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },

  modalContainer: {
    backgroundColor: palette.card,
    borderRadius: 20,
    width: "100%",
    maxWidth: 400,
    shadowColor: "rgba(16,39,30,0.2)",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 10,
    borderWidth: 1,
    borderColor: palette.border,
  },

  modalHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 24,
    paddingVertical: 18,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },

  modalTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: palette.text,
    letterSpacing: -0.3,
  },

  modalCloseButton: {
    padding: 4,
  },

  modalContent: {
    padding: 24,
  },

  modalOption: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 18,
    paddingHorizontal: 16,
    borderRadius: 14,
    marginBottom: 12,
    backgroundColor: palette.background,
    borderWidth: 1,
    borderColor: palette.border,
  },

  modalOptionIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: "rgba(10,127,89,0.1)",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 16,
  },

  modalOptionText: {
    fontSize: 16,
    fontWeight: "600",
    color: palette.text,
    flex: 1,
    letterSpacing: -0.2,
  },

  modalOptionSubtext: {
    fontSize: 13,
    color: palette.secondary,
    marginTop: 2,
    lineHeight: 18,
  },

  // ✅ Modal de déconnexion
  logoutModalContainer: {
    backgroundColor: palette.card,
    borderRadius: 20,
    width: "90%",
    maxWidth: 400,
    shadowColor: "rgba(16,39,30,0.2)",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 10,
    borderWidth: 1,
    borderColor: palette.border,
  },

  logoutModalHeader: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 24,
    paddingVertical: 18,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },

  logoutIconContainer: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: palette.errorLight,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 16,
  },

  logoutModalTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: palette.text,
    flex: 1,
    letterSpacing: -0.3,
  },

  logoutModalContent: {
    padding: 24,
  },

  logoutModalMessage: {
    fontSize: 17,
    color: palette.text,
    textAlign: "center",
    marginBottom: 10,
    fontWeight: "600",
    letterSpacing: -0.2,
  },

  logoutModalSubtext: {
    fontSize: 14,
    color: palette.secondary,
    textAlign: "center",
    marginBottom: 28,
    lineHeight: 20,
  },

  logoutModalActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
  },

  logoutCancelButton: {
    flex: 1,
    backgroundColor: palette.background,
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 14,
    alignItems: "center",
    borderWidth: 1,
    borderColor: palette.border,
  },

  logoutCancelButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: palette.text,
    letterSpacing: 0.2,
  },

  logoutConfirmButton: {
    flex: 1,
    backgroundColor: palette.error,
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 14,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    shadowColor: "rgba(211,47,47,0.3)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },

  logoutConfirmButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: palette.card,
    marginLeft: 8,
    letterSpacing: 0.2,
  },
});

