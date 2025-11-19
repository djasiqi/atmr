// styles/chatStyles.ts
import { StyleSheet } from "react-native";

const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
  placeholder: "#91A59D",
  driverMessageBg: "rgba(10,127,89,0.12)",
  driverMessageBorder: "rgba(10,127,89,0.2)",
  companyMessageBg: "rgba(95,115,105,0.08)",
  companyMessageBorder: "rgba(95,115,105,0.15)",
};

export const chatStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: palette.background,
  },

  // --- HEADER ---
  header: {
    width: "100%",
    paddingHorizontal: 28,
    paddingTop: 32,
    paddingBottom: 24,
    backgroundColor: palette.background,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: "700",
    color: palette.text,
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 15,
    color: palette.secondary,
    lineHeight: 22,
  },

  // Liste des messages
  messagesList: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingTop: 16,
    // ⚠️ pas de paddingBottom ici : on le met côté composant
  },

  messageContainer: {
    marginVertical: 2, // Réduit pour éviter les marges excessives
    maxWidth: "85%", // 85% pour plus de largeur
    borderRadius: 18,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderWidth: 1,
    shadowColor: "rgba(16,39,30,0.06)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 2,
  },
  driverMessage: {
    alignSelf: "flex-end",
    backgroundColor: palette.accent,
    borderColor: palette.driverMessageBorder,
  },
  companyMessage: {
    alignSelf: "flex-start",
    backgroundColor: "#FFFFFF",
    borderColor: palette.companyMessageBorder,
  },
  senderName: {
    fontSize: 12,
    fontWeight: "600",
    color: "#DFF6EC", // texte clair sur bulle verte
    marginBottom: 4,
    letterSpacing: 0.2,
  },
  messageContent: {
    fontSize: 15,
    color: palette.text,
    lineHeight: 20,
  },
  messageTextDriver: {
    fontSize: 15,
    color: "#FFFFFF", // texte blanc sur bulle verte
    lineHeight: 20,
  },
  messageTextCompany: {
    fontSize: 15,
    color: palette.text, // texte sombre sur bulle blanche
    lineHeight: 20,
  },

  // Timestamp + tick wrapper
  footerRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    alignItems: "center",
    marginTop: 4,
    gap: 4,
  },

  timestamp: {
    fontSize: 11,
    color: palette.secondary,
    marginTop: 6,
    alignSelf: "flex-end",
  },

  tickIcon: {
    marginLeft: 2,
    opacity: 0.8,
  },

  inputContainer: {
    flexDirection: "row",
    paddingHorizontal: 20,
    paddingTop: 8, // safeArea ajusté dans le composant
    backgroundColor: palette.background,
    borderTopWidth: 1,
    borderTopColor: palette.border,
    alignItems: "center",
  },

  input: {
    flex: 1,
    height: 50,
    paddingHorizontal: 18,
    paddingRight: 50,
    backgroundColor: palette.card,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: palette.border,
    fontSize: 15,
    color: palette.text,
    marginRight: 10,
    shadowColor: "rgba(16,39,30,0.04)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 1,
  },

  inputPlaceholder: {
    color: palette.placeholder,
  },

  sendButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: palette.accent,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: palette.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.24,
    shadowRadius: 8,
    elevation: 4,
  },

  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },

  emptyText: {
    fontSize: 16,
    color: palette.secondary,
    textAlign: "center",
    lineHeight: 24,
  },

  // --- IMAGE MESSAGE ---
  imageMessage: {
    borderRadius: 12,
    overflow: "hidden",
    marginVertical: 2,
  },
  imagePreview: {
    width: "100%",
    maxHeight: 300,
    borderRadius: 12,
  },

  // --- PDF MESSAGE ---
  pdfMessage: {
    flexDirection: "row",
    alignItems: "center",
    padding: 8,
    backgroundColor: "rgba(0,0,0,0.05)",
    borderRadius: 8,
    marginVertical: 2,
  },
  pdfIcon: {
    marginRight: 10,
  },
  pdfInfo: {
    flex: 1,
  },
  pdfFileName: {
    fontSize: 13,
    fontWeight: "600",
    marginBottom: 2,
  },
  pdfFileSize: {
    fontSize: 11,
    opacity: 0.7,
  },

  // --- SCROLL TO BOTTOM BUTTON ---
  scrollToBottomButton: {
    position: "absolute",
    bottom: 100,
    right: 20,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: palette.accent,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },

  // --- TYPING INDICATOR ---
  typingIndicator: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: palette.card,
    borderRadius: 18,
    maxWidth: "85%",
    alignSelf: "flex-start",
    marginVertical: 4,
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: palette.secondary,
    marginHorizontal: 2,
  },
});
