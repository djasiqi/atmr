import { StyleSheet, Platform } from "react-native";

const palette = {
  background: "#f4f7fc",
  card: "#FFFFFF",
  text: "#0f172a",
  secondary: "#6b7280",
  accent: "#00796b",
  accentDark: "#00695c",
  border: "#e5e7eb",
  placeholder: "#9ca3af",
  ownBubble: "#00796b",
  otherBubble: "#FFFFFF",
};

const msgShadow = Platform.OS === "web"
  ? { boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }
  : { shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.06, shadowRadius: 3, elevation: 1 };

const inputShadow = Platform.OS === "web"
  ? { boxShadow: "0 1px 4px rgba(0,0,0,0.04)" }
  : { shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 4, elevation: 1 };

const sendShadow = Platform.OS === "web"
  ? { boxShadow: "0 2px 6px rgba(0,121,107,0.3)" }
  : { shadowColor: palette.accent, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 6, elevation: 3 };

const scrollBtnShadow = Platform.OS === "web"
  ? { boxShadow: "0 2px 6px rgba(0,0,0,0.2)" }
  : { shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6, elevation: 4 };

export const chatStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: palette.background },

  header: {
    width: "100%",
    paddingHorizontal: 20,
    paddingTop: Platform.OS === "ios" ? 52 : 40,
    paddingBottom: 12,
    backgroundColor: palette.card,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },
  headerTitle: { fontSize: 20, fontWeight: "700", color: palette.text, letterSpacing: -0.3 },
  headerSubtitle: { fontSize: 12, color: palette.secondary, marginTop: 1 },

  messagesList: { flexGrow: 1, paddingHorizontal: 16, paddingTop: 12 },

  messageContainer: {
    marginVertical: 1,
    maxWidth: "80%",
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderWidth: 1,
    ...msgShadow,
  },
  driverMessage: {
    alignSelf: "flex-end",
    backgroundColor: palette.ownBubble,
    borderColor: "rgba(0,121,107,0.2)",
  },
  companyMessage: {
    alignSelf: "flex-start",
    backgroundColor: palette.otherBubble,
    borderColor: palette.border,
  },
  senderName: { fontSize: 11, fontWeight: "700", color: palette.accent, marginBottom: 2, letterSpacing: 0.1 },
  messageContent: { fontSize: 15, color: palette.text, lineHeight: 20 },
  messageTextDriver: { fontSize: 15, color: "#FFFFFF", lineHeight: 20 },
  messageTextCompany: { fontSize: 15, color: palette.text, lineHeight: 20 },

  footerRow: { flexDirection: "row", justifyContent: "flex-end", alignItems: "center", marginTop: 2, gap: 3 },
  timestamp: { fontSize: 10, color: palette.secondary, alignSelf: "flex-end", marginTop: 4 },
  tickIcon: { marginLeft: 2, opacity: 0.7 },

  inputContainer: {
    flexDirection: "row",
    paddingHorizontal: 16,
    paddingTop: 8,
    backgroundColor: palette.card,
    borderTopWidth: 1,
    borderTopColor: palette.border,
    alignItems: "center",
  },
  input: {
    flex: 1,
    height: 44,
    paddingHorizontal: 16,
    paddingRight: 44,
    backgroundColor: palette.background,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: palette.border,
    fontSize: 14,
    color: palette.text,
    marginRight: 8,
    ...inputShadow,
  },
  inputPlaceholder: { color: palette.placeholder },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: palette.accent,
    justifyContent: "center",
    alignItems: "center",
    ...sendShadow,
  },

  emptyContainer: { flex: 1, justifyContent: "center", alignItems: "center", paddingHorizontal: 24 },
  emptyText: { fontSize: 14, color: palette.secondary, textAlign: "center", lineHeight: 22 },

  imageMessage: { borderRadius: 10, overflow: "hidden", marginVertical: 2 },
  imagePreview: { width: "100%", maxHeight: 280, borderRadius: 10 },

  pdfMessage: {
    flexDirection: "row",
    alignItems: "center",
    padding: 10,
    backgroundColor: "rgba(0,121,107,0.06)",
    borderRadius: 10,
    marginVertical: 2,
    borderWidth: 1,
    borderColor: "rgba(0,121,107,0.12)",
  },
  pdfIcon: { marginRight: 10 },
  pdfInfo: { flex: 1 },
  pdfFileName: { fontSize: 13, fontWeight: "600", color: palette.text, marginBottom: 1 },
  pdfFileSize: { fontSize: 11, color: palette.secondary },

  scrollToBottomButton: {
    position: "absolute",
    bottom: 100,
    right: 16,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: palette.card,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: palette.border,
    ...scrollBtnShadow,
  },

  typingIndicator: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: palette.card,
    borderRadius: 16,
    maxWidth: "60%",
    alignSelf: "flex-start",
    marginVertical: 4,
    borderWidth: 1,
    borderColor: palette.border,
  },
  typingDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: palette.secondary, marginHorizontal: 2 },
});
