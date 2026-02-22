import React from "react";
import { Modal, View, TouchableOpacity, Text, StyleSheet, Pressable, Platform } from "react-native";
import { Ionicons } from "@expo/vector-icons";

const BRAND = "#00796b";
const TXT = "#0f172a";
const TXT_SEC = "#6b7280";
const BORDER = "#e5e7eb";
const CARD = "#FFFFFF";

type Props = {
  visible: boolean;
  onClose: () => void;
  onPickCamera: () => void;
  onPickGallery: () => void;
  onPickDocument: () => void;
};

export default function AttachmentSheet({ visible, onClose, onPickCamera, onPickGallery, onPickDocument }: Props) {
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={st.root}>
        <Pressable style={st.overlay} onPress={onClose} />
        <View style={st.sheet}>
          <View style={st.handle} />

          <Text style={st.title}>Envoyer un fichier</Text>

          <View style={st.grid}>
            <ActionItem icon="camera" label="Caméra" color="#00796b" bg="rgba(0,121,107,0.08)" onPress={onPickCamera} />
            <ActionItem icon="images" label="Galerie" color="#7c3aed" bg="rgba(124,58,237,0.08)" onPress={onPickGallery} />
            <ActionItem icon="document-text" label="Document" color="#ea580c" bg="rgba(234,88,12,0.08)" onPress={onPickDocument} />
          </View>

          <TouchableOpacity onPress={onClose} style={st.cancelBtn} activeOpacity={0.7}>
            <Text style={st.cancelText}>Annuler</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

function ActionItem({ icon, label, color, bg, onPress }: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  color: string;
  bg: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={st.actionItem} onPress={onPress} activeOpacity={0.7}>
      <View style={[st.actionIcon, { backgroundColor: bg }]}>
        <Ionicons name={icon} size={24} color={color} />
      </View>
      <Text style={st.actionLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

const sheetShadow = Platform.OS === "web"
  ? { boxShadow: "0 -4px 20px rgba(0,0,0,0.08)" }
  : { shadowColor: "#000", shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.08, shadowRadius: 20, elevation: 10 };

const st = StyleSheet.create({
  root: { flex: 1, justifyContent: "flex-end" },
  overlay: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.3)" },
  sheet: {
    backgroundColor: CARD,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 24,
    paddingTop: 8,
    paddingBottom: 32,
    ...sheetShadow,
  },
  handle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#d1d5db",
    alignSelf: "center",
    marginBottom: 16,
  },
  title: {
    fontSize: 16,
    fontWeight: "700",
    color: TXT,
    marginBottom: 20,
    letterSpacing: -0.2,
  },
  grid: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginBottom: 24,
  },
  actionItem: { alignItems: "center", gap: 8 },
  actionIcon: {
    width: 56,
    height: 56,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
  },
  actionLabel: { fontSize: 12, fontWeight: "600", color: TXT_SEC },
  cancelBtn: {
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: "#f4f4f5",
    alignItems: "center",
  },
  cancelText: { color: TXT_SEC, fontSize: 14, fontWeight: "600" },
});
