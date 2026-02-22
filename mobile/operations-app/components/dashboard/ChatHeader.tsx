import React from "react";
import { View, Text, StyleSheet, Platform } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useSocketStatus } from "@/hooks/useSocketStatus";

const BRAND = "#00796b";
const TXT = "#0f172a";
const TXT_SEC = "#6b7280";
const BORDER = "#e5e7eb";
const CARD = "#FFFFFF";

export default function ChatHeader() {
  const { connected, reconnecting } = useSocketStatus();

  const statusColor = reconnecting ? "#f59e0b" : connected ? "#16a34a" : "#ef4444";
  const statusBg = reconnecting ? "rgba(245,158,11,0.1)" : connected ? "rgba(22,163,74,0.1)" : "rgba(239,68,68,0.1)";
  const statusText = reconnecting ? "Reconnexion" : connected ? "En ligne" : "Hors ligne";

  return (
    <View style={st.container}>
      <View style={st.topRow}>
        <View style={{ flex: 1 }}>
          <Text style={st.title}>Équipe</Text>
          <Text style={st.subtitle}>Discussion en temps réel</Text>
        </View>
        <View style={[st.statusPill, { backgroundColor: statusBg }]}>
          <View style={[st.statusDot, { backgroundColor: statusColor }]} />
          <Text style={[st.statusLabel, { color: statusColor }]}>{statusText}</Text>
        </View>
      </View>
    </View>
  );
}

const st = StyleSheet.create({
  container: {
    backgroundColor: CARD,
    paddingTop: Platform.OS === "ios" ? 52 : 40,
    paddingBottom: 12,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  topRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: TXT,
    letterSpacing: -0.3,
  },
  subtitle: {
    fontSize: 12,
    color: TXT_SEC,
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
});
