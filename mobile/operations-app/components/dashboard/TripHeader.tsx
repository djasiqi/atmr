import React from "react";
import { View, Text, StyleSheet, Platform } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useSocketStatus } from "@/hooks/useSocketStatus";
import { formatTimeLocal } from "@/utils/formatTimeLocal";

const BRAND = "#00796B";
const TXT = "#0f172a";
const TXT_SEC = "#6b7280";
const BORDER = "#e5e7eb";
const CARD = "#FFFFFF";
const BG = "#f4f7fc";

type Props = {
  date?: string | null;
  totalTrips?: number;
  doneTrips?: number;
  remainingTrips?: number;
};

export default function TripHeader({
  date,
  totalTrips = 0,
  doneTrips = 0,
  remainingTrips = 0,
}: Props) {
  const { connected, reconnecting } = useSocketStatus();
  const safeDate = typeof date === "string" ? date : String(date ?? "");

  const statusColor = reconnecting ? "#f59e0b" : connected ? "#16a34a" : "#ef4444";
  const statusBg = reconnecting ? "rgba(245,158,11,0.1)" : connected ? "rgba(22,163,74,0.1)" : "rgba(239,68,68,0.1)";
  const statusText = reconnecting ? "Reconnexion" : connected ? "En ligne" : "Hors ligne";

  return (
    <View style={st.container}>
      <View style={st.topRow}>
        <View style={{ flex: 1 }}>
          <Text style={st.title}>Courses</Text>
          <Text style={st.date}>{safeDate}</Text>
        </View>
        <View style={[st.statusPill, { backgroundColor: statusBg }]}>
          <View style={[st.statusDot, { backgroundColor: statusColor }]} />
          <Text style={[st.statusLabel, { color: statusColor }]}>{statusText}</Text>
        </View>
      </View>

      <View style={st.summaryRow}>
        <View style={st.summaryCard}>
          <View style={st.summaryIconWrap}>
            <Ionicons name="car-outline" size={14} color={BRAND} />
          </View>
          <View>
            <Text style={st.summaryValue}>{remainingTrips}</Text>
            <Text style={st.summaryLabel}>{remainingTrips <= 1 ? "Course" : "Courses"}</Text>
          </View>
        </View>
        <View style={st.summaryCard}>
          <View style={st.summaryIconWrap}>
            <Ionicons name="time-outline" size={14} color={BRAND} />
          </View>
          <View>
            <Text style={st.summaryValue}>
              {formatTimeLocal(new Date())}
            </Text>
            <Text style={st.summaryLabel}>Heure</Text>
          </View>
        </View>
      </View>

      {totalTrips > 0 && (
        <View style={st.progressRow}>
          <View style={st.progressTrack}>
            <View style={[st.progressFill, { width: `${Math.round((doneTrips / totalTrips) * 100)}%` }]} />
          </View>
          <Text style={st.progressLabel}>{Math.round((doneTrips / totalTrips) * 100)}%</Text>
        </View>
      )}
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
    marginBottom: 10,
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: TXT,
    letterSpacing: -0.3,
  },
  date: {
    fontSize: 12,
    color: TXT_SEC,
    marginTop: 1,
    textTransform: "capitalize",
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
  summaryRow: {
    flexDirection: "row",
    gap: 8,
  },
  summaryCard: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: BG,
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: BORDER,
  },
  summaryIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 7,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#e0f2f1",
  },
  summaryValue: {
    fontSize: 15,
    fontWeight: "700",
    color: TXT,
    letterSpacing: -0.2,
  },
  summaryLabel: {
    fontSize: 10,
    color: TXT_SEC,
    letterSpacing: 0.1,
  },
  progressRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 10,
  },
  progressTrack: {
    flex: 1,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(0,121,107,0.1)",
    overflow: "hidden",
  },
  progressFill: {
    height: 4,
    borderRadius: 2,
    backgroundColor: BRAND,
  },
  progressLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: BRAND,
    minWidth: 28,
    textAlign: "right",
  },
});
