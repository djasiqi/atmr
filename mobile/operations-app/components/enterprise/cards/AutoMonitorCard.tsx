import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, ViewStyle } from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { EnterpriseCard } from "./EnterpriseCard";

type DecisionLog = {
  id: string;
  summary: string;
  timestamp: string;
};

type AutoMonitorProps = {
  lastDecision?: {
    bookingId: string;
    driverName: string;
    eta: string;
    confidence: number;
  } | null;
  anomalies?: string[];
  onPause?: () => void;
  onResume?: () => void;
  paused?: boolean;
  onOpenLogs?: () => void;
  logs?: DecisionLog[];
  style?: ViewStyle;
};

export const AutoMonitorCard: React.FC<AutoMonitorProps> = ({
  lastDecision,
  anomalies = [],
  paused = false,
  onPause,
  onResume,
  onOpenLogs,
  logs = [],
  style,
}) => {
  return (
    <EnterpriseCard style={[styles.card, style]}>
      <View style={styles.header}>
        <Text style={styles.title}>Moniteur automatique</Text>
        <TouchableOpacity
          style={[styles.pauseButton, paused && styles.pauseButtonPaused]}
          onPress={paused ? onResume : onPause}
        >
          <Ionicons
            name={paused ? "play" : "pause"}
            size={16}
            color={paused ? "#0F172A" : "#0F172A"}
            style={{ marginRight: 4 }}
          />
          <Text style={styles.pauseLabel}>{paused ? "Relancer" : "Pause"}</Text>
        </TouchableOpacity>
      </View>

      {lastDecision ? (
        <View style={styles.decisionCard}>
          <Text style={styles.decisionTitle}>Dernière décision</Text>
          <View style={styles.decisionLine}>
            <Ionicons name="people-outline" size={16} color="#5EEAD4" />
            <Text style={styles.decisionText}>
              {lastDecision.bookingId} → {lastDecision.driverName}
            </Text>
          </View>
          <View style={styles.decisionLine}>
            <Ionicons name="navigate-outline" size={16} color="#93C5FD" />
            <Text style={styles.decisionText}>ETA {lastDecision.eta}</Text>
            <Text style={styles.confidence}>Score {lastDecision.confidence}%</Text>
          </View>
        </View>
      ) : (
        <Text style={styles.placeholder}>Encore aucune décision enregistrée aujourd’hui.</Text>
      )}

      {anomalies.length ? (
        <View style={styles.anomalyBlock}>
          <View style={styles.anomalyHeader}>
            <Ionicons name="warning-outline" size={16} color="#FBBF24" />
            <Text style={styles.anomalyTitle}>Alertes</Text>
          </View>
          {anomalies.map((anomaly) => (
            <Text key={anomaly} style={styles.anomalyItem}>
              • {anomaly}
            </Text>
          ))}
        </View>
      ) : null}

      <TouchableOpacity style={styles.logButton} onPress={onOpenLogs}>
        <Text style={styles.logButtonText}>Journal des décisions ({logs.length})</Text>
        <Ionicons name="chevron-forward" size={16} color="#93C5FD" />
      </TouchableOpacity>
    </EnterpriseCard>
  );
};

const styles = StyleSheet.create({
  card: {
    gap: 14,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  title: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 16,
  },
  pauseButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#60A5FA",
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  pauseButtonPaused: {
    backgroundColor: "#FBBF24",
  },
  pauseLabel: {
    color: "#0B1736",
    fontWeight: "700",
  },
  decisionCard: {
    backgroundColor: "rgba(35,53,110,0.75)",
    borderRadius: 16,
    padding: 14,
    gap: 8,
  },
  decisionTitle: {
    color: "rgba(214,224,255,0.8)",
    fontSize: 12,
    letterSpacing: 0.2,
    textTransform: "uppercase",
  },
  decisionLine: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  decisionText: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "500",
  },
  confidence: {
    color: "#5EEAD4",
    fontWeight: "700",
    marginLeft: "auto",
  },
  placeholder: {
    color: "rgba(184,196,240,0.7)",
    fontStyle: "italic",
  },
  anomalyBlock: {
    backgroundColor: "rgba(124,58,18,0.15)",
    borderRadius: 12,
    padding: 12,
    gap: 6,
  },
  anomalyHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  anomalyTitle: {
    color: "#FBBF24",
    fontWeight: "700",
    fontSize: 13,
  },
  anomalyItem: {
    color: "rgba(255,232,200,0.85)",
    fontSize: 12,
  },
  logButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(111,139,255,0.2)",
  },
  logButtonText: {
    color: "#93C5FD",
    fontWeight: "600",
  },
});

export default AutoMonitorCard;

