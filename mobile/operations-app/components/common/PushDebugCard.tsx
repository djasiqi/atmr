/**
 * Push Debug Card — dev-only
 * Affiche l'état complet pour diagnostic "app kill" : permissions, canaux missions/missions_v2, device.
 */
import React, { useState, useEffect, useCallback } from "react";
import { View, Text, TouchableOpacity, StyleSheet, Platform } from "react-native";
import { getKillModeState, KillModeState } from "@/services/notificationChannels";

export function PushDebugCard() {
  const [state, setState] = useState<KillModeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const s = await getKillModeState();
      setState(s);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (Platform.OS !== "android") return;
    refresh();
  }, [refresh]);

  if (Platform.OS !== "android" || !__DEV__) return null;
  if (loading && !state) return null;

  const readyColor =
    state?.ready === "✓"
      ? "#0A7F59"
      : state?.ready?.includes("missions_v2 non créé")
        ? "#E67E22"
        : "#C0392B";

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={[styles.header, { borderLeftColor: readyColor }]}
        onPress={() => setExpanded((e) => !e)}
        activeOpacity={0.8}
      >
        <Text style={styles.title}>🔔 Push Debug (app kill)</Text>
        <Text style={[styles.ready, { color: readyColor }]}>{state?.ready ?? "?"}</Text>
        <Text style={styles.expand}>{expanded ? "▼" : "▶"}</Text>
      </TouchableOpacity>
      {expanded && state && (
        <View style={styles.body}>
          <Row label="Platform" value={`${state.platform} ${state.androidVersion}`} />
          <Row label="Device" value={`${state.manufacturer} / ${state.model}`} />
          <Row label="App" value={state.appOwnership} />
          <Row
            label="Permissions"
            value={`${state.permissions.status} ${state.permissions.granted ? "✓" : "✗"}`}
          />
          <Row
            label="missions"
            value={
              state.missions.exists
                ? `exists importance=${state.missions.importance} high=${state.missions.isHigh} sound=${state.missions.hasSound} vib=${state.missions.hasVibration}`
                : "absent"
            }
          />
          <Row
            label="missions_v2"
            value={
              state.missions_v2.exists
                ? `exists importance=${state.missions_v2.importance} high=${state.missions_v2.isHigh} sound=${state.missions_v2.hasSound} vib=${state.missions_v2.hasVibration}`
                : "absent — ouvrir app une fois"
            }
          />
          <TouchableOpacity style={styles.refreshBtn} onPress={refresh}>
            <Text style={styles.refreshText}>Rafraîchir</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}:</Text>
      <Text style={styles.rowValue} numberOfLines={2}>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 16,
    marginTop: 12,
    backgroundColor: "#f8f9fa",
    borderRadius: 8,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#e9ecef",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    padding: 10,
    borderLeftWidth: 4,
    gap: 8,
  },
  title: {
    fontSize: 12,
    fontWeight: "600",
    color: "#1a1a1a",
  },
  ready: {
    fontSize: 11,
    fontWeight: "700",
    flex: 1,
  },
  expand: {
    fontSize: 10,
    color: "#888",
  },
  body: {
    padding: 10,
    paddingTop: 0,
    borderTopWidth: 1,
    borderTopColor: "#e9ecef",
  },
  row: {
    flexDirection: "row",
    marginBottom: 4,
    gap: 6,
  },
  rowLabel: {
    fontSize: 11,
    color: "#6c757d",
    width: 80,
  },
  rowValue: {
    fontSize: 11,
    color: "#1a1a1a",
    flex: 1,
  },
  refreshBtn: {
    marginTop: 8,
    paddingVertical: 6,
    paddingHorizontal: 12,
    backgroundColor: "#e9ecef",
    borderRadius: 6,
    alignSelf: "flex-start",
  },
  refreshText: {
    fontSize: 11,
    color: "#495057",
    fontWeight: "500",
  },
});
