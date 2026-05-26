import React from "react";
import { StyleSheet, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import type { DriverBackgroundTrackingUiState } from "../hooks/useDriverBackgroundTrackingUi";

type Props = {
  ui: DriverBackgroundTrackingUiState;
};

function formatTs(ts: number | null): string {
  if (ts == null) return "never";
  return new Date(ts).toLocaleTimeString("fr-CH");
}

export function DriverTrackingQaPanel({ ui }: Props) {
  return (
    <View style={styles.panel}>
      <AppText variant="label" style={styles.title}>
        Tracking QA
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Task defined: {ui.taskDefined ? "yes" : "no"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Task started: {ui.taskStarted ? "yes" : "no"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        BG flag: {ui.bgFlagEnabled ? "yes" : "no"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Runtime: {ui.runtime}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Pending FGS: {ui.pendingFgsStart ? "yes" : "no"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Last error: {ui.lastNativeStartError ?? "none"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Last invoked: {formatTs(ui.lastTaskInvokedAt)}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    marginBottom: 12,
    padding: 12,
    borderRadius: 8,
    backgroundColor: "#F1F5F9",
    borderWidth: 1,
    borderColor: "#CBD5E1",
  },
  title: {
    marginBottom: 6,
    fontWeight: "600",
  },
  line: {
    fontSize: 12,
    fontFamily: "monospace",
  },
});
