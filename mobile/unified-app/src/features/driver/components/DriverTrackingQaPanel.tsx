import React from "react";
import { StyleSheet, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import type { DriverBackgroundTrackingUiState } from "../hooks/useDriverBackgroundTrackingUi";
import { useTrackingState } from "../hooks/useTrackingState";

type Props = {
  ui: DriverBackgroundTrackingUiState;
};

function formatTs(ts: number | null | undefined): string {
  if (ts == null) return "never";
  return new Date(ts).toLocaleTimeString("fr-CH");
}

function shortId(id: string | null | undefined): string {
  if (id == null || id.length === 0) return "null";
  if (id.length <= 24) return id;
  return `…${id.slice(-20)}`;
}

export function DriverTrackingQaPanel({ ui }: Props) {
  const tracking = useTrackingState();

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
        Native phase: {ui.nativeStartPhase ?? "none"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Native error: {ui.nativeStartError ?? "none"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        TM defined: {ui.nativeTaskDefined == null ? "?" : ui.nativeTaskDefined ? "yes" : "no"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Started before/after:{" "}
        {ui.nativeStartedBefore == null ? "?" : ui.nativeStartedBefore ? "1" : "0"} /{" "}
        {ui.nativeStartedAfter == null ? "?" : ui.nativeStartedAfter ? "1" : "0"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        Last invoked: {formatTs(ui.lastTaskInvokedAt)}
      </AppText>
      <AppText variant="label" style={styles.section}>
        Q1 ACK (bridge) — source de vérité RCA
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        lastAckStatus: {tracking.lastAckStatus ?? "null"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        lastAckError: {tracking.lastAckError ?? "null"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        lastAckSeq: {tracking.lastAckAttemptSeq ?? "null"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        lastAckEventId: {tracking.lastAckEventId ?? "null"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        currentSeq: {tracking.currentAttemptSeq}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        currentEventId: {tracking.currentAttemptEventId ?? "null"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        queueDepth: {tracking.queueDepth}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        lastAckIsQueued: {tracking.lastAckIsQueued ? "yes" : "no"}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        lastAckAt: {formatTs(tracking.lastAckAt)}
      </AppText>
      <AppText variant="bodyMuted" style={styles.line}>
        ids(short): cur={shortId(tracking.currentAttemptEventId)} ack=
        {shortId(tracking.lastAckEventId)}
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
  section: {
    marginTop: 10,
    marginBottom: 4,
    fontWeight: "600",
  },
  line: {
    fontSize: 12,
    fontFamily: "monospace",
  },
});
