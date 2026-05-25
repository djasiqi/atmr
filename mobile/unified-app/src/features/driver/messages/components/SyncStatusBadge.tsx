import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { SyncPresenceStatus } from "../types";

const LABELS: Record<SyncPresenceStatus, { dot: string; label: string; color: string }> = {
  connected: { dot: "🟢", label: "Connecté", color: "#047857" },
  slow: { dot: "🟠", label: "Synchronisation lente", color: "#b45309" },
  offline: { dot: "🔴", label: "Hors ligne", color: "#b91c1c" },
};

type Props = { status: SyncPresenceStatus };

export function SyncStatusBadge({ status }: Props) {
  const meta = LABELS[status];
  return (
    <View style={styles.wrap} accessibilityRole="text" accessibilityLabel={meta.label}>
      <AppText variant="caption" style={[styles.text, { color: meta.color }]}>
        {meta.dot} {meta.label}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { alignSelf: "flex-start" },
  text: { fontWeight: "600" },
});
