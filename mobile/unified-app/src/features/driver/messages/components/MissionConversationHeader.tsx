import { Pressable, StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { SyncStatusBadge } from "./SyncStatusBadge";
import { PriorityBadge } from "./PriorityBadge";
import type { MessageHubThread, MessagePriority, SyncPresenceStatus } from "../types";
import { useRouter } from "expo-router";

type Props = {
  thread: MessageHubThread;
  syncStatus: SyncPresenceStatus;
  etaMinutes?: number | null;
  kmRemaining?: number | null;
  onOpenDetails?: () => void;
  onOpenFiles?: () => void;
};

export function MissionConversationHeader({
  thread,
  syncStatus,
  etaMinutes,
  kmRemaining,
  onOpenDetails,
  onOpenFiles,
}: Props) {
  const router = useRouter();
  const isMission = Boolean(thread.booking_id);
  const statusLabel = (thread.status ?? "").replace(/_/g, " ");

  return (
    <View style={styles.card}>
      <View style={styles.topRow}>
        <SyncStatusBadge status={syncStatus} />
        <PriorityBadge priority={(thread.priority as MessagePriority) ?? "normal"} />
      </View>
      <AppText variant="sectionTitle" style={styles.patient}>
        {thread.title}
      </AppText>
      {thread.booking_id ? (
        <AppText variant="bodyMuted" style={styles.sub}>
          Mission #{thread.booking_id}
        </AppText>
      ) : thread.subtitle ? (
        <AppText variant="bodyMuted" style={styles.sub}>
          {thread.subtitle}
        </AppText>
      ) : null}
      {isMission ? (
        <View style={styles.metaRow}>
          {statusLabel ? (
            <AppText variant="label" style={styles.badge}>
              {statusLabel}
            </AppText>
          ) : null}
          {thread.scheduled_time ? (
            <AppText variant="caption" style={styles.meta}>
              {new Date(thread.scheduled_time).toLocaleTimeString("fr-FR", {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </AppText>
          ) : null}
          {kmRemaining != null ? (
            <AppText variant="caption" style={styles.meta}>
              {kmRemaining.toFixed(1)} km restants
            </AppText>
          ) : null}
          {etaMinutes != null ? (
            <AppText variant="caption" style={styles.eta}>
              ETA {etaMinutes} min
            </AppText>
          ) : null}
        </View>
      ) : null}
      {(thread.pickup_location || thread.dropoff_location) && isMission ? (
        <View style={styles.route}>
          {thread.pickup_location ? (
            <AppText variant="caption" numberOfLines={1}>
              Départ · {thread.pickup_location}
            </AppText>
          ) : null}
          {thread.dropoff_location ? (
            <AppText variant="caption" numberOfLines={1}>
              Arrivée · {thread.dropoff_location}
            </AppText>
          ) : null}
        </View>
      ) : null}
      <View style={styles.actions}>
        {onOpenDetails ? (
          <Pressable onPress={onOpenDetails} style={styles.actionBtn}>
            <AppText variant="caption" style={styles.actionText}>
              Détails
            </AppText>
          </Pressable>
        ) : null}
        {onOpenFiles ? (
          <Pressable onPress={onOpenFiles} style={styles.actionBtn}>
            <AppText variant="caption" style={styles.actionText}>
              Fichiers & position
            </AppText>
          </Pressable>
        ) : null}
        {thread.booking_id ? (
          <Pressable
            onPress={() => router.push(`/(app)/(driver)/missions/${thread.booking_id}`)}
            style={styles.actionBtn}
          >
            <AppText variant="caption" style={styles.actionText}>
              Mission complète
            </AppText>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#e5e7eb",
    gap: 6,
  },
  topRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  patient: { color: "#111827" },
  sub: { color: "#6b7280" },
  metaRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, alignItems: "center" },
  badge: {
    backgroundColor: "#ecfdf5",
    color: "#047857",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
    overflow: "hidden",
  },
  meta: { color: "#6b7280" },
  eta: { color: "#047857", fontWeight: "700" },
  route: { gap: 2 },
  actions: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 4 },
  actionBtn: {
    backgroundColor: "#f3f4f6",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },
  actionText: { color: "#374151", fontWeight: "600" },
});
