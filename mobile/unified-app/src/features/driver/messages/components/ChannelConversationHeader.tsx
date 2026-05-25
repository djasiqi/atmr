import { Pressable, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import { getThreadDisplayLines } from "../inboxDisplay";
import { InboxThreadAvatar } from "./InboxThreadAvatar";
import { EmergencyFab } from "./EmergencyFab";
import type { EmergencyIssueType, MessageHubThread, SyncPresenceStatus } from "../types";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  thread: MessageHubThread;
  syncStatus: SyncPresenceStatus;
  /** Canal équipe uniquement — `null` = chargement. */
  memberCount?: number | null;
  /** Mission : ETA affiché après « Connecté ». */
  etaMinutes?: number | null;
  onOpenColleagues?: () => void;
  onOpenDetails?: () => void;
  /** Entreprise : ouvre gestion du canal (dispatch). */
  onPressHeader?: () => void;
  /** Affiche chevron si le header est cliquable. */
  headerManageable?: boolean;
  /** Dispatch / mission : signalement urgence dans le header (droite). */
  emergency?: {
    onReport: (issue: EmergencyIssueType) => void;
    pending?: boolean;
  };
};

function syncLabel(status: SyncPresenceStatus): string {
  if (status === "connected") return "Connecté";
  if (status === "slow") return "Synchronisation lente";
  return "Hors ligne";
}

function syncColor(status: SyncPresenceStatus): string {
  if (status === "connected") return M.SYNC_OK;
  if (status === "slow") return M.SYNC_SLOW;
  return M.SYNC_OFF;
}

export function ChannelConversationHeader({
  thread,
  syncStatus,
  memberCount,
  etaMinutes,
  onOpenColleagues,
  onOpenDetails,
  onPressHeader,
  headerManageable,
  emergency,
}: Props) {
  const router = useRouter();
  const lines = getThreadDisplayLines(thread);
  const isTeam = thread.thread_id === "team";
  const isMission = Boolean(thread.booking_id);

  const statusSuffix = (() => {
    if (isTeam && memberCount != null) {
      return ` · ${memberCount} membre${memberCount > 1 ? "s" : ""}`;
    }
    if (isMission && etaMinutes != null) {
      return ` · ETA ${etaMinutes} min`;
    }
    return "";
  })();

  return (
    <View style={styles.wrap}>
      <View style={styles.bar}>
        <Pressable
          style={styles.backBtn}
          onPress={() => router.back()}
          accessibilityLabel="Retour"
          accessibilityRole="button"
        >
          <Ionicons name="chevron-back" size={26} color={M.BRAND} />
        </Pressable>

        <Pressable
          style={styles.identity}
          onPress={onPressHeader}
          disabled={!onPressHeader}
          accessibilityRole={onPressHeader ? "button" : undefined}
          accessibilityLabel={onPressHeader ? "Gérer le canal" : undefined}
        >
          <InboxThreadAvatar lines={lines} titleFallback={lines.headline} />
          <View style={styles.titles}>
            <View style={styles.titleRow}>
              <AppText variant="body" style={styles.title} numberOfLines={1}>
                {lines.headline}
              </AppText>
              {headerManageable ? (
                <Ionicons name="chevron-forward" size={16} color={M.TEXT_MUTED} />
              ) : null}
            </View>
            <View style={styles.statusRow}>
              <View style={[styles.statusDot, { backgroundColor: syncColor(syncStatus) }]} />
              <AppText variant="caption" style={styles.statusText}>
                {syncLabel(syncStatus)}
                {statusSuffix}
              </AppText>
            </View>
          </View>
        </Pressable>

        {isTeam ? (
          <>
            <Pressable
              style={styles.iconBtn}
              onPress={onOpenColleagues}
              accessibilityLabel="Voir les collègues"
            >
              <Ionicons name="call-outline" size={22} color={M.TEXT} />
            </Pressable>
            <Pressable
              style={styles.iconBtn}
              onPress={onOpenColleagues}
              accessibilityLabel="Menu équipe"
            >
              <Ionicons name="ellipsis-vertical" size={22} color={M.TEXT} />
            </Pressable>
          </>
        ) : (
          <View style={styles.trailing}>
            {onOpenDetails ? (
              <Pressable
                style={styles.iconBtn}
                onPress={onOpenDetails}
                accessibilityLabel="Détails"
              >
                <Ionicons name="ellipsis-vertical" size={22} color={M.TEXT} />
              </Pressable>
            ) : null}
            {emergency ? (
              <EmergencyFab
                variant="header"
                pending={emergency.pending}
                onReport={emergency.onReport}
              />
            ) : null}
            {!onOpenDetails && !emergency ? <View style={styles.iconPlaceholder} /> : null}
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: "100%",
    backgroundColor: M.CARD,
  },
  bar: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    paddingHorizontal: 4,
    gap: 4,
  },
  backBtn: { padding: 6, marginLeft: 2 },
  identity: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    minWidth: 0,
  },
  titles: { flex: 1, minWidth: 0, gap: 2 },
  titleRow: { flexDirection: "row", alignItems: "center", gap: 4, minWidth: 0 },
  title: { fontWeight: "700", fontSize: FONT_SIZE.px17, color: M.TEXT, flexShrink: 1 },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 1 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { color: M.TEXT_SEC, fontSize: FONT_SIZE.px12 },
  iconBtn: { padding: 8 },
  trailing: {
    flexDirection: "row",
    alignItems: "center",
    flexShrink: 0,
  },
  iconPlaceholder: { width: 40 },
});
