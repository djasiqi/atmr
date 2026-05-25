import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import { formatInboxTime, getThreadDisplayLines } from "../inboxDisplay";
import { InboxThreadAvatar } from "./InboxThreadAvatar";
import type { MessageHubThread } from "../types";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  thread: MessageHubThread;
  onPress: () => void;
};

/** Ligne inbox type maquette : avatar | texte (titre, sous-titre, aperçu) | heure + statut à droite. */
export function InboxThreadRow({ thread, onPress }: Props) {
  const lines = getThreadDisplayLines(thread);
  const time = formatInboxTime(thread.last_message_at);
  const unread = (thread.unread_count ?? 0) > 0;
  const preview = thread.last_message_preview?.trim();
  const isUrgent = thread.priority === "urgent" || thread.section === "urgent";

  return (
    <Pressable
      style={({ pressed }) => [
        styles.row,
        isUrgent && styles.rowUrgent,
        pressed && styles.rowPressed,
      ]}
      onPress={onPress}
      accessibilityRole="button"
    >
      <InboxThreadAvatar lines={lines} titleFallback={thread.title} />

      <View style={styles.content}>
        <AppText variant="body" style={styles.headline} numberOfLines={1}>
          {lines.headline}
        </AppText>

        {lines.subline ? (
          <AppText variant="body" style={styles.subline} numberOfLines={1}>
            {lines.subline}
          </AppText>
        ) : null}

        {preview ? (
          <AppText variant="caption" style={styles.preview} numberOfLines={1}>
            {thread.last_message_from_self ? (
              <>
                <AppText variant="caption" style={styles.youPrefix}>
                  Vous :{" "}
                </AppText>
                {preview}
              </>
            ) : (
              preview
            )}
          </AppText>
        ) : null}
      </View>

      <View style={styles.trailing}>
        {time ? (
          <AppText variant="caption" style={styles.time}>
            {time}
          </AppText>
        ) : (
          <View style={styles.timePlaceholder} />
        )}
        <View style={styles.statusRow}>
          {lines.showPin ? <Ionicons name="pin" size={15} color={M.TEXT_MUTED} /> : null}
          {unread ? (
            <View style={styles.badge}>
              <AppText variant="caption" style={styles.badgeText}>
                {thread.unread_count > 99 ? "99+" : thread.unread_count}
              </AppText>
            </View>
          ) : preview ? (
            <Ionicons name="checkmark-done" size={18} color={M.BRAND} />
          ) : null}
        </View>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    paddingVertical: 12,
    paddingHorizontal: 16,
    gap: 12,
    backgroundColor: M.CARD,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: M.BORDER,
  },
  rowPressed: { backgroundColor: "#F8FAF9" },
  rowUrgent: {
    borderLeftWidth: 3,
    borderLeftColor: M.BRAND,
    paddingLeft: 13,
  },
  content: {
    flex: 1,
    minWidth: 0,
    gap: 2,
    paddingTop: 2,
  },
  headline: {
    fontSize: FONT_SIZE.px16,
    fontWeight: "700",
    color: M.TEXT,
  },
  subline: {
    fontSize: FONT_SIZE.px14,
    color: "#374151",
  },
  preview: {
    fontSize: FONT_SIZE.px14,
    color: M.TEXT_MUTED,
    marginTop: 2,
  },
  youPrefix: {
    color: M.BRAND,
    fontWeight: "600",
    fontSize: FONT_SIZE.px14,
  },
  trailing: {
    alignItems: "flex-end",
    minWidth: 52,
    gap: 8,
    paddingTop: 2,
  },
  time: {
    fontSize: FONT_SIZE.px13,
    color: M.TEXT_MUTED,
  },
  timePlaceholder: { height: 16 },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "flex-end",
    gap: 6,
    minHeight: 22,
  },
  badge: {
    backgroundColor: M.BRAND,
    borderRadius: 11,
    minWidth: 22,
    height: 22,
    paddingHorizontal: 6,
    alignItems: "center",
    justifyContent: "center",
  },
  badgeText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: FONT_SIZE.px12,
  },
});
