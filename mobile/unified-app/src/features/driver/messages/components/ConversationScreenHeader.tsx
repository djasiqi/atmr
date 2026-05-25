import { Pressable, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { getThreadDisplayLines } from "../inboxDisplay";
import { InboxThreadAvatar } from "./InboxThreadAvatar";
import type { MessageHubThread } from "../types";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  thread: MessageHubThread;
  onOpenDetails?: () => void;
};

export function ConversationScreenHeader({ thread, onOpenDetails }: Props) {
  const router = useRouter();
  const lines = getThreadDisplayLines(thread);

  return (
    <View style={styles.bar}>
      <Pressable
        style={styles.backBtn}
        onPress={() => router.back()}
        accessibilityLabel="Retour"
        accessibilityRole="button"
      >
        <Ionicons name="chevron-back" size={26} color="#0A8F7A" />
      </Pressable>

      <Pressable
        style={styles.center}
        onPress={onOpenDetails}
        disabled={!onOpenDetails}
        accessibilityRole={onOpenDetails ? "button" : "text"}
      >
        <InboxThreadAvatar lines={lines} titleFallback={thread.title} />
        <View style={styles.titles}>
          <AppText variant="body" style={styles.headline} numberOfLines={1}>
            {lines.headline}
          </AppText>
          {lines.subline ? (
            <AppText variant="caption" style={styles.subline} numberOfLines={1}>
              {lines.subline}
            </AppText>
          ) : null}
        </View>
      </Pressable>

      {onOpenDetails ? (
        <Pressable
          style={styles.actionBtn}
          onPress={onOpenDetails}
          accessibilityLabel="Détails"
        >
          <Ionicons name="information-circle-outline" size={24} color="#111827" />
        </Pressable>
      ) : (
        <View style={styles.actionPlaceholder} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    paddingHorizontal: 4,
    backgroundColor: "#fff",
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#E5E7EB",
    gap: 4,
  },
  backBtn: {
    padding: 6,
    marginLeft: 4,
  },
  center: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    minWidth: 0,
  },
  titles: { flex: 1, minWidth: 0 },
  headline: { fontWeight: "700", fontSize: FONT_SIZE.px16, color: "#111827" },
  subline: { color: "#6B7280", marginTop: 1 },
  actionBtn: { padding: 8 },
  actionPlaceholder: { width: 40 },
});
