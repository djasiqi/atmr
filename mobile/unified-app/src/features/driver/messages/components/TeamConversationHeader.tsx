import { Pressable, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { SyncPresenceStatus } from "../types";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  /** `null` = chargement (évite d’afficher « 0 membre » à tort). */
  memberCount: number | null;
  syncStatus: SyncPresenceStatus;
  onOpenColleagues?: () => void;
};

export function TeamConversationHeader({ memberCount, syncStatus, onOpenColleagues }: Props) {
  const router = useRouter();
  const syncColor =
    syncStatus === "connected" ? "#22C55E" : syncStatus === "slow" ? "#F59E0B" : "#94A3B8";
  const syncLabel =
    syncStatus === "connected"
      ? "Connecté"
      : syncStatus === "slow"
        ? "Synchronisation lente"
        : "Hors ligne";

  return (
    <View style={styles.wrap}>
      <View style={styles.bar}>
        <Pressable
          style={styles.backBtn}
          onPress={() => router.back()}
          accessibilityLabel="Retour"
          accessibilityRole="button"
        >
          <Ionicons name="chevron-back" size={26} color="#0A8F7A" />
        </Pressable>

        <View style={styles.avatar}>
          <Ionicons name="people" size={26} color="#fff" />
        </View>

        <View style={styles.titles}>
          <AppText variant="body" style={styles.title} numberOfLines={1}>
            Équipe chauffeurs
          </AppText>
          <View style={styles.statusRow}>
            <View style={[styles.statusDot, { backgroundColor: syncColor }]} />
            <AppText variant="caption" style={styles.statusText}>
              {syncLabel}
              {memberCount == null
                ? ""
                : ` · ${memberCount} membre${memberCount > 1 ? "s" : ""}`}
            </AppText>
          </View>
        </View>

        <Pressable
          style={styles.iconBtn}
          onPress={onOpenColleagues}
          accessibilityLabel="Voir les collègues"
        >
          <Ionicons name="call-outline" size={22} color="#111827" />
        </Pressable>
        <Pressable
          style={styles.iconBtn}
          onPress={onOpenColleagues}
          accessibilityLabel="Menu équipe"
        >
          <Ionicons name="ellipsis-vertical" size={22} color="#111827" />
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: "100%",
    backgroundColor: "#FFFFFF",
  },
  bar: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    paddingHorizontal: 4,
    gap: 8,
  },
  backBtn: { padding: 6, marginLeft: 2 },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "#64748B",
    alignItems: "center",
    justifyContent: "center",
  },
  titles: { flex: 1, minWidth: 0, gap: 2 },
  title: { fontWeight: "700", fontSize: FONT_SIZE.px17, color: "#111827" },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 1 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { color: "#6B7280", fontSize: FONT_SIZE.px12 },
  iconBtn: { padding: 8 },
});
