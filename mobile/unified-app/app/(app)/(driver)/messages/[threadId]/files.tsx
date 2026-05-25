import { useMemo, useState } from "react";
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
  Linking,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../../../src/core/guards";
import { Screen, AppText, useAppViewport } from "../../../../../src/design/responsive";
import { D } from "../../../../../src/features/driver/theme/driverDashboardTheme";
import {
  useDriverCompanyId,
  useMissionEtaMinutes,
  useMessageHubThreads,
  useThreadMessages,
} from "../../../../../src/features/driver/messages/hooks";
import { resolveMediaUrl } from "../../../../../src/core/api/mediaUrl";

type TabKey = "all" | "photos" | "documents";

export default function MessageThreadFilesScreen() {
  const { threadId: threadIdParam } = useLocalSearchParams<{ threadId: string }>();
  const threadId = typeof threadIdParam === "string" ? threadIdParam : "dispatch";
  const { horizontalPadding } = useAppViewport();
  const companyId = useDriverCompanyId();
  const threadsQuery = useMessageHubThreads(companyId);
  const thread = threadsQuery.data?.threads.find((t) => t.thread_id === threadId);
  const messagesQuery = useThreadMessages(companyId, threadId, thread?.conversation_id);
  const [tab, setTab] = useState<TabKey>("all");

  const etaQuery = useMissionEtaMinutes(thread?.booking_id ?? null, thread?.status);

  const media = useMemo(() => {
    const items = (messagesQuery.data ?? []).flatMap((m) => {
      const rows: { id: string; kind: "photo" | "document"; url: string; label: string; at: string }[] = [];
      if (m.image_url) {
        rows.push({
          id: `${m.id}-img`,
          kind: "photo",
          url: m.image_url,
          label: "Photo",
          at: m.timestamp,
        });
      }
      if (m.pdf_url) {
        rows.push({
          id: `${m.id}-pdf`,
          kind: "document",
          url: m.pdf_url,
          label: m.pdf_filename ?? "Document PDF",
          at: m.timestamp,
        });
      }
      return rows;
    });
    if (tab === "photos") return items.filter((i) => i.kind === "photo");
    if (tab === "documents") return items.filter((i) => i.kind === "document");
    return items;
  }, [messagesQuery.data, tab]);

  return (
    <DriverContextGuard>
      <PermissionGuard permission="chat:read">
        <Screen scroll backgroundColor={D.pageBg}>
          <ScrollView contentContainerStyle={{ paddingHorizontal: horizontalPadding, paddingBottom: 32, gap: 12 }}>
            <View style={styles.mapCard}>
              <AppText variant="sectionTitle">Position & ETA</AppText>
              <AppText variant="bodyMuted">
                Mission · {thread?.title ?? threadId}
              </AppText>
              {etaQuery.data?.driver_lat != null && etaQuery.data?.driver_lon != null ? (
                <AppText variant="caption">
                  GPS · {etaQuery.data.driver_lat?.toFixed(5)}, {etaQuery.data.driver_lon?.toFixed(5)}
                </AppText>
              ) : (
                <AppText variant="caption">Position en cours de synchronisation…</AppText>
              )}
              <AppText variant="label" style={styles.eta}>
                ETA · {etaQuery.data?.eta_minutes != null ? `${etaQuery.data.eta_minutes} min` : "—"}
              </AppText>
              <Pressable
                style={styles.shareBtn}
                onPress={() => {
                  const lat = etaQuery.data?.driver_lat;
                  const lon = etaQuery.data?.driver_lon;
                  if (lat == null || lon == null) return;
                  void Linking.openURL(`https://maps.google.com/?q=${lat},${lon}`);
                }}
              >
                <AppText variant="label" style={styles.shareBtnText}>
                  Ouvrir dans Maps
                </AppText>
              </Pressable>
            </View>

            <View style={styles.tabs}>
              {(["all", "photos", "documents"] as TabKey[]).map((key) => (
                <Pressable
                  key={key}
                  style={[styles.tab, tab === key && styles.tabActive]}
                  onPress={() => setTab(key)}
                >
                  <AppText variant="caption" style={tab === key ? styles.tabTextActive : styles.tabText}>
                    {key === "all" ? "Tous" : key === "photos" ? "Photos" : "Documents"}
                  </AppText>
                </Pressable>
              ))}
            </View>

            <View style={styles.grid}>
              {media.length === 0 ? (
                <AppText variant="bodyMuted">Aucune pièce jointe dans cette conversation.</AppText>
              ) : (
                media.map((item) => (
                  <Pressable
                    key={item.id}
                    style={styles.tile}
                    onPress={() => void Linking.openURL(resolveMediaUrl(item.url) ?? item.url)}
                  >
                    {item.kind === "photo" ? (
                      <Image
                        source={{ uri: resolveMediaUrl(item.url) ?? item.url }}
                        style={styles.thumb}
                      />
                    ) : (
                      <View style={styles.docIcon}>
                        <AppText variant="caption">PDF</AppText>
                      </View>
                    )}
                    <AppText variant="caption" numberOfLines={2} style={styles.tileLabel}>
                      {item.label}
                    </AppText>
                  </Pressable>
                ))
              )}
            </View>
          </ScrollView>
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  mapCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 14,
    gap: 6,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#e5e7eb",
  },
  eta: { color: "#047857", marginTop: 4 },
  shareBtn: {
    marginTop: 8,
    backgroundColor: "#0A8F7A",
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
  },
  shareBtnText: { color: "#fff" },
  tabs: { flexDirection: "row", gap: 8 },
  tab: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "#f3f4f6",
  },
  tabActive: { backgroundColor: "#d1fae5" },
  tabText: { color: "#6b7280" },
  tabTextActive: { color: "#065f46", fontWeight: "700" },
  grid: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  tile: { width: "47%", backgroundColor: "#fff", borderRadius: 10, padding: 8 },
  thumb: { width: "100%", aspectRatio: 1, borderRadius: 8, backgroundColor: "#e5e7eb" },
  docIcon: {
    aspectRatio: 1,
    borderRadius: 8,
    backgroundColor: "#fef3c7",
    alignItems: "center",
    justifyContent: "center",
  },
  tileLabel: { marginTop: 4, color: "#374151" },
});
