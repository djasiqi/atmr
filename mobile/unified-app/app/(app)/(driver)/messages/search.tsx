import { useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { Screen, AppText, useAppViewport } from "../../../../src/design/responsive";
import { useDriverCompanyId } from "../../../../src/features/driver/messages/hooks";
import { searchMessageHub } from "../../../../src/features/driver/messages/api";
import { formatInboxTime } from "../../../../src/features/driver/messages/inboxDisplay";
import { FONT_SIZE } from "../../../../src/design/responsive/typographyTokens";

export default function DriverMessagesSearchScreen() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Awaited<ReturnType<typeof searchMessageHub>>>([]);
  const [loading, setLoading] = useState(false);
  const companyId = useDriverCompanyId();
  const router = useRouter();
  const { horizontalPadding } = useAppViewport();

  const runSearch = async () => {
    if (!companyId || !query.trim()) return;
    setLoading(true);
    try {
      const rows = await searchMessageHub(companyId, query);
      setResults(rows);
    } finally {
      setLoading(false);
    }
  };

  return (
    <DriverContextGuard>
      <PermissionGuard permission="chat:read">
        <Screen scroll={false} backgroundColor="#fff">
          <View style={[styles.wrap, { paddingHorizontal: horizontalPadding }]}>
            <View style={styles.searchRow}>
              <Ionicons name="search-outline" size={20} color="#9CA3AF" style={styles.searchIcon} />
              <TextInput
                style={styles.input}
                placeholder="Messages, patients, missions…"
                placeholderTextColor="#9CA3AF"
                value={query}
                onChangeText={setQuery}
                onSubmitEditing={() => void runSearch()}
                returnKeyType="search"
                autoFocus
              />
            </View>
            {loading ? <ActivityIndicator color="#0A8F7A" style={styles.loader} /> : null}
            <FlatList
              data={results}
              keyExtractor={(item) => String(item.id)}
              contentContainerStyle={results.length === 0 ? styles.emptyList : undefined}
              renderItem={({ item }) => (
                <Pressable
                  style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}
                  onPress={() =>
                    router.push({
                      pathname: "/(app)/(driver)/messages/[threadId]",
                      params: { threadId: item.thread_id ?? "dispatch" },
                    })
                  }
                >
                  <View style={styles.rowBody}>
                    <AppText variant="body" numberOfLines={2} style={styles.content}>
                      {item.content}
                    </AppText>
                    <AppText variant="caption" style={styles.meta}>
                      {item.sender_name ?? "—"} · {formatInboxTime(item.timestamp)}
                    </AppText>
                  </View>
                  <Ionicons name="chevron-forward" size={18} color="#9CA3AF" />
                </Pressable>
              )}
              ListEmptyComponent={
                query.trim() ? (
                  <View style={styles.emptyWrap}>
                    <Ionicons name="document-text-outline" size={36} color="#94A3B8" />
                    <AppText variant="bodyMuted" style={styles.empty}>
                      Aucun résultat pour « {query.trim()} »
                    </AppText>
                  </View>
                ) : (
                  <AppText variant="bodyMuted" style={styles.hint}>
                    Saisissez au moins un mot puis validez la recherche.
                  </AppText>
                )
              }
            />
          </View>
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, gap: 12, paddingTop: 8 },
  searchRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#F5F7F6",
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#E5E7EB",
    paddingHorizontal: 12,
  },
  searchIcon: { marginRight: 4 },
  input: {
    flex: 1,
    paddingVertical: 12,
    fontSize: FONT_SIZE.px15,
    color: "#111827",
  },
  loader: { marginVertical: 8 },
  row: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 14,
    gap: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#E5E7EB",
  },
  rowPressed: { backgroundColor: "#F5F7F6" },
  rowBody: { flex: 1, gap: 4 },
  content: { color: "#111827" },
  meta: { color: "#9CA3AF" },
  emptyList: { flexGrow: 1 },
  emptyWrap: { alignItems: "center", gap: 12, paddingTop: 48 },
  empty: { textAlign: "center" },
  hint: { textAlign: "center", marginTop: 24, paddingHorizontal: 16 },
});
