import { useCallback, useMemo, useState } from "react";
import {
  FlatList,
  RefreshControl,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { Screen, AppText, useAppViewport } from "../../../../design/responsive";
import { computeCompanyFloatingBottomPad } from "../../../../design/navigation/BaseFloatingBar";
import {
  useCompanyHubRealtimeSync,
  useCompanyMessageHubThreads,
  useMarkAllCompanyInboxRead,
} from "../hooks";
import { useCompanyNumericId } from "../companyId";
import { useInboxSyncBanner } from "../../../driver/messages/useInboxSyncBanner";
import {
  type InboxTab,
  countUnreadForTab,
  sortThreadsByRecent,
  threadMatchesTab,
} from "../../../driver/messages/inboxDisplay";
import type { MessageHubThread } from "../../../driver/messages/types";
import { inboxThreadListKey } from "../../../driver/messages/dedupeHubThreads";
import { InboxThreadRow } from "../../../driver/messages/components/InboxThreadRow";
import { InboxThreadListSkeleton } from "../../../driver/messages/components/InboxThreadRowSkeleton";
import { DataReveal } from "../../../../design/navigation/presets/dataReveal";
import { MessagesInboxEmpty } from "../../../driver/messages/components/MessagesInboxEmpty";
import { MessagesInboxHeader } from "../../../driver/messages/components/MessagesInboxHeader";
import { MessagesInboxMenu } from "../../../messaging/components/MessagesInboxMenu";
import { M } from "../../../messaging/messagingTheme";
import { messagesInboxStyles as styles } from "../../../messaging/messagesInboxStyles";
import { MessagesInboxTabs } from "../../../driver/messages/components/MessagesInboxTabs";
import { usePerfScreenReady } from "../../../../core/observability/usePerfScreenReady";

const EMPTY_THREADS: MessageHubThread[] = [];

export function CompanyMessagesInboxView() {
  const router = useRouter();
  const { horizontalPadding, topInset, bottomInset } = useAppViewport();
  const companyId = useCompanyNumericId();
  const threadsQuery = useCompanyMessageHubThreads(companyId);
  const showSyncBanner = useInboxSyncBanner(threadsQuery);
  const markAllRead = useMarkAllCompanyInboxRead(companyId);
  const [search, setSearch] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<InboxTab>("all");
  const [menuOpen, setMenuOpen] = useState(false);
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  useCompanyHubRealtimeSync(companyId);
  usePerfScreenReady(
    "company.inbox",
    "company.inbox.data_ready",
    threadsQuery.isSuccess || threadsQuery.isError
  );

  const listBottomPad = 64 + computeCompanyFloatingBottomPad(bottomInset);
  const allThreads = useMemo(
    () => threadsQuery.data?.threads ?? EMPTY_THREADS,
    [threadsQuery.data?.threads]
  );

  const unreadByTab = useMemo(
    () => ({
      all: countUnreadForTab(allThreads, "all"),
      missions: countUnreadForTab(allThreads, "missions"),
      contacts: countUnreadForTab(allThreads, "contacts"),
    }),
    [allThreads]
  );

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await threadsQuery.refetch();
    } finally {
      setRefreshing(false);
    }
  }, [threadsQuery]);

  const filteredThreads = useMemo(() => {
    const q = search.trim().toLowerCase();
    let list = allThreads.filter((t) => threadMatchesTab(t, activeTab));
    if (unreadOnly) {
      list = list.filter((t) => (t.unread_count ?? 0) > 0);
    }
    if (q) {
      list = list.filter(
        (t) =>
          t.title.toLowerCase().includes(q) ||
          (t.last_message_preview ?? "").toLowerCase().includes(q) ||
          (t.subtitle ?? "").toLowerCase().includes(q) ||
          String(t.booking_id ?? "").includes(q)
      );
    }
    return sortThreadsByRecent(list);
  }, [activeTab, allThreads, search, unreadOnly]);

  const unreadThreadIds = useMemo(
    () => allThreads.filter((t) => (t.unread_count ?? 0) > 0).map((t) => t.thread_id),
    [allThreads]
  );

  const openThread = useCallback(
    (threadId: string) => {
      router.push({
        pathname: "/(app)/(company)/messages/[threadId]",
        params: { threadId },
      });
    },
    [router]
  );

  const renderItem = useCallback(
    ({ item }: { item: MessageHubThread }) => (
      <InboxThreadRow thread={item} onPress={() => openThread(item.thread_id)} />
    ),
    [openThread]
  );

  const isInitialLoading = !threadsQuery.isFetched && allThreads.length === 0;

  const menuItems = useMemo(() => {
    const items = [];
    if (unreadThreadIds.length > 0) {
      items.push({
        icon: "checkmark-done-outline" as const,
        label: "Tout marquer comme lu",
        onPress: () => void markAllRead.mutateAsync(unreadThreadIds),
      });
    }
    items.push({
      icon: "business-outline" as const,
      label: "Canal dispatch",
      onPress: () => openThread("dispatch"),
    });
    return items;
  }, [markAllRead, openThread, unreadThreadIds]);

  return (
    <Screen
      scroll={false}
      backgroundColor={M.CARD}
      withHorizontalPadding={false}
      safeTop
      pageTransition={false}
    >
      <View
        style={[
          styles.headerBlock,
          { paddingHorizontal: horizontalPadding, paddingTop: topInset > 0 ? 2 : 8 },
        ]}
      >
        <MessagesInboxHeader
          urgentFilterActive={unreadOnly}
          searchOpen={searchOpen}
          onToggleFilter={() => setUnreadOnly((v) => !v)}
          onToggleSearch={() => {
            setSearchOpen((v) => !v);
            if (searchOpen) setSearch("");
          }}
          onOpenMenu={() => setMenuOpen(true)}
        />

        {searchOpen ? (
          <TextInput
            style={styles.search}
            placeholder="Rechercher une mission, un chauffeur…"
            placeholderTextColor={M.TEXT_MUTED}
            value={search}
            onChangeText={setSearch}
            autoFocus
            clearButtonMode="while-editing"
          />
        ) : null}

        <View style={styles.syncBannerSlot}>
          {showSyncBanner ? (
            <View style={styles.offlineChip}>
              <Ionicons name="cloud-offline-outline" size={14} color={M.OFFLINE_ICON} />
              <AppText variant="caption" style={styles.offlineText}>
                Données locales — sync partielle
              </AppText>
            </View>
          ) : null}
        </View>
      </View>

      <MessagesInboxTabs active={activeTab} onChange={setActiveTab} unreadByTab={unreadByTab} />

      {isInitialLoading ? (
        <InboxThreadListSkeleton />
      ) : (
        <DataReveal visible={!isInitialLoading} screen="company.inbox">
          <FlatList
            data={filteredThreads}
            keyExtractor={inboxThreadListKey}
            renderItem={renderItem}
            ListEmptyComponent={
              <MessagesInboxEmpty
                tab={activeTab}
                hasSearch={Boolean(search.trim())}
                urgentFilter={unreadOnly}
                onOpenDispatch={() => openThread("dispatch")}
              />
            }
            contentContainerStyle={
              filteredThreads.length === 0
                ? { flexGrow: 1, paddingBottom: listBottomPad }
                : { paddingBottom: listBottomPad }
            }
            style={styles.list}
            refreshControl={
              <RefreshControl
                refreshing={refreshing}
                onRefresh={() => void onRefresh()}
                tintColor={M.BRAND}
              />
            }
            keyboardShouldPersistTaps="handled"
            initialNumToRender={12}
            windowSize={8}
          />
        </DataReveal>
      )}

      <MessagesInboxMenu
        visible={menuOpen}
        horizontalPadding={horizontalPadding}
        items={menuItems}
        onClose={() => setMenuOpen(false)}
      />
    </Screen>
  );
}
