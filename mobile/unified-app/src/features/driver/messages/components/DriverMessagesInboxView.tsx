import { useCallback, useEffect, useMemo, useState } from "react";
import { endPageLoad, startPageLoad } from "../../../../core/observability/perfKpi";
import {
  FlatList,
  RefreshControl,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { Screen, AppText, useAppViewport } from "../../../../design/responsive";
import { useDriverFloatingTabScrollPadding } from "../../navigation/DriverFloatingTabBar";
import {
  useDriverCompanyId,
  useEnsureDriverSocketForHub,
  useHubMessageRealtimeSync,
  useMarkAllInboxRead,
  useMessageHubThreads,
} from "../hooks";
import { useInboxSyncBanner } from "../useInboxSyncBanner";
import {
  type InboxTab,
  countUnreadForTab,
  threadMatchesTab,
} from "../inboxDisplay";
import {
  filterThreadsForMobileTab,
  sortThreadsForMobileInbox,
} from "../inboxThreadPolicy";
import type { MessageHubThread } from "../types";
import { inboxThreadListKey } from "../dedupeHubThreads";
import { InboxThreadRow } from "./InboxThreadRow";
import { InboxThreadListSkeleton } from "./InboxThreadRowSkeleton";
import { DataReveal } from "../../../../design/navigation/presets/dataReveal";
import { MessagesInboxEmpty } from "./MessagesInboxEmpty";
import { MessagesInboxHeader } from "./MessagesInboxHeader";
import { MessagesInboxMenu } from "../../../messaging/components/MessagesInboxMenu";
import { M } from "../../../messaging/messagingTheme";
import { messagesInboxStyles as styles } from "../../../messaging/messagesInboxStyles";
import { MessagesInboxTabs } from "./MessagesInboxTabs";
import { ContactsInboxShortcuts } from "./ContactsInboxShortcuts";

const EMPTY_THREADS: MessageHubThread[] = [];

/**
 * Inbox chauffeur — liste plate type messagerie (maquette TOUTES / MISSIONS / CONTACTS).
 * Remplace l’ancien rendu par sections (MISSION ACTIVE, ENTREPRISE, cartes).
 */
export function DriverMessagesInboxView() {
  const router = useRouter();
  const { horizontalPadding, topInset } = useAppViewport();
  const scrollPad = useDriverFloatingTabScrollPadding();
  const companyId = useDriverCompanyId();
  const threadsQuery = useMessageHubThreads(companyId);
  useEffect(() => {
    startPageLoad("driver.inbox");
  }, []);
  useEffect(() => {
    if (!threadsQuery.isSuccess || threadsQuery.isFetching) return;
    endPageLoad("driver.inbox", "driver.messages.inbox.data_ready");
  }, [threadsQuery.isFetching, threadsQuery.isSuccess]);
  const showSyncBanner = useInboxSyncBanner(threadsQuery);
  const markAllRead = useMarkAllInboxRead(companyId);
  const [search, setSearch] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<InboxTab>("contacts");
  const [menuOpen, setMenuOpen] = useState(false);
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  useEnsureDriverSocketForHub();
  useHubMessageRealtimeSync(companyId);

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
    list = filterThreadsForMobileTab(list, activeTab);
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
    return sortThreadsForMobileInbox(list);
  }, [activeTab, allThreads, search, unreadOnly]);

  const unreadThreadIds = useMemo(
    () => allThreads.filter((t) => (t.unread_count ?? 0) > 0).map((t) => t.thread_id),
    [allThreads]
  );

  const openThread = useCallback(
    (threadId: string) => {
      router.push({
        pathname: "/(app)/(driver)/messages/[threadId]",
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

  const isInitialLoading =
    threadsQuery.isLoading && !threadsQuery.data && allThreads.length === 0;

  const listHeader = useMemo(() => {
    if (activeTab === "contacts" || activeTab === "all") {
      if (search.trim()) return null;
      return (
        <ContactsInboxShortcuts
          onOpenTeam={() => openThread("team")}
          onOpenDispatch={() => openThread("dispatch")}
          onNewColleague={() => router.push("/(app)/(driver)/messages/colleagues")}
          onOpenSupport={() => openThread("support")}
        />
      );
    }
    return null;
  }, [activeTab, openThread, router, search]);

  const menuItems = useMemo(() => {
    const items = [];
    if (unreadThreadIds.length > 0) {
      items.push({
        icon: "checkmark-done-outline" as const,
        label: "Tout marquer comme lu",
        onPress: () => void markAllRead.mutateAsync(unreadThreadIds),
      });
    }
    items.push(
      {
        icon: "settings-outline" as const,
        label: "Paramètres messagerie",
        onPress: () => router.push("/(app)/(driver)/messages/settings"),
      },
      {
        icon: "person-add-outline" as const,
        label: "Nouveau contact équipe",
        onPress: () => router.push("/(app)/(driver)/messages/colleagues"),
      },
      {
        icon: "people-outline" as const,
        label: "Canal équipe (groupe)",
        onPress: () => openThread("team"),
      },
      {
        icon: "headset-outline" as const,
        label: "Support LIRIE",
        onPress: () => openThread("support"),
      },
      {
        icon: "search-outline" as const,
        label: "Recherche avancée",
        onPress: () => router.push("/(app)/(driver)/messages/search"),
      }
    );
    return items;
  }, [markAllRead, openThread, router, unreadThreadIds]);

  return (
    <Screen scroll={false} backgroundColor={M.CARD} withHorizontalPadding={false} safeTop pageTransition={false}>
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
            placeholder="Rechercher une mission, un contact…"
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
        <DataReveal visible={!isInitialLoading} screen="driver.inbox">
          <FlatList
            data={filteredThreads}
            keyExtractor={inboxThreadListKey}
            renderItem={renderItem}
            ListHeaderComponent={listHeader}
            ListEmptyComponent={
              <MessagesInboxEmpty
                tab={activeTab}
                hasSearch={Boolean(search.trim())}
                urgentFilter={unreadOnly}
                onOpenTeam={() => openThread("team")}
                onOpenDispatch={() => openThread("dispatch")}
                onOpenColleagues={() => router.push("/(app)/(driver)/messages/colleagues")}
                onOpenSupport={() => openThread("support")}
              />
            }
            contentContainerStyle={
              filteredThreads.length === 0
                ? { flexGrow: 1, paddingBottom: scrollPad }
                : { paddingBottom: scrollPad }
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
