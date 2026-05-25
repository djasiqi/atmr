import { useCallback, useEffect, useMemo, useRef, memo } from "react";
import {
  FlatList,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
  Pressable,
  StyleSheet,
  View,
  type FlatList as FlatListType,
  type ListRenderItem,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { useKeyboardLayout } from "../../../chat/useKeyboardLayout";
import { M } from "../../../messaging/messagingTheme";
import { MessageBubble } from "../../../chat/components/MessageBubble";
import { GroupMessageBubble } from "./GroupMessageBubble";
import { buildGroupMessageMeta } from "../groupMessageLayout";
import { TEAM_CHAT_BACKGROUND } from "../teamChatTheme";
import { SystemMessageRow } from "./SystemMessageRow";
import {
  ChatDateSeparator,
  dayKeyFromIso,
  formatChatDayLabel,
} from "./ChatDateSeparator";
import type { HubChatMessage } from "../types";
import type { SharedChatMessage } from "../../../chat/types";
import type { HubFeedInitialAnchorMode, HubFeedScrollItem } from "../hubFeedInitialScroll";
import { useHubFeedInitialScroll } from "../useHubFeedInitialScroll";

/** Marge entre dernier message et quick actions. */
const LIST_BOTTOM_GAP = 20;

type FeedItem =
  | { kind: "date"; id: string; label: string }
  | { kind: "message"; id: string; message: HubChatMessage };

type Props = {
  messages: HubChatMessage[];
  ownSenderId?: string | null;
  ownDisplayName?: string;
  compact?: boolean;
  variant?: "default" | "team";
  onOpenImage?: (url: string) => void;
  onOpenPdf?: (url: string) => void;
  onLoadMore?: () => void;
  loadingMore?: boolean;
  /** Incrémenté au focus écran pour réancrer la position initiale. */
  listAnchorKey?: number;
  /** Mode d’ancrage à l’ouverture: premier non lu ou dernier message. */
  initialAnchorMode?: HubFeedInitialAnchorMode;
  loading?: boolean;
  /** Appelé une fois la position initiale appliquée (ex. marquer le fil lu). */
  onInitialScrollSettled?: () => void;
  /** Appelé au premier rendu de la liste visible (first paint). */
  onFirstListRender?: () => void;
};

function toShared(
  message: HubChatMessage,
  ownDisplayName: string,
  ownSenderId: string | null
): SharedChatMessage {
  const isOwn =
    ownSenderId != null &&
    message.sender_id != null &&
    String(message.sender_id) === ownSenderId;
  return {
    id: message.id,
    content: message.content,
    type:
      message.message_type === "system"
        ? "system"
        : message.audio_url
          ? "audio"
          : message.image_url
            ? "image"
            : message.pdf_url
              ? "pdf"
              : "text",
    senderId: message.sender_id ?? null,
    senderRole: isOwn ? "DRIVER" : message.sender_role,
    senderName: isOwn ? ownDisplayName : message.sender_name,
    timestamp: message.timestamp,
    imageUrl: message.image_url,
    pdfUrl: message.pdf_url,
    pdfFilename: message.pdf_filename,
    audioUrl: message.audio_url,
    threadId: message.thread_id,
  };
}

function buildFeedItems(messages: HubChatMessage[], withDates: boolean): FeedItem[] {
  if (!withDates) {
    return messages.map((m) => ({
      kind: "message" as const,
      id: String(m.id),
      message: m,
    }));
  }
  const items: FeedItem[] = [];
  let lastDay = "";
  for (const m of messages) {
    const day = dayKeyFromIso(m.timestamp);
    if (day && day !== lastDay) {
      lastDay = day;
      items.push({
        kind: "date",
        id: `date-${day}`,
        label: formatChatDayLabel(m.timestamp),
      });
    }
    items.push({ kind: "message", id: String(m.id), message: m });
  }
  return items;
}

function toScrollItems(items: FeedItem[]): HubFeedScrollItem[] {
  return items.map((item) =>
    item.kind === "date"
      ? { kind: "date", id: item.id }
      : { kind: "message", id: item.id, message: item.message }
  );
}

type FeedRowProps = {
  item: FeedItem;
  isTeam: boolean;
  ownDisplayName: string;
  ownSenderId: string | null;
  groupMeta: ReturnType<typeof buildGroupMessageMeta> | null;
  onOpenImage?: (url: string) => void;
  onOpenPdf?: (url: string) => void;
  onMediaLayout: () => void;
  compact: boolean;
};

function areFeedItemsEquivalent(a: FeedItem, b: FeedItem): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "date" && b.kind === "date") {
    return a.id === b.id && a.label === b.label;
  }
  if (a.kind === "message" && b.kind === "message") {
    const am = a.message;
    const bm = b.message;
    return (
      String(am.id) === String(bm.id) &&
      am.content === bm.content &&
      am.timestamp === bm.timestamp &&
      am.acked_at === bm.acked_at &&
      am.is_read === bm.is_read &&
      am.message_type === bm.message_type &&
      am.image_url === bm.image_url &&
      am.pdf_url === bm.pdf_url &&
      am.audio_url === bm.audio_url &&
      am._localId === bm._localId
    );
  }
  return false;
}

function sameGroupMetaForItem(
  prevMeta: ReturnType<typeof buildGroupMessageMeta> | null,
  nextMeta: ReturnType<typeof buildGroupMessageMeta> | null,
  item: FeedItem
): boolean {
  if (item.kind !== "message") return true;
  const itemId = String(item.message.id);
  const prev = prevMeta?.get(itemId);
  const next = nextMeta?.get(itemId);
  return (
    prev?.showAvatar === next?.showAvatar &&
    prev?.showSenderName === next?.showSenderName &&
    prev?.isFirstInGroup === next?.isFirstInGroup &&
    prev?.isLastInGroup === next?.isLastInGroup
  );
}

const FeedRow = memo(function FeedRow({
  item,
  isTeam,
  ownDisplayName,
  ownSenderId,
  groupMeta,
  onOpenImage,
  onOpenPdf,
  onMediaLayout,
  compact,
}: FeedRowProps) {
  if (item.kind === "date") {
    return <ChatDateSeparator label={item.label} density={isTeam ? "compact" : "default"} />;
  }
  const msg = item.message;
  if (msg.message_type === "system") {
    return (
      <SystemMessageRow
        content={msg.content}
        timestamp={msg.timestamp}
        senderName={msg.sender_name}
        variant={isTeam ? "team" : "default"}
      />
    );
  }
  const shared = toShared(msg, ownDisplayName, ownSenderId);
  const isOwn =
    ownSenderId != null &&
    msg.sender_id != null &&
    String(msg.sender_id) === ownSenderId;

  if (isTeam && groupMeta) {
    const meta = groupMeta.get(String(msg.id)) ?? {
      showAvatar: !isOwn,
      showSenderName: !isOwn,
      isFirstInGroup: true,
      isLastInGroup: true,
    };
    return (
      <View>
        <GroupMessageBubble
          message={shared}
          ownSenderId={ownSenderId}
          ownDisplayName={ownDisplayName}
          group={meta}
          onOpenImage={onOpenImage}
          onOpenPdf={onOpenPdf}
          onMediaLayout={onMediaLayout}
        />
        {isOwn ? (
          <View style={styles.readReceipt}>
            <Ionicons name="checkmark-done" size={16} color={M.BRAND} />
          </View>
        ) : null}
      </View>
    );
  }

  return (
    <View style={compact ? styles.compactBubbleLegacy : undefined}>
      <MessageBubble
        message={shared}
        ownSenderId={ownSenderId}
        ownSenderRoles={["DRIVER", "driver", "COMPANY", "company"]}
        density="default"
        onOpenImage={onOpenImage}
        onOpenPdf={onOpenPdf}
        onMediaLayout={onMediaLayout}
      />
    </View>
  );
}, (prev, next) => {
  return (
    areFeedItemsEquivalent(prev.item, next.item) &&
    prev.isTeam === next.isTeam &&
    prev.ownDisplayName === next.ownDisplayName &&
    prev.ownSenderId === next.ownSenderId &&
    prev.compact === next.compact &&
    prev.onOpenImage === next.onOpenImage &&
    prev.onOpenPdf === next.onOpenPdf &&
    prev.onMediaLayout === next.onMediaLayout &&
    sameGroupMetaForItem(prev.groupMeta, next.groupMeta, next.item)
  );
});

export function HubMessageFeed({
  messages,
  ownSenderId = null,
  ownDisplayName = "Moi",
  compact = true,
  variant = "default",
  onOpenImage,
  onOpenPdf,
  onLoadMore,
  loadingMore = false,
  listAnchorKey = 0,
  initialAnchorMode = "first_unread",
  loading = false,
  onInitialScrollSettled,
  onFirstListRender,
}: Props) {
  const isTeam = variant === "team";
  const data = useMemo(() => buildFeedItems(messages, isTeam), [messages, isTeam]);
  const scrollItems = useMemo(() => toScrollItems(data), [data]);
  const groupMeta = useMemo(
    () => (isTeam ? buildGroupMessageMeta(messages, ownSenderId) : null),
    [isTeam, messages, ownSenderId]
  );

  const listRef = useRef<FlatListType<FeedItem> | null>(null);
  const {
    initialScrollIndex,
    initialNumToRender,
    onContentSizeChange,
    onScrollToIndexFailed,
    scheduleMediaRelayout,
  } = useHubFeedInitialScroll({
    listRef,
    feedItems: scrollItems,
    messages,
    ownSenderId,
    listAnchorKey,
    initialAnchorMode,
    loading,
    loadingMore,
    onInitialScrollSettled,
  });

  const onMediaLayout = useCallback(() => {
    scheduleMediaRelayout();
  }, [scheduleMediaRelayout]);
  const firstListRenderDoneRef = useRef(false);
  const previousKeyboardVisibleRef = useRef<boolean | null>(null);
  const scrollOffsetYRef = useRef(0);
  const viewportHeightRef = useRef(0);
  const contentHeightRef = useRef(0);
  const userNearBottomRef = useRef(true);

  const updateNearBottom = useCallback(() => {
    const distanceToBottom =
      contentHeightRef.current - (scrollOffsetYRef.current + viewportHeightRef.current);
    userNearBottomRef.current = distanceToBottom <= 120;
  }, []);

  const handleScroll = useCallback(
    (event: NativeSyntheticEvent<NativeScrollEvent>) => {
      scrollOffsetYRef.current = event.nativeEvent.contentOffset.y;
      updateNearBottom();
    },
    [updateNearBottom]
  );

  const handleListLayout = useCallback(
    (event: { nativeEvent: { layout: { height: number } } }) => {
      viewportHeightRef.current = event.nativeEvent.layout.height;
      updateNearBottom();
      if (!isTeam) return;
      if (!firstListRenderDoneRef.current) return;
      if (!userNearBottomRef.current) return;
      const list = listRef.current;
      if (!list) return;
      requestAnimationFrame(() => {
        try {
          list.scrollToEnd({ animated: false });
        } catch {
          /* ignore */
        }
      });
    },
    [isTeam, updateNearBottom]
  );

  const handleContentSizeChange = useCallback(
    (w: number, h: number) => {
      contentHeightRef.current = h;
      onContentSizeChange(w, h);
      updateNearBottom();
      if (!firstListRenderDoneRef.current && data.length > 0) {
        firstListRenderDoneRef.current = true;
        onFirstListRender?.();
      }
    },
    [data.length, onContentSizeChange, onFirstListRender, updateNearBottom]
  );
  const renderItem: ListRenderItem<FeedItem> = useCallback(
    ({ item }) => (
      <FeedRow
        item={item}
        isTeam={isTeam}
        ownDisplayName={ownDisplayName}
        ownSenderId={ownSenderId}
        groupMeta={groupMeta}
        onOpenImage={onOpenImage}
        onOpenPdf={onOpenPdf}
        onMediaLayout={onMediaLayout}
        compact={compact}
      />
    ),
    [compact, groupMeta, isTeam, onMediaLayout, onOpenImage, onOpenPdf, ownDisplayName, ownSenderId]
  );

  const keyboardLayout = useKeyboardLayout();

  /** Réaligner le fil sur le dernier message quand le clavier s'ouvre/se ferme. */
  useEffect(() => {
    firstListRenderDoneRef.current = false;
  }, [listAnchorKey, data.length]);

  useEffect(() => {
    if (data.length === 0) return undefined;
    if (!userNearBottomRef.current) return undefined;
    const previousKeyboardVisible = previousKeyboardVisibleRef.current;
    previousKeyboardVisibleRef.current = keyboardLayout.keyboardVisible;
    if (previousKeyboardVisible == null) return undefined;
    if (previousKeyboardVisible === keyboardLayout.keyboardVisible) return undefined;
    if (!firstListRenderDoneRef.current) return undefined;
    const list = listRef.current;
    if (!list) return undefined;
    const timer = setTimeout(() => {
      try {
        list.scrollToEnd({ animated: keyboardLayout.keyboardVisible });
      } catch {
        /* ignore */
      }
    }, keyboardLayout.keyboardVisible ? 50 : 0);
    return () => clearTimeout(timer);
  }, [data.length, keyboardLayout.keyboardVisible, listAnchorKey]);

  useEffect(() => {
    if (!isTeam || data.length === 0) return;
    if (!firstListRenderDoneRef.current) return;
    if (!userNearBottomRef.current) return;
    const list = listRef.current;
    if (!list) return;
    const timer = setTimeout(() => {
      try {
        list.scrollToEnd({ animated: false });
      } catch {
        /* ignore */
      }
    }, 0);
    return () => clearTimeout(timer);
  }, [data.length, isTeam]);

  const tunedInitialNumToRender = Math.min(Math.max(initialNumToRender, 10), 14);

  return (
    <FlatList
      ref={listRef}
      data={data}
      keyExtractor={(item) => item.id}
      initialScrollIndex={initialScrollIndex}
      initialNumToRender={tunedInitialNumToRender}
      onContentSizeChange={handleContentSizeChange}
      onLayout={handleListLayout}
      onScroll={handleScroll}
      scrollEventThrottle={16}
      onScrollToIndexFailed={onScrollToIndexFailed}
      keyboardShouldPersistTaps="handled"
      keyboardDismissMode="interactive"
      removeClippedSubviews
      windowSize={5}
      maxToRenderPerBatch={8}
      updateCellsBatchingPeriod={40}
      maintainVisibleContentPosition={
        isTeam
          ? {
              minIndexForVisible: 1,
            }
          : undefined
      }
      style={[styles.listFill, isTeam && styles.listTeamBg]}
      contentContainerStyle={[
        styles.list,
        compact && !isTeam && styles.listCompact,
        isTeam && styles.listTeam,
        isTeam && { paddingBottom: LIST_BOTTOM_GAP },
      ]}
      ListHeaderComponent={
        onLoadMore ? (
          <Pressable onPress={onLoadMore} disabled={loadingMore} style={styles.loadMore}>
            <AppText variant="caption" style={styles.loadMoreText}>
              {loadingMore ? "Chargement…" : "Messages plus anciens"}
            </AppText>
          </Pressable>
        ) : null
      }
      renderItem={renderItem}
    />
  );
}

const styles = StyleSheet.create({
  listFill: { flex: 1 },
  listTeamBg: { backgroundColor: TEAM_CHAT_BACKGROUND },
  list: { paddingVertical: 8, gap: 4 },
  listCompact: { paddingVertical: 4 },
  listTeam: { paddingVertical: 4, paddingHorizontal: 2 },
  compactBubbleLegacy: { transform: [{ scale: 0.96 }], marginVertical: -2 },
  readReceipt: {
    alignSelf: "flex-end",
    marginRight: 8,
    marginTop: -4,
    marginBottom: 0,
  },
  loadMore: { alignItems: "center", paddingVertical: 8 },
  loadMoreText: { color: M.BRAND, fontWeight: "600" },
});
