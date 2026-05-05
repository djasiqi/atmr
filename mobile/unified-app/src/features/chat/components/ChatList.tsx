import { useCallback, useLayoutEffect, useRef, useEffect } from "react";
import {
  FlatList,
  type ListRenderItem,
  Pressable,
  StyleSheet,
  View,
  type FlatList as FlatListType,
} from "react-native";
import { useAppViewport } from "../../../design/responsive/useAppViewport";
import { AppText } from "../../../design/ui/AppText";
import type { SharedChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";

const CHAT_STRIP_BG = "#fafafa";
const CHAT_STRIP_TOP_BORDER = "#e5e7eb";
/**
 * Marge de lecture horizontale + `safeLeft` / `safeRight` depuis `useAppViewport`
 * (même normalisation que le reste de l’app ; pas d’insets bruts ici).
 */
const CONTENT_GUTTER = 16;

export type ChatListInitialScroll = { type: "last" } | { type: "index"; index: number };

type ChatListProps = {
  messages: SharedChatMessage[];
  loading?: boolean;
  emptyLabel?: string;
  onLoadMore?: () => void;
  loadMoreLabel?: string;
  loadMoreDisabled?: boolean;
  loadingMore?: boolean;
  onOpenImage?: (url: string) => void;
  onOpenPdf?: (url: string) => void;
  /**
   * Incrémenté à chaque focus d’écran (ex. onglet chat) pour réappliquer l’ancre.
   */
  listAnchorKey?: number;
  /** Si absent, équivalent à `{ type: "last" }`. */
  initialScroll?: ChatListInitialScroll;
  /**
   * Annule le padding horizontal du parent (ex. 24) pour un fond bord-à-bord
   * dans la zone sûre ; le contenu reste géré par la gouttière + safe area.
   */
  bleedOverParentPadding?: number;
};

export function ChatList({
  messages,
  loading = false,
  emptyLabel = "Aucun message pour le moment.",
  onLoadMore,
  loadMoreLabel = "Charger plus ancien",
  loadMoreDisabled = false,
  loadingMore = false,
  onOpenImage,
  onOpenPdf,
  listAnchorKey = 0,
  initialScroll = { type: "last" },
  bleedOverParentPadding = 24,
}: ChatListProps) {
  const { safeLeft, safeRight } = useAppViewport();
  const listRef = useRef<FlatListType<SharedChatMessage> | null>(null);
  /** Demande de scroll en bas : incrémentée à chaque ancre ; satisfaite après scroll réussi ou repli timeout. */
  const scrollToEndRequestRef = useRef(0);
  const satisfiedScrollToEndRequestRef = useRef(0);
  const scrollToEndFallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Retarde `satisfied` après la fin des variations de hauteur (virtualisation / web). */
  const scrollEndSettleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const contentPad = {
    paddingLeft: CONTENT_GUTTER + safeLeft,
    paddingRight: CONTENT_GUTTER + safeRight,
  };
  const prevAnchorKeyRef = useRef<number | null>(null);
  const hadMessagesRef = useRef(false);

  const tryScrollToEndForRequest = useCallback(
    (requestId: number) => {
      if (requestId === 0 || satisfiedScrollToEndRequestRef.current >= requestId) return;
      const list = listRef.current;
      if (!list || messages.length === 0) return;
      list.scrollToEnd({ animated: false });
      satisfiedScrollToEndRequestRef.current = requestId;
    },
    [messages.length]
  );

  const applyInitialScroll = useCallback(() => {
    const list = listRef.current;
    if (!list || messages.length === 0) return;
    if (initialScroll.type === "last") {
      list.scrollToEnd({ animated: false });
      requestAnimationFrame(() => {
        list.scrollToEnd({ animated: false });
      });
      return;
    }
    const index = Math.min(Math.max(0, initialScroll.index), messages.length - 1);
    try {
      list.scrollToIndex({ index, animated: false, viewPosition: 0 });
    } catch {
      list.scrollToEnd({ animated: false });
    }
  }, [initialScroll, messages.length]);

  const onScrollToIndexFailed = useCallback(
    (info: { index: number; averageItemLength: number }) => {
      const list = listRef.current;
      if (!list) return;
      const offset = Math.max(0, info.index * (info.averageItemLength || 80));
      list.scrollToOffset({ offset, animated: false });
      setTimeout(() => {
        try {
          list.scrollToIndex({ index: info.index, animated: false, viewPosition: 0 });
        } catch {
          list.scrollToEnd({ animated: false });
        }
      }, 50);
    },
    []
  );

  useLayoutEffect(() => {
    if (loading) return;
    if (messages.length === 0) {
      hadMessagesRef.current = false;
      return;
    }
    const anchorBumped = prevAnchorKeyRef.current !== listAnchorKey;
    const firstDataPaint = !hadMessagesRef.current;
    if (!anchorBumped && !firstDataPaint) return;
    hadMessagesRef.current = true;
    prevAnchorKeyRef.current = listAnchorKey;
    if (initialScroll.type === "last") {
      scrollToEndRequestRef.current += 1;
    }
    const requestId = scrollToEndRequestRef.current;
    if (scrollToEndFallbackTimerRef.current) {
      clearTimeout(scrollToEndFallbackTimerRef.current);
      scrollToEndFallbackTimerRef.current = null;
    }
    if (scrollEndSettleTimerRef.current) {
      clearTimeout(scrollEndSettleTimerRef.current);
      scrollEndSettleTimerRef.current = null;
    }
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        applyInitialScroll();
        if (initialScroll.type === "last" && requestId > 0) {
          scrollToEndFallbackTimerRef.current = setTimeout(() => {
            scrollToEndFallbackTimerRef.current = null;
            tryScrollToEndForRequest(requestId);
          }, 320);
        }
      });
    });
  }, [listAnchorKey, loading, messages.length, initialScroll, applyInitialScroll, tryScrollToEndForRequest]);

  useEffect(
    () => () => {
      if (scrollToEndFallbackTimerRef.current) {
        clearTimeout(scrollToEndFallbackTimerRef.current);
      }
      if (scrollEndSettleTimerRef.current) {
        clearTimeout(scrollEndSettleTimerRef.current);
      }
    },
    []
  );

  const onContentSizeChange = useCallback(() => {
    if (initialScroll.type !== "last" || messages.length === 0 || loading) return;
    const requestId = scrollToEndRequestRef.current;
    if (requestId === 0 || satisfiedScrollToEndRequestRef.current >= requestId) return;
    const list = listRef.current;
    if (!list) return;
    list.scrollToEnd({ animated: false });
    requestAnimationFrame(() => {
      list.scrollToEnd({ animated: false });
    });
    if (scrollEndSettleTimerRef.current) {
      clearTimeout(scrollEndSettleTimerRef.current);
    }
    scrollEndSettleTimerRef.current = setTimeout(() => {
      scrollEndSettleTimerRef.current = null;
      const l = listRef.current;
      if (!l || scrollToEndRequestRef.current !== requestId) return;
      if (satisfiedScrollToEndRequestRef.current >= requestId) return;
      l.scrollToEnd({ animated: false });
      satisfiedScrollToEndRequestRef.current = requestId;
    }, 140);
  }, [initialScroll.type, messages.length, loading]);

  const renderItem: ListRenderItem<SharedChatMessage> = useCallback(
    ({ item }) => (
      <MessageBubble message={item} onOpenImage={onOpenImage} onOpenPdf={onOpenPdf} />
    ),
    [onOpenImage, onOpenPdf]
  );

  const keyExtractor = useCallback((item: SharedChatMessage) => String(item.id), []);

  const header = onLoadMore ? (
    <Pressable
      onPress={onLoadMore}
      disabled={loadMoreDisabled || loadingMore}
      style={{
        borderWidth: 1,
        borderColor: "#e5e7eb",
        borderRadius: 10,
        paddingVertical: 10,
        paddingHorizontal: 12,
        marginBottom: 12,
        backgroundColor: "#fff",
        opacity: loadMoreDisabled || loadingMore ? 0.6 : 1,
      }}
    >
      <AppText variant="label" style={{ textAlign: "center" }}>
        {loadingMore ? "Chargement…" : loadMoreLabel}
      </AppText>
    </Pressable>
  ) : null;

  const emptyC =
    loading && messages.length === 0 ? (
      <View style={{ paddingVertical: 24, alignItems: "center" }}>
        <AppText variant="bodyMuted" style={{ color: "#6b7280" }}>
          Chargement des messages…
        </AppText>
      </View>
    ) : (
      <View style={{ paddingVertical: 24, alignItems: "center" }}>
        <AppText variant="bodyMuted">{emptyLabel}</AppText>
      </View>
    );

  return (
    <View
      style={[
        styles.bleedStrip,
        { marginHorizontal: -bleedOverParentPadding, backgroundColor: CHAT_STRIP_BG },
      ]}
    >
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={keyExtractor}
        renderItem={renderItem}
        style={styles.flat}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="interactive"
        onLayout={() => {
          if (initialScroll.type !== "last" || messages.length === 0 || loading) return;
          const requestId = scrollToEndRequestRef.current;
          if (requestId === 0 || satisfiedScrollToEndRequestRef.current >= requestId) return;
          const list = listRef.current;
          if (!list) return;
          requestAnimationFrame(() => {
            list.scrollToEnd({ animated: false });
          });
        }}
        onContentSizeChange={onContentSizeChange}
        onScrollToIndexFailed={onScrollToIndexFailed}
        ListHeaderComponent={header}
        ListEmptyComponent={messages.length === 0 ? emptyC : null}
        contentContainerStyle={[
          styles.listContent,
          contentPad,
          { paddingTop: 12, paddingBottom: 20, flexGrow: 1 },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  bleedStrip: {
    flex: 1,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderColor: CHAT_STRIP_TOP_BORDER,
  },
  flat: {
    flex: 1,
  },
  listContent: {
    backgroundColor: CHAT_STRIP_BG,
  },
});
