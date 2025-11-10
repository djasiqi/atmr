import React, { useCallback, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { useFocusEffect } from "@react-navigation/native";
import dayjs from "dayjs";

import { useAuth } from "@/hooks/useAuth";
import {
  getDispatchMessages,
  sendDispatchMessage,
} from "@/services/enterpriseDispatch";
import { DispatchMessage } from "@/types/enterpriseDispatch";

const palette = {
  background: "#07130E",
  heroText: "#E6F2EA",
  heroMeta: "rgba(184,214,198,0.75)",
  heroBorder: "rgba(46,128,94,0.32)",
  outgoingBg: "rgba(30,185,128,0.22)",
  outgoingBorder: "rgba(30,185,128,0.35)",
  incomingBg: "#0F2C21",
  incomingBorder: "rgba(59,143,105,0.28)",
  timestamp: "rgba(184,214,198,0.6)",
  sender: "rgba(184,214,198,0.85)",
  inputBg: "rgba(10,34,26,0.9)",
  inputBorder: "rgba(59,143,105,0.28)",
  inputPlaceholder: "rgba(184,214,198,0.55)",
  sendButton: "#1EB980",
  sendButtonDisabled: "rgba(30,185,128,0.32)",
  sendIcon: "#052015",
  error: "#F87171",
};

const CHAT_FETCH_LIMIT = 60;
const DAYS_WINDOW = 7;

const sortMessagesAsc = (items: DispatchMessage[]) =>
  [...items].sort(
    (a, b) =>
      new Date(a.created_at).valueOf() - new Date(b.created_at).valueOf()
  );

export default function EnterpriseChatScreen() {
  const { enterpriseSession } = useAuth();
  const [messages, setMessages] = useState<DispatchMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [oldestTimestamp, setOldestTimestamp] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const listRef = useRef<FlatList<DispatchMessage>>(null);

  const mergeMessages = useCallback(
    (incoming: DispatchMessage[], { replace }: { replace: boolean }) => {
      setMessages((prev) => {
        const base = replace ? [] : prev;
        const map = new Map<string | number, DispatchMessage>();
        base.forEach((msg) => {
          map.set(msg.id, msg);
        });
        incoming.forEach((msg) => {
          map.set(msg.id, msg);
        });
        return sortMessagesAsc(Array.from(map.values()));
      });
    },
    []
  );

  const loadRecentMessages = useCallback(
    async ({
      replace = false,
      scrollToEnd = false,
      showLoader = false,
    }: {
      replace?: boolean;
      scrollToEnd?: boolean;
      showLoader?: boolean;
    } = {}) => {
      if (!enterpriseSession) return;
      if (showLoader) setLoading(true);
      try {
        const fetched = await getDispatchMessages({
          limit: CHAT_FETCH_LIMIT,
        });
        const sorted = sortMessagesAsc(fetched);
        const cutoff = dayjs().subtract(DAYS_WINDOW, "day");
        const recentOnly = sorted.filter(
          (msg) => !dayjs(msg.created_at).isBefore(cutoff)
        );
        const display = recentOnly.length > 0 && replace ? recentOnly : sorted;

        mergeMessages(display, { replace });

        const earliest = sorted[0];
        setOldestTimestamp(earliest?.created_at ?? null);
        setHasMore(
          sorted.length >= CHAT_FETCH_LIMIT ||
            (earliest ? dayjs(earliest.created_at).isBefore(cutoff) : false)
        );
        setError(null);
        if (scrollToEnd) {
          requestAnimationFrame(() =>
            listRef.current?.scrollToEnd({ animated: true })
          );
        } else {
          requestAnimationFrame(() =>
            listRef.current?.scrollToEnd({ animated: false })
          );
        }
      } catch (err: any) {
        if (!loading || showLoader) {
          const message =
            err?.response?.data?.error ??
            err?.message ??
            "Impossible de charger les messages.";
          setError(message);
        }
      } finally {
        if (showLoader) {
          setLoading(false);
        } else {
          setLoading(false);
        }
      }
    },
    [enterpriseSession, mergeMessages]
  );

  const loadOlderMessages = useCallback(async () => {
    if (!enterpriseSession || !hasMore || loadingMore || !oldestTimestamp) {
      return;
    }
    setLoadingMore(true);
    try {
      const older = await getDispatchMessages({
        before: oldestTimestamp,
        limit: CHAT_FETCH_LIMIT,
      });
      if (older.length === 0) {
        setHasMore(false);
        return;
      }
      const sortedOlder = sortMessagesAsc(older);
      setOldestTimestamp(sortedOlder[0]?.created_at ?? oldestTimestamp);
      mergeMessages(sortedOlder, { replace: false });
      if (sortedOlder.length < CHAT_FETCH_LIMIT) {
        setHasMore(false);
      }
      setError(null);
    } catch (err: any) {
      const message =
        err?.response?.data?.error ??
        err?.message ??
        "Impossible de charger plus de messages.";
      setError(message);
    } finally {
      setLoadingMore(false);
    }
  }, [enterpriseSession, hasMore, loadingMore, mergeMessages, oldestTimestamp]);

  useFocusEffect(
    useCallback(() => {
      if (!enterpriseSession) return;
      loadRecentMessages({
        replace: true,
        scrollToEnd: true,
        showLoader: true,
      });
      pollRef.current = setInterval(
        () => loadRecentMessages({ replace: false, scrollToEnd: false }),
        5000
      );
      return () => {
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      };
    }, [enterpriseSession, loadRecentMessages])
  );

  const handleSend = async () => {
    const content = input.trim();
    if (!content || sending) return;
    setSending(true);
    try {
      const message = await sendDispatchMessage(content);
      setMessages((prev) => {
        const dedup = new Map<string | number, DispatchMessage>();
        [...prev, message].forEach((item) => {
          dedup.set(item.id, item);
        });
        return sortMessagesAsc(Array.from(dedup.values()));
      });
      setError(null);
      requestAnimationFrame(() =>
        listRef.current?.scrollToEnd({ animated: true })
      );
      await loadRecentMessages({ replace: false, scrollToEnd: true });
      setInput("");
    } catch (err: any) {
      const message =
        err?.response?.data?.error ??
        err?.message ??
        "Impossible d’envoyer le message.";
      setError(message);
    } finally {
      setSending(false);
    }
  };

  const handleScroll = useCallback(
    ({ nativeEvent }: { nativeEvent: any }) => {
      if (
        nativeEvent.contentOffset?.y <= 24 &&
        hasMore &&
        !loadingMore &&
        !loading
      ) {
        loadOlderMessages();
      }
    },
    [hasMore, loadingMore, loadOlderMessages, loading]
  );

  const renderItem = ({ item }: { item: DispatchMessage }) => {
    const isCompany = (item.sender_role || "").toUpperCase() === "COMPANY";
    const bubbleStyles = [
      styles.messageBubble,
      isCompany ? styles.outgoingBubble : styles.incomingBubble,
      isCompany ? styles.outgoingShadow : styles.incomingShadow,
    ];
    return (
      <View
        style={[
          styles.messageContainer,
          isCompany ? styles.messageOutgoing : styles.messageIncoming,
        ]}
      >
        <View style={bubbleStyles}>
          <View style={styles.messageHeader}>
            <Text style={styles.messageSender}>
              {item.sender_name || (isCompany ? "Vous" : "Dispatch")}
            </Text>
            <Text style={styles.messageTime}>
              {new Date(item.created_at).toLocaleTimeString("fr-CH", {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </Text>
          </View>
          <Text style={styles.messageContent}>{item.content}</Text>
        </View>
      </View>
    );
  };

  const canSend = !!input.trim() && !sending;

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      {loading ? (
        <View style={styles.loading}>
          <ActivityIndicator color={palette.sendButton} />
        </View>
      ) : (
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id.toString()}
          renderItem={renderItem}
          contentContainerStyle={styles.listContent}
          ListHeaderComponent={
            hasMore ? (
              <View style={styles.loadMoreWrapper}>
                {loadingMore ? (
                  <ActivityIndicator color={palette.heroMeta} size="small" />
                ) : (
                  <TouchableOpacity
                    style={styles.loadMoreButton}
                    onPress={loadOlderMessages}
                  >
                    <Ionicons
                      name="time-outline"
                      size={16}
                      color={palette.heroText}
                    />
                    <Text style={styles.loadMoreText}>
                      Voir l’historique plus ancien
                    </Text>
                  </TouchableOpacity>
                )}
              </View>
            ) : null
          }
          ListEmptyComponent={
            <View style={styles.emptyState}>
              <Ionicons
                name="chatbubble-ellipses-outline"
                size={28}
                color={palette.heroMeta}
              />
              <Text style={styles.emptyStateText}>
                Aucune conversation pour le moment. Démarre un échange
                ci-dessous.
              </Text>
            </View>
          }
          refreshing={loading && !loadingMore}
          onRefresh={() =>
            loadRecentMessages({
              replace: true,
              scrollToEnd: true,
              showLoader: true,
            })
          }
          onScroll={handleScroll}
          scrollEventThrottle={100}
          maintainVisibleContentPosition={{
            minIndexForVisible: 0,
          }}
        />
      )}

      {error ? (
        <View style={styles.errorBanner}>
          <Ionicons
            name="alert-circle-outline"
            size={18}
            color={palette.error}
          />
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity
            onPress={() =>
              loadRecentMessages({
                replace: false,
                scrollToEnd: true,
                showLoader: true,
              })
            }
            style={styles.retryButton}
          >
            <Ionicons name="refresh" size={16} color={palette.heroText} />
          </TouchableOpacity>
        </View>
      ) : null}

      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Écrire un message…"
          placeholderTextColor={palette.inputPlaceholder}
          multiline
        />
        <TouchableOpacity
          style={[styles.sendButton, !canSend && styles.sendButtonDisabled]}
          onPress={handleSend}
          disabled={!canSend}
          activeOpacity={0.85}
        >
          {sending ? (
            <ActivityIndicator size="small" color={palette.sendIcon} />
          ) : (
            <Ionicons name="send" size={20} color={palette.sendIcon} />
          )}
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: palette.background,
  },
  loading: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  listContent: {
    padding: 20,
    paddingBottom: 90,
    gap: 12,
  },
  emptyState: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 40,
    gap: 8,
  },
  emptyStateText: {
    color: palette.heroMeta,
    fontSize: 13,
    textAlign: "center",
    lineHeight: 18,
  },
  loadMoreWrapper: {
    paddingVertical: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  loadMoreButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: palette.heroBorder,
    backgroundColor: "rgba(10,34,26,0.65)",
  },
  loadMoreText: {
    color: palette.heroText,
    fontSize: 12,
    fontWeight: "600",
  },
  messageContainer: {
    maxWidth: "82%",
    marginBottom: 8,
  },
  messageOutgoing: {
    alignSelf: "flex-end",
  },
  messageIncoming: {
    alignSelf: "flex-start",
  },
  messageBubble: {
    borderRadius: 18,
    padding: 14,
    overflow: "hidden",
  },
  outgoingBubble: {
    backgroundColor: palette.outgoingBg,
  },
  incomingBubble: {
    backgroundColor: palette.incomingBg,
  },
  outgoingShadow: {
    shadowColor: palette.outgoingBorder,
    shadowOpacity: 0.25,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 14,
    elevation: 4,
  },
  incomingShadow: {
    shadowColor: palette.incomingBorder,
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 10,
    elevation: 2,
  },
  messageHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
    marginBottom: 6,
  },
  messageSender: {
    color: palette.sender,
    fontSize: 12,
    fontWeight: "600",
  },
  messageContent: {
    color: palette.heroText,
    fontSize: 15,
    lineHeight: 21,
  },
  messageTime: {
    color: palette.timestamp,
    fontSize: 11,
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 12,
    paddingHorizontal: 18,
    paddingVertical: 18,
    paddingBottom: 80,
    borderTopWidth: 1,
    borderColor: palette.heroBorder,
    backgroundColor: "rgba(5,21,16,0.92)",
  },
  input: {
    flex: 1,
    backgroundColor: palette.inputBg,
    borderRadius: 18,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: palette.heroText,
    borderWidth: 1,
    borderColor: palette.inputBorder,
    maxHeight: 120,
  },
  sendButton: {
    backgroundColor: palette.sendButton,
    borderRadius: 16,
    paddingHorizontal: 18,
    paddingVertical: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  sendButtonDisabled: {
    backgroundColor: palette.sendButtonDisabled,
  },
  errorBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: palette.heroBorder,
    backgroundColor: "rgba(241,104,104,0.12)",
  },
  errorText: {
    color: palette.error,
    flex: 1,
    fontSize: 12,
  },
  retryButton: {
    padding: 6,
    borderRadius: 999,
    backgroundColor: "rgba(10,34,26,0.6)",
  },
});
