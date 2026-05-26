import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Platform, StyleSheet, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import * as Linking from "expo-linking";
import { StatusBar } from "expo-status-bar";
import { CompanyContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { contextRealtimeRouter } from "../../../../src/core/realtime/contextRealtimeRouter";
import { normalizeCompanyEventType } from "../../../../src/core/realtime/eventContracts";
import { isFeatureEnabled } from "../../../../src/core/featureFlags/registry";
import { useSession } from "../../../../src/core/sessionProvider";
import { Screen, AppText, useAppViewport } from "../../../../src/design/responsive";
import {
  ChatComposer,
  ChatConversationShell,
  ImagePreviewModal,
  PdfPreviewModal,
} from "../../../../src/features/chat";
import { resolveConversationId, fetchThreadMessages } from "../../../../src/features/driver/messages/api";
import { ChannelConversationHeader } from "../../../../src/features/driver/messages/components/ChannelConversationHeader";
import { HubMessageFeed } from "../../../../src/features/driver/messages/components/HubMessageFeed";
import { QuickRepliesBar } from "../../../../src/features/driver/messages/components/QuickRepliesBar";
import { buildFallbackHubThread } from "../../../../src/features/driver/messages/channelThreadFallback";
import type { HubChatMessage } from "../../../../src/features/driver/messages/types";
import { M } from "../../../../src/features/messaging/messagingTheme";
import { resolveCompanyQuickRepliesMode } from "../../../../src/features/company/messages/companyChannelQuickReplies";
import {
  useCompanyHubRealtimeSync,
  useCompanyMessageHubThreads,
  useCompanySendHubMessage,
  useCompanySyncPresenceStatus,
  useCompanyThreadMessages,
  useMarkCompanyThreadRead,
} from "../../../../src/features/company/messages/hooks";
import { useCompanyNumericId } from "../../../../src/features/company/messages/companyId";
import { mapCompanyRealtimeToHubMessage } from "../../../../src/features/company/messages/mapRealtimeMessage";
import {
  mergeHubChatMessageLists,
  resolveHubMessageLocalId,
  upsertHubChatMessage,
} from "../../../../src/features/driver/messages/mergeHubChatMessages";
import { usePerfScreenReady } from "../../../../src/core/observability/usePerfScreenReady";
import { useActiveThreadScreenTracking } from "../../../../src/core/notifications/useActiveScreenTracking";

function messageBelongsToThread(
  parsed: HubChatMessage,
  activeThreadId: string,
  activeConversationId?: number | null
): boolean {
  if (
    activeConversationId != null &&
    parsed.conversation_id != null &&
    Number.isFinite(Number(parsed.conversation_id)) &&
    Number(parsed.conversation_id) === Number(activeConversationId)
  ) {
    return true;
  }
  return (parsed.thread_id ?? activeThreadId) === activeThreadId;
}

export default function CompanyMessageConversationScreen() {
  const { threadId: threadIdParam } = useLocalSearchParams<{ threadId: string }>();
  const threadId = typeof threadIdParam === "string" ? threadIdParam : "dispatch";
  useActiveThreadScreenTracking(threadId, "company");
  const router = useRouter();
  const { horizontalPadding, topInset } = useAppViewport();
  const { bootstrap, activeContext } = useSession();
  const companyId = useCompanyNumericId();
  const threadsQuery = useCompanyMessageHubThreads(companyId);
  const messagesQuery = useCompanyThreadMessages(companyId, threadId);
  const markRead = useMarkCompanyThreadRead(companyId);
  const markReadRef = useRef(markRead);
  markReadRef.current = markRead;
  const sendMutation = useCompanySendHubMessage(companyId, threadId);
  const httpOk = messagesQuery.isSuccess && !messagesQuery.isError;
  const syncStatus = useCompanySyncPresenceStatus(httpOk);
  useCompanyHubRealtimeSync(companyId, { excludeThreadId: threadId });
  const chatSocketEnabled = isFeatureEnabled("company_mobile_chat_socket_enabled");
  usePerfScreenReady(
    "company.thread",
    "company.thread.data_ready",
    (messagesQuery.isSuccess || messagesQuery.isError) && !threadsQuery.isLoading
  );

  const thread = useMemo(
    () => threadsQuery.data?.threads.find((t) => t.thread_id === threadId),
    [threadId, threadsQuery.data?.threads]
  );

  const [input, setInput] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);
  const [liveMessages, setLiveMessages] = useState<HubChatMessage[]>([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [listAnchorKey, setListAnchorKey] = useState(0);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [pdfPreview, setPdfPreview] = useState<{ url: string; filename: string | null } | null>(null);

  const ownUserId = useMemo(() => {
    const candidate = bootstrap?.user?.id;
    return typeof candidate === "number" || typeof candidate === "string" ? String(candidate) : null;
  }, [bootstrap?.user?.id]);

  const ownDisplayName = useMemo(() => {
    const name = bootstrap?.user?.first_name;
    return typeof name === "string" && name.trim() ? name.trim() : "Exploitation";
  }, [bootstrap?.user?.first_name]);

  useEffect(() => {
    if (!chatSocketEnabled || !activeContext || activeContext.context_type !== "company") return;
    return contextRealtimeRouter.subscribe(activeContext.context_id, (event) => {
      const eventType = normalizeCompanyEventType((event as { event_type?: unknown }).event_type);
      if (eventType !== "booking_message_sent" && eventType !== "team_chat_message") return;
      const parsed = mapCompanyRealtimeToHubMessage(event);
      if (!parsed) return;
      if (!messageBelongsToThread(parsed, threadId, thread?.conversation_id ?? null)) return;
      setLiveMessages((prev) => upsertHubChatMessage(prev, parsed));
    });
  }, [activeContext, chatSocketEnabled, thread?.conversation_id, threadId]);

  const mergedMessages = useMemo(
    () => mergeHubChatMessageLists(messagesQuery.data ?? [], liveMessages),
    [liveMessages, messagesQuery.data]
  );

  useEffect(() => {
    setLiveMessages([]);
    setListAnchorKey((k) => k + 1);
  }, [threadId]);

  const handleInitialScrollSettled = useCallback(() => {
    if (companyId && threadId) {
      void markReadRef.current.mutateAsync(threadId).catch(() => undefined);
    }
  }, [companyId, threadId]);

  const applyOutbound = (optimistic: HubChatMessage, confirmed?: HubChatMessage) => {
    const next = confirmed ?? optimistic;
    setLiveMessages((prev) =>
      upsertHubChatMessage(prev, next, {
        replaceLocalId:
          resolveHubMessageLocalId(optimistic) ??
          resolveHubMessageLocalId(next) ??
          String(optimistic.id),
      })
    );
    setSendError(null);
  };

  const emitMessage = (payload: Record<string, unknown>, optimistic: HubChatMessage) => {
    if (!companyId) {
      setSendError("Connexion indisponible. Réessayez.");
      return false;
    }
    const outbound: Record<string, unknown> = {
      ...payload,
      thread_id: threadId,
      booking_id: thread?.booking_id ?? null,
      conversation_id: thread?.conversation_id ?? null,
    };
    applyOutbound(optimistic);
    void sendMutation
      .mutateAsync(outbound)
      .then((confirmed) => applyOutbound(optimistic, confirmed))
      .catch(() => setSendError("Envoi impossible. Réessayez."));
    return true;
  };

  const sendText = (text: string): boolean => {
    const content = text.trim();
    if (!content) return false;
    const localId = `local-${Date.now()}`;
    const ok = emitMessage(
      { content, _localId: localId },
      {
        id: localId,
        _localId: localId,
        sender_id: ownUserId,
        content,
        sender_role: "COMPANY",
        sender_name: ownDisplayName,
        timestamp: new Date().toISOString(),
        thread_id: threadId,
        booking_id: thread?.booking_id ?? null,
        message_type: "text",
        priority: "normal",
      }
    );
    if (ok) setInput("");
    return ok;
  };

  const sendVoiceMessage = (uri: string) => {
    const localId = `local-voice-${Date.now()}`;
    emitMessage(
      { content: "Message vocal", audio_url: uri, _localId: localId },
      {
        id: localId,
        _localId: localId,
        sender_id: ownUserId,
        content: "Message vocal",
        sender_role: "COMPANY",
        sender_name: ownDisplayName,
        timestamp: new Date().toISOString(),
        audio_url: uri,
        thread_id: threadId,
        message_type: "text",
        priority: "normal",
      }
    );
  };

  const loadMore = async () => {
    if (!companyId || loadingMore || mergedMessages.length === 0) return;
    setLoadingMore(true);
    try {
      const older = await fetchThreadMessages(companyId, threadId, {
        before: mergedMessages[0]?.timestamp,
        limit: 40,
      });
      setLiveMessages((prev) => [...older, ...prev]);
    } catch {
      setSendError("Impossible de charger les messages plus anciens.");
    } finally {
      setLoadingMore(false);
    }
  };

  const [resolvedConversationId, setResolvedConversationId] = useState<number | null>(null);

  useEffect(() => {
    if (thread?.conversation_id != null) {
      setResolvedConversationId(thread.conversation_id);
      return;
    }
    if (threadId !== "dispatch" || !companyId) {
      setResolvedConversationId(null);
      return;
    }
    void resolveConversationId(companyId, threadId)
      .then((id) => setResolvedConversationId(id))
      .catch(() => setResolvedConversationId(null));
  }, [companyId, thread?.conversation_id, threadId]);

  const openDispatchManage = useCallback(() => {
    const conversationId = resolvedConversationId ?? thread?.conversation_id;
    if (conversationId == null) return;
    router.push({
      pathname: "/(app)/(company)/messages/[threadId]/manage",
      params: {
        threadId,
        conversationId: String(conversationId),
      },
    });
  }, [resolvedConversationId, router, thread?.conversation_id, threadId]);

  const isDispatchManageable = threadId === "dispatch";
  const isMissionThread = Boolean(thread?.booking_id) || threadId.startsWith("mission:");
  const quickRepliesMode = resolveCompanyQuickRepliesMode(threadId, thread?.booking_id);
  const headerThread = thread ?? buildFallbackHubThread(threadId);

  const openMissionDetails = useCallback(() => {
    const bookingId = thread?.booking_id ?? (threadId.startsWith("mission:") ? threadId.split(":")[1] : null);
    if (!bookingId) return;
    router.push({
      pathname: "/(app)/(company)/ride-details",
      params: { rideId: String(bookingId) },
    });
  }, [router, thread?.booking_id, threadId]);

  return (
    <CompanyContextGuard>
      <PermissionGuard permission="company:dashboard:read">
        <StatusBar style="dark" backgroundColor="#FFFFFF" />
        <Screen
          scroll={false}
          backgroundColor={M.PAGE_BG}
          withHorizontalPadding={false}
          safeTop={false}
          safeBottom={false}
          keyboardAware={Platform.OS === "ios"}
          keyboardVerticalOffset={0}
          pageTransition={false}
        >
          <ChatConversationShell
            style={styles.wrapChannel}
            horizontalPadding={horizontalPadding}
            pageBackground={M.PAGE_BG}
            header={
              <View style={[styles.chrome, { paddingTop: topInset }]}>
                <ChannelConversationHeader
                  thread={headerThread}
                  syncStatus={syncStatus}
                  headerManageable={isDispatchManageable}
                  onPressHeader={isDispatchManageable ? openDispatchManage : undefined}
                  onOpenDetails={isMissionThread ? openMissionDetails : undefined}
                />
              </View>
            }
            feed={
              <View style={[styles.feed, styles.feedChannel]}>
                <HubMessageFeed
                  messages={mergedMessages}
                  ownSenderId={ownUserId}
                  ownDisplayName={ownDisplayName}
                  compact={false}
                  variant="team"
                  initialAnchorMode={threadId === "team" ? "latest" : "first_unread"}
                  onOpenImage={setImagePreviewUrl}
                  onOpenPdf={(url) => {
                    const source = mergedMessages.find((m) => m.pdf_url === url);
                    setPdfPreview({ url, filename: source?.pdf_filename ?? null });
                  }}
                  onLoadMore={mergedMessages.length > 0 ? () => void loadMore() : undefined}
                  loadingMore={loadingMore}
                  listAnchorKey={listAnchorKey}
                  loading={messagesQuery.isLoading && mergedMessages.length === 0}
                  onInitialScrollSettled={handleInitialScrollSettled}
                />
              </View>
            }
            footer={
              <>
                <QuickRepliesBar mode={quickRepliesMode} onSelect={(content) => sendText(content)} />
                <ChatComposer
                  value={input}
                  onChangeText={setInput}
                  onSend={() => sendText(input)}
                  placeholder="Écrire un message…"
                  onPickImage={async () => {
                    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
                    if (!permission.granted) return;
                    const result = await ImagePicker.launchImageLibraryAsync({
                      mediaTypes: ImagePicker.MediaTypeOptions.Images,
                      quality: 0.7,
                    });
                    if (result.canceled || !result.assets[0]) return;
                    const localId = `local-img-${Date.now()}`;
                    emitMessage(
                      { image_url: result.assets[0].uri, _localId: localId },
                      {
                        id: localId,
                        _localId: localId,
                        sender_id: ownUserId,
                        content: "",
                        sender_role: "COMPANY",
                        sender_name: ownDisplayName,
                        timestamp: new Date().toISOString(),
                        image_url: result.assets[0].uri,
                        thread_id: threadId,
                        message_type: "text",
                        priority: "normal",
                      }
                    );
                  }}
                  onPickPdf={async () => {
                    const result = await DocumentPicker.getDocumentAsync({ copyToCacheDirectory: true });
                    if (result.canceled || !result.assets[0]) return;
                    const asset = result.assets[0];
                    const localId = `local-pdf-${Date.now()}`;
                    emitMessage(
                      {
                        pdf_url: asset.uri,
                        pdf_filename: asset.name,
                        _localId: localId,
                      },
                      {
                        id: localId,
                        _localId: localId,
                        sender_id: ownUserId,
                        content: asset.name ?? "Document",
                        sender_role: "COMPANY",
                        sender_name: ownDisplayName,
                        timestamp: new Date().toISOString(),
                        pdf_url: asset.uri,
                        pdf_filename: asset.name ?? null,
                        thread_id: threadId,
                        message_type: "text",
                        priority: "normal",
                      }
                    );
                  }}
                  onVoiceMessage={sendVoiceMessage}
                />
                {sendError ? <AppText variant="error">{sendError}</AppText> : null}
              </>
            }
          />
          <ImagePreviewModal
            visible={imagePreviewUrl != null}
            imageUrl={imagePreviewUrl}
            onClose={() => setImagePreviewUrl(null)}
          />
          <PdfPreviewModal
            visible={pdfPreview != null}
            filename={pdfPreview?.filename ?? null}
            pdfUrl={pdfPreview?.url ?? null}
            onOpenPdf={() => {
              if (pdfPreview?.url) void Linking.openURL(pdfPreview.url);
            }}
            onClose={() => setPdfPreview(null)}
          />
        </Screen>
      </PermissionGuard>
    </CompanyContextGuard>
  );
}

const styles = StyleSheet.create({
  wrapChannel: { flex: 1 },
  chrome: {
    width: "100%",
    backgroundColor: M.CARD,
  },
  feed: { flex: 1, minHeight: 0 },
  feedChannel: {
    borderRadius: 0,
    borderWidth: 0,
    backgroundColor: M.PAGE_BG,
  },
});
