import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Platform, StyleSheet, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import * as Linking from "expo-linking";
import { StatusBar } from "expo-status-bar";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { realtimeManager } from "../../../../src/core/realtime/realtimeManager";
import { useSession } from "../../../../src/core/sessionProvider";
import { Screen, AppText, useAppViewport } from "../../../../src/design/responsive";
import {
  ChatComposer,
  ChatComposerError,
  ChatConversationShell,
  ImagePreviewModal,
  PdfPreviewModal,
  TypingIndicator,
} from "../../../../src/features/chat";
import { uploadChatAttachment } from "../../../../src/features/chat/services/chatMediaUpload";
import {
  useAckMessage,
  useDriverCompanyId,
  useEnsureDriverSocketForHub,
  useHubMessageRealtimeSync,
  useDriverSendHubMessage,
  useMarkThreadRead,
  useMessageHubThreads,
  useMissionEtaMinutes,
  useReportEmergency,
  useSyncPresenceStatus,
  useTeamMemberCount,
  useThreadMessages,
} from "../../../../src/features/driver/messages/hooks";
import { ChannelConversationHeader } from "../../../../src/features/driver/messages/components/ChannelConversationHeader";
import { resolveChannelQuickRepliesMode } from "../../../../src/features/driver/messages/channelQuickReplies";
import { buildFallbackHubThread } from "../../../../src/features/driver/messages/channelThreadFallback";
import { QuickRepliesBar } from "../../../../src/features/driver/messages/components/QuickRepliesBar";
import { HubMessageFeed } from "../../../../src/features/driver/messages/components/HubMessageFeed";
import type { EmergencyIssueType, HubChatMessage } from "../../../../src/features/driver/messages/types";
import { fetchThreadMessages } from "../../../../src/features/driver/messages/api";
import { parseDirectThreadId } from "../../../../src/features/driver/messages/contracts";
import {
  mergeHubChatMessageLists,
  resolveHubMessageLocalId,
  upsertHubChatMessage,
} from "../../../../src/features/driver/messages/mergeHubChatMessages";
import { M } from "../../../../src/features/messaging/messagingTheme";
import { usePerfScreenReady } from "../../../../src/core/observability/usePerfScreenReady";
import { emitPerfKpi } from "../../../../src/core/observability/perfKpi";
import { useActiveThreadScreenTracking } from "../../../../src/core/notifications/useActiveScreenTracking";

function parseTeamChatMessage(event: unknown): HubChatMessage | null {
  if (!event || typeof event !== "object") return null;
  const payload = event as Record<string, unknown>;
  const timestamp =
    typeof payload.timestamp === "string" && payload.timestamp.length > 0
      ? payload.timestamp
      : new Date().toISOString();
  return {
    id:
      typeof payload.id === "number" || typeof payload.id === "string"
        ? payload.id
        : `${timestamp}-${Math.random().toString(36).slice(2, 8)}`,
    sender_id:
      typeof payload.sender_id === "number" || typeof payload.sender_id === "string"
        ? payload.sender_id
        : null,
    content: typeof payload.content === "string" ? payload.content : "",
    sender_role: typeof payload.sender_role === "string" ? payload.sender_role : undefined,
    sender_name: typeof payload.sender_name === "string" ? payload.sender_name : null,
    timestamp,
    image_url: typeof payload.image_url === "string" ? payload.image_url : null,
    pdf_url: typeof payload.pdf_url === "string" ? payload.pdf_url : null,
    pdf_filename: typeof payload.pdf_filename === "string" ? payload.pdf_filename : null,
      audio_url: typeof payload.audio_url === "string" ? payload.audio_url : null,
      receiver_id:
        typeof payload.receiver_id === "number" || typeof payload.receiver_id === "string"
          ? payload.receiver_id
          : null,
      conversation_id:
        typeof payload.conversation_id === "number"
          ? payload.conversation_id
          : typeof payload.conversation_id === "string"
            ? Number.parseInt(payload.conversation_id, 10)
            : null,
      thread_id: typeof payload.thread_id === "string" ? payload.thread_id : null,
    booking_id: typeof payload.booking_id === "number" ? payload.booking_id : null,
    message_type: typeof payload.message_type === "string" ? payload.message_type : "text",
    priority:
      payload.priority === "urgent" || payload.priority === "important"
        ? payload.priority
        : "normal",
    _localId:
      typeof payload._localId === "string" && payload._localId.length > 0
        ? payload._localId
        : null,
  };
}

export default function DriverMessageConversationScreen() {
  const { threadId: threadIdParam } = useLocalSearchParams<{ threadId: string }>();
  const threadIdEarly =
    typeof threadIdParam === "string" ? threadIdParam : Array.isArray(threadIdParam) ? threadIdParam[0] : null;
  useActiveThreadScreenTracking(threadIdEarly, "driver");
  const threadId = typeof threadIdParam === "string" ? threadIdParam : "dispatch";
  const router = useRouter();
  const { horizontalPadding, topInset } = useAppViewport();
  const { bootstrap } = useSession();
  const companyId = useDriverCompanyId();
  const threadsQuery = useMessageHubThreads(companyId);
  const thread = useMemo(
    () => threadsQuery.data?.threads.find((t) => t.thread_id === threadId),
    [threadId, threadsQuery.data?.threads]
  );
  const messagesQuery = useThreadMessages(companyId, threadId, thread?.conversation_id);
  const markRead = useMarkThreadRead(companyId);
  const markReadRef = useRef(markRead);
  markReadRef.current = markRead;
  const ackMutation = useAckMessage(companyId);
  const emergencyMutation = useReportEmergency(companyId, threadId);
  const httpOk = messagesQuery.isSuccess && !messagesQuery.isError;
  const syncStatus = useSyncPresenceStatus(httpOk);
  useEnsureDriverSocketForHub();
  useHubMessageRealtimeSync(companyId, { excludeThreadId: threadId });
  usePerfScreenReady(
    "driver.thread",
    "driver.thread.data_ready",
    (messagesQuery.isSuccess || messagesQuery.isError) && !threadsQuery.isLoading
  );

  const peerUserId = useMemo(() => {
    if (thread?.peer_user_id != null) return thread.peer_user_id;
    return parseDirectThreadId(threadId);
  }, [thread?.peer_user_id, threadId]);
  const etaQuery = useMissionEtaMinutes(thread?.booking_id ?? null, thread?.status);
  const [input, setInput] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);
  const [sendErrorOpensSettings, setSendErrorOpensSettings] = useState(false);
  const clearSendError = () => {
    setSendError(null);
    setSendErrorOpensSettings(false);
  };
  const showSendError = (message: string, opensSettings = false) => {
    setSendError(message);
    setSendErrorOpensSettings(opensSettings);
  };
  const [liveMessages, setLiveMessages] = useState<HubChatMessage[]>([]);
  const [typingNames, setTypingNames] = useState<string[]>([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [listAnchorKey, setListAnchorKey] = useState(0);
  const [composerVisible, setComposerVisible] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [pdfPreview, setPdfPreview] = useState<{ url: string; filename: string | null } | null>(null);
  const openPhaseRef = useRef({
    mountedAtMs: Date.now(),
    messagesReady: false,
    listFirstRender: false,
    composerReady: false,
    keyboardReady: false,
  });

  const emitThreadRuntime = useCallback(
    (phase: string, extra?: Record<string, unknown>) => {
      emitPerfKpi("perf.thread_runtime", {
        source: "driver.thread.runtime",
        role: "driver",
        screen: "driver.thread",
        thread_id: threadId,
        phase,
        duration_ms: Date.now() - openPhaseRef.current.mountedAtMs,
        ...extra,
      });
    },
    [threadId]
  );

  const ownUserId = useMemo(() => {
    const candidate = bootstrap?.user?.id;
    return typeof candidate === "number" || typeof candidate === "string" ? String(candidate) : null;
  }, [bootstrap?.user?.id]);
  const ownDisplayName = useMemo(() => {
    const name = bootstrap?.user?.first_name;
    return typeof name === "string" && name.trim() ? name.trim() : "Chauffeur";
  }, [bootstrap?.user?.first_name]);
  const sendMutation = useDriverSendHubMessage(companyId, threadId, {
    senderId: ownUserId,
    senderName: ownDisplayName,
    conversationId: thread?.conversation_id ?? null,
  });

  useEffect(() => {
    return realtimeManager.subscribeTeamChatEvents((event) => {
      if (event.type === "team_chat_message") {
        const parsed = parseTeamChatMessage(event.payload);
        if (!parsed) return;
        if (
          !messageBelongsToThread(
            parsed,
            threadId,
            ownUserId,
            peerUserId,
            thread?.conversation_id ?? null
          )
        ) {
          return;
        }
        setLiveMessages((prev) => upsertHubChatMessage(prev, parsed));
        const numericId = typeof parsed.id === "number" ? parsed.id : Number.parseInt(String(parsed.id), 10);
        if (Number.isFinite(numericId) && companyId) {
          void ackMutation.mutateAsync(numericId).catch(() => undefined);
        }
      }
      if (event.type === "team_chat_typing") {
        const payload = event.payload as Record<string, unknown>;
        const sender =
          typeof payload.sender_name === "string" ? payload.sender_name : "Équipe";
        setTypingNames((prev) => (prev.includes(sender) ? prev : [...prev, sender]));
        setTimeout(() => setTypingNames((prev) => prev.filter((n) => n !== sender)), 3000);
      }
    });
  }, [ackMutation, companyId, ownUserId, peerUserId, thread?.conversation_id, threadId]);

  const mergedMessages = useMemo(
    () => mergeHubChatMessageLists(messagesQuery.data ?? [], liveMessages),
    [liveMessages, messagesQuery.data]
  );

  useEffect(() => {
    setLiveMessages([]);
    setListAnchorKey((k) => k + 1);
    openPhaseRef.current = {
      mountedAtMs: Date.now(),
      messagesReady: false,
      listFirstRender: false,
      composerReady: false,
      keyboardReady: false,
    };
    emitThreadRuntime("perf.thread.mount_start");
    setComposerVisible(false);
    const rafId = requestAnimationFrame(() => {
      setComposerVisible(true);
      if (!openPhaseRef.current.composerReady) {
        openPhaseRef.current.composerReady = true;
        emitThreadRuntime("perf.thread.composer_ready");
      }
    });
    return () => cancelAnimationFrame(rafId);
  }, [emitThreadRuntime, threadId]);

  useEffect(() => {
    if (!messagesQuery.isSuccess || openPhaseRef.current.messagesReady) return;
    openPhaseRef.current.messagesReady = true;
    emitThreadRuntime("perf.thread.messages_ready", {
      message_count: Array.isArray(messagesQuery.data) ? messagesQuery.data.length : 0,
    });
    emitThreadRuntime("perf.thread.ready", {
      message_count: Array.isArray(messagesQuery.data) ? messagesQuery.data.length : 0,
    });
  }, [emitThreadRuntime, messagesQuery.data, messagesQuery.isSuccess]);

  const handleFirstListRender = useCallback(() => {
    if (openPhaseRef.current.listFirstRender) return;
    openPhaseRef.current.listFirstRender = true;
    emitThreadRuntime("perf.thread.list_first_render");
  }, [emitThreadRuntime]);

  const handleComposerInputFocus = useCallback(() => {
    if (openPhaseRef.current.keyboardReady) return;
    openPhaseRef.current.keyboardReady = true;
    emitThreadRuntime("perf.thread.keyboard_ready");
  }, [emitThreadRuntime]);

  const handleInitialScrollSettled = useCallback(() => {
    if (companyId && threadId) {
      void markReadRef.current.mutateAsync(threadId).catch(() => {
        /* hub/read peut échouer (CSRF, réseau) — ne pas crasher l'écran */
      });
    }
  }, [companyId, threadId]);

  const emitMessage = (payload: Record<string, unknown>, optimistic: HubChatMessage) => {
    const outbound: Record<string, unknown> = {
      ...payload,
      thread_id: threadId,
      booking_id: thread?.booking_id ?? null,
      conversation_id: thread?.conversation_id ?? null,
    };
    if (peerUserId != null) {
      outbound.receiver_id = peerUserId;
    }

    const applySuccess = (confirmed?: HubChatMessage, optimisticLocalId?: string) => {
      const next = confirmed ?? optimistic;
      setLiveMessages((prev) =>
        upsertHubChatMessage(prev, next, {
          replaceLocalId:
            optimisticLocalId ??
            resolveHubMessageLocalId(next) ??
            resolveHubMessageLocalId(optimistic),
        })
      );
      clearSendError();
    };

    if (realtimeManager.isDriverSocketReady()) {
      const sent = realtimeManager.emitTeamChatMessage(outbound);
      if (sent) {
        applySuccess(undefined, resolveHubMessageLocalId(optimistic) ?? String(optimistic.id));
        return true;
      }
    }

    if (!companyId) {
      showSendError("Connexion indisponible. Réessayez.");
      return false;
    }

    const localId =
      resolveHubMessageLocalId(optimistic) ?? String(optimistic.id);
    applySuccess(undefined, localId);

    void sendMutation
      .mutateAsync({ ...outbound, _localId: localId })
      .then((confirmed) => applySuccess(confirmed, localId))
      .catch(() => {
        showSendError(
          realtimeManager.isDriverSocketReady()
            ? "Envoi impossible."
            : "Connexion indisponible. Réessayez."
        );
      });
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
        sender_role: "DRIVER",
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

  const sendVoiceMessage = async (uri: string) => {
    const localId = `local-voice-${Date.now()}`;
    const optimistic: HubChatMessage = {
      id: localId,
      _localId: localId,
      sender_id: ownUserId,
      content: "Message vocal",
      sender_role: "DRIVER",
      sender_name: ownDisplayName,
      timestamp: new Date().toISOString(),
      audio_url: uri,
      thread_id: threadId,
      message_type: "text",
      priority: "normal",
    };
    setLiveMessages((prev) => upsertHubChatMessage(prev, optimistic, { replaceLocalId: localId }));
    try {
      const publicUrl = await uploadChatAttachment({ uri });
      emitMessage(
        { content: "Message vocal", audio_url: publicUrl, _localId: localId },
        { ...optimistic, audio_url: publicUrl }
      );
    } catch {
      setLiveMessages((prev) => prev.filter((m) => String(m.id) !== localId && m._localId !== localId));
      showSendError("Impossible d'envoyer le message vocal.");
    }
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
      showSendError("Impossible de charger les messages plus anciens.");
    } finally {
      setLoadingMore(false);
    }
  };

  const isMissionThread = Boolean(thread?.booking_id) || threadId.startsWith("mission:");
  const isTeamThread = threadId === "team";
  const showEmergency = isMissionThread || threadId === "dispatch";
  const showMissionPanels = isMissionThread;
  const quickRepliesMode = resolveChannelQuickRepliesMode(threadId, thread?.booking_id);
  const teamRosterQuery = useTeamMemberCount(isTeamThread ? companyId : null);
  const teamMemberCount =
    teamRosterQuery.isPending && teamRosterQuery.data === undefined
      ? null
      : (teamRosterQuery.data ?? null);

  const openColleagues = useCallback(() => {
    router.push("/(app)/(driver)/messages/colleagues");
  }, [router]);

  const headerThread = thread ?? buildFallbackHubThread(threadId);

  const handleEmergencyReport = useCallback(
    (issue: EmergencyIssueType) => {
      void emergencyMutation
        .mutateAsync({
          issue_type: issue,
          booking_id: thread?.booking_id ?? null,
        })
        .then((result) => {
          const parsed = parseEmergencySystemMessage(result, threadId);
          if (parsed) {
            setLiveMessages((prev) => upsertHubChatMessage(prev, parsed));
          }
          void messagesQuery.refetch();
          clearSendError();
        })
        .catch(() => {
          showSendError("Impossible de signaler le problème. Réessayez.");
        });
    },
    [emergencyMutation, messagesQuery, thread?.booking_id, threadId]
  );

  return (
    <DriverContextGuard>
      <PermissionGuard permission="chat:read">
        <StatusBar style="dark" backgroundColor="#FFFFFF" />
        <Screen
          scroll={false}
          backgroundColor={M.PAGE_BG}
          withHorizontalPadding={false}
          safeTop={false}
          safeBottom={false}
          keyboardAware={Platform.OS === "ios"}
          keyboardVerticalOffset={0}
        >
          <ChatConversationShell
            style={styles.wrapTeam}
            horizontalPadding={horizontalPadding}
            pageBackground={M.PAGE_BG}
            header={
              <View style={[styles.teamChrome, { paddingTop: topInset }]}>
                <ChannelConversationHeader
                  thread={headerThread}
                  syncStatus={syncStatus}
                  memberCount={isTeamThread ? teamMemberCount : undefined}
                  etaMinutes={showMissionPanels ? etaQuery.data?.eta_minutes ?? null : null}
                  onOpenColleagues={isTeamThread ? openColleagues : undefined}
                  onOpenDetails={
                    showMissionPanels
                      ? () =>
                          router.push({
                            pathname: "/(app)/(driver)/messages/[threadId]/details",
                            params: { threadId },
                          })
                      : undefined
                  }
                  emergency={
                    showEmergency
                      ? {
                          pending: emergencyMutation.isPending,
                          onReport: handleEmergencyReport,
                        }
                      : undefined
                  }
                />
              </View>
            }
            feed={
              <View style={[styles.feed, styles.feedTeam]}>
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
                  onFirstListRender={handleFirstListRender}
                />
              </View>
            }
            footer={
              <>
                <TypingIndicator visible={typingNames.length > 0} />
                <QuickRepliesBar
                  mode={quickRepliesMode}
                  onSelect={(content) => {
                    sendText(content);
                  }}
                />
                {composerVisible ? (
                  <ChatComposer
                    value={input}
                    onChangeText={setInput}
                    onSend={() => sendText(input)}
                    onInputFocus={handleComposerInputFocus}
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
                          sender_role: "DRIVER",
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
                          sender_role: "DRIVER",
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
                    onVoiceError={(message, kind) =>
                      showSendError(message, kind === "permission")
                    }
                  />
                ) : (
                  <View style={styles.composerPlaceholder} />
                )}
                {sendError ? (
                  <ChatComposerError message={sendError} openSettings={sendErrorOpensSettings} />
                ) : null}
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
    </DriverContextGuard>
  );
}

function parseEmergencySystemMessage(
  result: Record<string, unknown>,
  threadId: string
): HubChatMessage | null {
  const raw = result.message;
  if (!raw || typeof raw !== "object") return null;
  const message = raw as Record<string, unknown>;
  const id = message.id;
  if (typeof id !== "number" && typeof id !== "string") return null;
  return {
    id,
    content: typeof message.content === "string" ? message.content : "",
    sender_role: typeof message.sender_role === "string" ? message.sender_role : "DRIVER",
    sender_name:
      typeof message.sender_name === "string" && message.sender_name.trim().length > 0
        ? message.sender_name
        : null,
    timestamp:
      typeof message.timestamp === "string" && message.timestamp.length > 0
        ? message.timestamp
        : new Date().toISOString(),
    thread_id: typeof message.thread_id === "string" ? message.thread_id : threadId,
    booking_id: typeof message.booking_id === "number" ? message.booking_id : null,
    conversation_id:
      typeof message.conversation_id === "number"
        ? message.conversation_id
        : typeof message.conversation_id === "string"
          ? Number.parseInt(message.conversation_id, 10) || null
          : null,
    message_type: "system",
    priority:
      message.priority === "urgent" || message.priority === "important"
        ? message.priority
        : "normal",
  };
}

function messageBelongsToThread(
  parsed: HubChatMessage,
  activeThreadId: string,
  ownUserId: string | null,
  peerUserId: number | null | undefined,
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

  if (activeThreadId.startsWith("direct:")) {
    const peer = peerUserId ?? parseDirectThreadId(activeThreadId);
    if (peer == null || ownUserId == null) return false;
    const myNum = Number(ownUserId);
    const sender =
      parsed.sender_id != null && Number.isFinite(Number(parsed.sender_id))
        ? Number(parsed.sender_id)
        : null;
    const receiver =
      parsed.receiver_id != null && Number.isFinite(Number(parsed.receiver_id))
        ? Number(parsed.receiver_id)
        : null;
    return (
      (sender === myNum && receiver === peer) || (receiver === myNum && sender === peer)
    );
  }
  if (activeThreadId === "team") {
    return parsed.thread_id === "team";
  }
  if (activeThreadId === "dispatch") {
    if (parsed.thread_id === "dispatch") return true;
    if (
      activeConversationId != null &&
      parsed.conversation_id != null &&
      Number(parsed.conversation_id) === Number(activeConversationId)
    ) {
      return true;
    }
    return false;
  }
  if (activeThreadId === "support") {
    return parsed.thread_id === "support";
  }
  return (parsed.thread_id ?? activeThreadId) === activeThreadId;
}

const styles = StyleSheet.create({
  wrapTeam: { flex: 1 },
  teamChrome: {
    width: "100%",
    backgroundColor: M.CARD,
  },
  feed: {
    flex: 1,
    minHeight: 0,
    backgroundColor: M.PAGE_BG,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#D1D9D6",
  },
  feedTeam: {
    borderRadius: 0,
    borderWidth: 0,
    backgroundColor: M.PAGE_BG,
  },
  composerPlaceholder: {
    minHeight: 56,
  },
});
