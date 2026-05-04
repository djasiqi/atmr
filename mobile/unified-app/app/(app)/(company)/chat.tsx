import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { InteractionManager, Platform, Pressable, StyleSheet, View } from "react-native";
import * as Linking from "expo-linking";
import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { useFocusEffect } from "@react-navigation/native";
import { useLocalSearchParams } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import {
  useCompanyChatMessages,
  useCompanyChatUnread,
  useCompanyDashboardQuery,
  useCompanyRealtimeInvalidation,
  useCompanyRealtimeStatus,
  useCompanySendChatMessage,
} from "../../../src/features/company/hooks";
import { emitCompanyDispatchTelemetry } from "../../../src/features/company/telemetry/companyTelemetry";
import {
  ChatComposer,
  ChatList,
  getChatListInitialScroll,
  ImagePreviewModal,
  PdfPreviewModal,
  SharedChatMessage,
  TypingIndicator,
} from "../../../src/features/chat";
import { contextRealtimeRouter } from "../../../src/core/realtime/contextRealtimeRouter";
import { normalizeCompanyEventType } from "../../../src/core/realtime/eventContracts";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { resolveMediaUrl } from "../../../src/core/api/mediaUrl";
import type { CompanyRealtimeSnapshot, CompanyRealtimeStatus } from "../../../src/features/company/realtime/companyRealtimeState";
import { AppText, Screen, useAppViewport, useResponsiveTokens } from "../../../src/design/responsive";

function mapRealtimeChatMessage(event: unknown): SharedChatMessage | null {
  if (!event || typeof event !== "object") return null;
  const payload = event as Record<string, unknown>;
  const messageCandidate =
    (payload.message as Record<string, unknown> | undefined) ??
    (payload.data as Record<string, unknown> | undefined) ??
    payload;
  if (!messageCandidate || typeof messageCandidate !== "object") return null;
  const raw = messageCandidate as Record<string, unknown>;
  const content = typeof raw.content === "string" ? raw.content : "";
  const imageRaw = raw.image_url ?? raw.image;
  const pdfRaw = raw.pdf_url ?? raw.pdf;
  const imageUrl = resolveMediaUrl(typeof imageRaw === "string" ? imageRaw : null);
  const pdfUrl = resolveMediaUrl(typeof pdfRaw === "string" ? pdfRaw : null);
  const audioRaw = raw.audio_url ?? raw.audio;
  const audioUrl = resolveMediaUrl(typeof audioRaw === "string" ? audioRaw : null);
  if (!content.trim() && !imageUrl && !pdfUrl && !audioUrl) return null;
  const timestamp =
    (typeof raw.created_at === "string" && raw.created_at) ||
    (typeof raw.timestamp === "string" && raw.timestamp) ||
    new Date().toISOString();
  const messageIdCandidate = raw.id ?? payload.event_id ?? payload.message_id;
  const id =
    typeof messageIdCandidate === "number" || typeof messageIdCandidate === "string"
      ? messageIdCandidate
      : `${timestamp}-${Math.random().toString(36).slice(2, 8)}`;
  return {
    id,
    content,
    senderRole:
      (typeof raw.sender_type === "string" && raw.sender_type) ||
      (typeof raw.sender_role === "string" && raw.sender_role) ||
      undefined,
    senderName:
      (typeof raw.sender_label === "string" && raw.sender_label) ||
      (typeof raw.sender_name === "string" && raw.sender_name) ||
      null,
    timestamp,
    threadId:
      typeof payload.booking_id === "number" || typeof payload.booking_id === "string"
        ? String(payload.booking_id)
        : null,
    imageUrl,
    pdfUrl,
    pdfFilename: typeof raw.pdf_filename === "string" ? raw.pdf_filename : null,
    audioUrl,
  };
}

const REFRESH_FOOTER =
  "Les messages se rafraîchissent automatiquement, avec un court délai possible.";

function isCompanyRealtimeAllGood(
  status: CompanyRealtimeStatus,
  connected: boolean
): boolean {
  return status === "healthy" && connected;
}

type ConnectionBanner =
  | { show: false }
  | { show: true; tone: "caution" | "warning"; title: string; body?: string };

/**
 * Remplace les libellés techniques (failed, degraded) par un message opérationnel.
 * Quand tout est OK, on n'affiche rien : le chat n'est pas un tableau de bord d'ops.
 *
 * Sur cet écran, on n'alerte le flux Socket.IO **que** si le feature flag
 * `company_mobile_chat_socket_enabled` est actif (chat conçu pour du live).
 * Sinon le chat repose sur le repli HTTP (polling) : un bridge entreprise
 * en échec ne doit pas afficher « Connexion directe indisponible » ici.
 */
function getCompanyChatConnectionBanner(
  realtime: CompanyRealtimeSnapshot,
  companyRealtimeEnabled: boolean,
  chatSocketEnabled: boolean
): ConnectionBanner {
  if (isCompanyRealtimeAllGood(realtime.status, realtime.connected)) {
    return { show: false };
  }

  if (!companyRealtimeEnabled) {
    return { show: false };
  }
  if (!chatSocketEnabled) {
    return { show: false };
  }

  if (realtime.status === "idle") {
    return {
      show: true,
      tone: "caution",
      title: "Initialisation de la session…",
    };
  }

  if (realtime.status === "connecting" || realtime.status === "reconnecting") {
    return {
      show: true,
      tone: "caution",
      title: "Reconnexion en cours…",
      body: REFRESH_FOOTER,
    };
  }

  if (realtime.status === "degraded") {
    return {
      show: true,
      tone: "caution",
      title: "Synchronisation ralentie. Les messages peuvent arriver avec un petit délai.",
    };
  }

  if (realtime.status === "failed" || !realtime.connected) {
    return {
      show: true,
      tone: "warning",
      title: "Connexion directe indisponible",
      body:
        "Si les messages restent figés, réessayez en rouvrant l’écran ou vérifiez le réseau. Les commandes côté serveur enregistrent quand même les envois.",
    };
  }

  return { show: false };
}

export default function CompanyChatScreen() {
  const { topInset, bottomInset, horizontalPadding } = useAppViewport();
  const t = useResponsiveTokens();
  const params = useLocalSearchParams<{ threadId?: string }>();
  const { activeContext } = useSession();
  const date = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const dashboardQuery = useCompanyDashboardQuery(new Date().toISOString().slice(0, 10));
  const messagesQuery = useCompanyChatMessages(date);
  const sendMutation = useCompanySendChatMessage();
  const realtime = useCompanyRealtimeStatus();
  const { invalidate } = useCompanyRealtimeInvalidation();
  const companyRealtimeEnabled = isFeatureEnabled("company_realtime_enabled");
  const chatSocketEnabled = isFeatureEnabled("company_mobile_chat_socket_enabled");
  const [input, setInput] = useState("");
  const [localMessages, setLocalMessages] = useState<SharedChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastFailedContent, setLastFailedContent] = useState<string | null>(null);
  const [typing, setTyping] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [pdfPreview, setPdfPreview] = useState<{ url: string; filename: string | null } | null>(null);
  const previousRealtimeStatus = useRef<string | null>(null);

  const remoteMessages = useMemo(() => {
    const pages = messagesQuery.data?.pages ?? [];
    return pages.flatMap((page) => page.messages);
  }, [messagesQuery.data?.pages]);

  const merged = useMemo(() => {
    const all = [...remoteMessages, ...localMessages];
    const dedup = new Map<string, SharedChatMessage>();
    all.forEach((message) => {
      dedup.set(String(message.id), message);
    });
    return [...dedup.values()].sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
  }, [localMessages, remoteMessages]);

  const { markAsRead, lastReadAt: companyLastRead, isLoading: companyLastReadMetadataLoading } =
    useCompanyChatUnread();
  const [listAnchorKey, setListAnchorKey] = useState(0);
  const wasLastReadQueryLoading = useRef(true);

  const listInitialScroll = useMemo(
    () => getChatListInitialScroll({ kind: "company", messages: merged, lastReadAt: companyLastRead }),
    [merged, companyLastRead]
  );

  useEffect(() => {
    if (wasLastReadQueryLoading.current && !companyLastReadMetadataLoading) {
      setListAnchorKey((k) => k + 1);
    }
    wasLastReadQueryLoading.current = companyLastReadMetadataLoading;
  }, [companyLastReadMetadataLoading]);

  useFocusEffect(
    useCallback(() => {
      setListAnchorKey((k) => k + 1);
      let cancelled = false;
      const task = InteractionManager.runAfterInteractions(() => {
        if (cancelled) return;
        const latest = merged[merged.length - 1]?.timestamp;
        if (latest) void markAsRead(latest);
      });
      return () => {
        cancelled = true;
        (task as { cancel?: () => void } | void)?.cancel?.();
      };
    }, [markAsRead, merged])
  );

  useEffect(() => {
    emitCompanyDispatchTelemetry(
      "company.dispatch.opened",
      {
        source: "company.chat.screen",
        context_type: "company",
        context_id: activeContext?.context_id ?? null,
      },
      { allowWhenDisabled: true }
    );
  }, [activeContext?.context_id]);

  useEffect(() => {
    if (!chatSocketEnabled) return;
    if (!activeContext || activeContext.context_type !== "company") return;
    return contextRealtimeRouter.subscribe(activeContext.context_id, (event) => {
      if (!event || typeof event !== "object") return;
      const eventType = normalizeCompanyEventType((event as { event_type?: unknown }).event_type);
      if (eventType !== "booking_message_sent" && eventType !== "team_chat_message") return;
      const incomingMessage = mapRealtimeChatMessage(event);
      if (!incomingMessage) return;
      setLocalMessages((previous) => {
        const exists = previous.some((entry) => String(entry.id) === String(incomingMessage.id));
        if (exists) return previous;
        return [...previous, incomingMessage];
      });
      invalidate("booking_message_sent");
    });
  }, [activeContext, chatSocketEnabled, invalidate]);

  useEffect(() => {
    if (!chatSocketEnabled) return;
    const previousStatus = previousRealtimeStatus.current;
    if (realtime.status === "healthy" && previousStatus && previousStatus !== "healthy") {
      void messagesQuery.refetch();
    }
    previousRealtimeStatus.current = realtime.status;
  }, [chatSocketEnabled, messagesQuery, realtime.status]);

  const sendMessage = async () => {
    const content = input.trim();
    if (!content) return;
    const localId = `pending-${Date.now()}`;
    setLocalMessages((previous) => [
      ...previous,
      {
        id: localId,
        content,
        senderRole: "COMPANY",
        senderName: "Moi",
        timestamp: new Date().toISOString(),
      },
    ]);
    setInput("");
    try {
      await sendMutation.mutateAsync({ content });
      setLocalMessages((previous) => previous.filter((entry) => entry.id !== localId));
      setError(null);
      setLastFailedContent(null);
    } catch (e) {
      setLocalMessages((previous) =>
        previous.map((entry) =>
          entry.id === localId
            ? {
                ...entry,
                id: `failed-${Date.now()}`,
                content: `${entry.content} (echec envoi)`,
              }
            : entry
        )
      );
      setLastFailedContent(content);
      setError(e instanceof Error ? e.message : "Echec envoi message.");
    }
  };

  const sendVoiceMessage = (uri: string) => {
    setLocalMessages((previous) => [
      ...previous,
      {
        id: `voice-${Date.now()}`,
        content: "Message vocal",
        type: "audio" as const,
        senderRole: "COMPANY",
        senderName: "Moi",
        timestamp: new Date().toISOString(),
        audioUrl: uri,
        mimeType: "audio/m4a",
      },
    ]);
  };

  const pickImageAttachment = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8,
    });
    if (result.canceled || !result.assets[0]?.uri) return;
    const asset = result.assets[0];
    const uri = asset.uri;
    setImagePreviewUrl(uri);
    setLocalMessages((previous) => [
      ...previous,
      {
        id: `image-${Date.now()}`,
        content: "Image jointe",
        type: "image",
        senderRole: "COMPANY",
        senderName: "Moi",
        timestamp: new Date().toISOString(),
        imageUrl: uri,
        mimeType: asset.mimeType ?? "image/jpeg",
      },
    ]);
  };

  const pickPdfAttachment = async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: ["application/pdf"],
      copyToCacheDirectory: true,
      multiple: false,
    });
    if (result.canceled || !result.assets[0]?.uri) return;
    const asset = result.assets[0];
    setPdfPreview({ url: asset.uri, filename: asset.name ?? null });
    setLocalMessages((previous) => [
      ...previous,
      {
        id: `pdf-${Date.now()}`,
        content: asset.name ?? "Document PDF",
        type: "pdf",
        senderRole: "COMPANY",
        senderName: "Moi",
        timestamp: new Date().toISOString(),
        pdfUrl: asset.uri,
        pdfFilename: asset.name ?? null,
        mimeType: asset.mimeType ?? "application/pdf",
      },
    ]);
  };

  const keyboardOffset = Platform.OS === "ios" ? topInset : 0;

  const dispatchMetrics = useMemo(() => {
    const d = dashboardQuery.data;
    if (!d && dashboardQuery.isLoading) {
      return "Journée en cours: chargement des indicateurs…";
    }
    if (!d) {
      return "Journée en cours: — · —";
    }
    const delayed = d.delayed_bookings_metrics_available === false ? "—" : (d.delayed_bookings ?? "—");
    const opps = d.opportunities_metrics_available === false ? "—" : (d.opportunities ?? "—");
    return `Journée en cours · retards: ${delayed} · opportunités: ${opps}`;
  }, [dashboardQuery.data, dashboardQuery.isLoading]);

  const connectionBanner = getCompanyChatConnectionBanner(
    realtime,
    companyRealtimeEnabled,
    chatSocketEnabled
  );

  return (
    <PermissionGuard permission="company:dashboard:read">
      <Screen
        scroll={false}
        keyboardAware={Platform.OS !== "web"}
        keyboardVerticalOffset={keyboardOffset}
        withHorizontalPadding={false}
        safeBottom
      >
        <View
          style={{
            flex: 1,
            paddingTop: t.spacingMd,
            paddingHorizontal: horizontalPadding,
            paddingBottom: Math.max(t.spacingMd, bottomInset),
            gap: t.spacingXs,
            position: "relative",
          }}
        >
          <View>
            <AppText variant="sectionTitle" style={styles.headerTitle}>
              Chat dispatch
            </AppText>
            <AppText variant="bodyMuted" style={styles.metricsLine}>
              {dispatchMetrics}
            </AppText>
            {params.threadId ? (
              <AppText variant="caption" style={styles.threadLine}>
                Fil lié: {params.threadId}
              </AppText>
            ) : null}
          </View>
          {connectionBanner.show ? (
            <View
              style={[
                styles.connectionBanner,
                connectionBanner.tone === "warning" ? styles.connectionBannerWarning : styles.connectionBannerCaution,
              ]}
            >
              <AppText
                variant="label"
                style={
                  connectionBanner.tone === "warning" ? styles.connectionTitleWarning : styles.connectionTitleCaution
                }
              >
                {connectionBanner.title}
              </AppText>
              {connectionBanner.body ? (
                <AppText
                  variant="body"
                  style={
                    connectionBanner.tone === "warning" ? styles.connectionBodyWarning : styles.connectionBodyCaution
                  }
                >
                  {connectionBanner.body}
                </AppText>
              ) : null}
            </View>
          ) : null}
          <View style={{ flex: 1, minHeight: 0 }}>
            <ChatList
              messages={merged}
              loading={messagesQuery.isLoading}
              onLoadMore={messagesQuery.hasNextPage ? () => void messagesQuery.fetchNextPage() : undefined}
              loadingMore={messagesQuery.isFetchingNextPage}
              loadMoreDisabled={!messagesQuery.hasNextPage || messagesQuery.isFetchingNextPage}
              emptyLabel="Aucun message dispatch pour le moment."
              onOpenImage={(url) => setImagePreviewUrl(url)}
              onOpenPdf={(url) => {
                const source = merged.find((message) => message.pdfUrl === url);
                setPdfPreview({ url, filename: source?.pdfFilename ?? null });
              }}
              listAnchorKey={listAnchorKey}
              initialScroll={listInitialScroll}
            />
          </View>
          <TypingIndicator visible={typing || sendMutation.isPending} />
          <ChatComposer
            value={input}
            onChangeText={setInput}
            onTypingStateChange={setTyping}
            onPickImage={() => void pickImageAttachment()}
            onPickPdf={() => void pickPdfAttachment()}
            onSend={() => void sendMessage()}
            onVoiceMessage={sendVoiceMessage}
            sendLabel={sendMutation.isPending ? "Envoi..." : "Envoyer"}
          />
          {lastFailedContent ? (
            <Pressable onPress={() => setInput(lastFailedContent)} accessibilityRole="button">
              <AppText variant="error">
                Dernier envoi en echec. Touchez pour pre-remplir et renvoyer.
              </AppText>
            </Pressable>
          ) : null}
          {messagesQuery.error ? (
            <AppText variant="error">
              {messagesQuery.error instanceof Error ? messagesQuery.error.message : "Chargement chat impossible."}
            </AppText>
          ) : null}
          {error ? <AppText variant="error">{error}</AppText> : null}
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
        </View>
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  headerTitle: {
    color: "#111827",
  },
  metricsLine: {
    marginTop: 4,
  },
  threadLine: {
    marginTop: 4,
  },
  connectionBanner: {
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: StyleSheet.hairlineWidth,
  },
  connectionBannerCaution: {
    backgroundColor: "#fffbeb",
    borderColor: "#fde68a",
  },
  connectionBannerWarning: {
    backgroundColor: "#fef2f2",
    borderColor: "#fecaca",
  },
  connectionTitleCaution: {
    color: "#92400e",
  },
  connectionTitleWarning: {
    color: "#991b1b",
  },
  connectionBodyCaution: {
    color: "#b45309",
    marginTop: 4,
    lineHeight: 18,
  },
  connectionBodyWarning: {
    color: "#b91c1c",
    marginTop: 4,
    lineHeight: 18,
  },
});
