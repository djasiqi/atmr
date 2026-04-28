import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { InteractionManager, KeyboardAvoidingView, Platform, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import { io, Socket } from "socket.io-client";
import { useFocusEffect } from "@react-navigation/native";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import { useDriverMissionsQuery } from "../../../src/features/driver/hooks";
import {
  useDriverChatMessages,
  useUnreadMessages,
} from "../../../src/features/driver/chatHooks";
import { getDriverMessages, type DriverChatMessage } from "../../../src/features/driver/api";
import { ChatComposer, ChatList, getChatListInitialScroll } from "../../../src/features/chat";

function toCompanyId(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export default function DriverChatScreen() {
  const insets = useSafeAreaInsets();
  const { activeContext } = useSession();
  const missionsQuery = useDriverMissionsQuery();
  const [input, setInput] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);
  const [liveMessages, setLiveMessages] = useState<DriverChatMessage[]>([]);
  const [socketConnected, setSocketConnected] = useState(false);
  const [socketInstance, setSocketInstance] = useState<Socket | null>(null);
  const [typingNames, setTypingNames] = useState<string[]>([]);
  const [typingUntilMs, setTypingUntilMs] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);
  const contextId = activeContext?.context_id ?? null;
  const companyId = useMemo(() => {
    const fromContext = toCompanyId(activeContext?.organization_id);
    if (fromContext != null) return fromContext;
    const missionCompany = (missionsQuery.data ?? [])
      .map((mission) => toCompanyId((mission as Record<string, unknown>).company_id))
      .find((value) => value != null);
    return missionCompany ?? null;
  }, [activeContext?.organization_id, missionsQuery.data]);
  const messagesQuery = useDriverChatMessages(companyId, contextId);
  const unread = useUnreadMessages(companyId, contextId, messagesQuery.data);
  const [listAnchorKey, setListAnchorKey] = useState(0);

  useEffect(() => {
    const socketUrl = process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL;
    if (!socketUrl || !contextId) return;
    const socket = io(socketUrl, {
      transports: ["websocket"],
      reconnection: true,
      query: { context_id: contextId, surface: "driver" },
    });
    setSocketInstance(socket);
    socket.on("connect", () => setSocketConnected(true));
    socket.on("disconnect", () => setSocketConnected(false));
    socket.on("team_chat_message", (event: unknown) => {
      if (!event || typeof event !== "object") return;
      const payload = event as Record<string, unknown>;
      const timestamp =
        typeof payload.timestamp === "string" && payload.timestamp.length > 0
          ? payload.timestamp
          : new Date().toISOString();
      const nextMessage: DriverChatMessage = {
        id:
          typeof payload.id === "number" || typeof payload.id === "string"
            ? payload.id
            : `${timestamp}-${Math.random().toString(36).slice(2, 8)}`,
        content: typeof payload.content === "string" ? payload.content : "",
        sender_role: typeof payload.sender_role === "string" ? payload.sender_role : undefined,
        sender_name: typeof payload.sender_name === "string" ? payload.sender_name : null,
        timestamp,
        image_url: typeof payload.image_url === "string" ? payload.image_url : null,
        pdf_url: typeof payload.pdf_url === "string" ? payload.pdf_url : null,
        pdf_filename: typeof payload.pdf_filename === "string" ? payload.pdf_filename : null,
        audio_url: typeof payload.audio_url === "string" ? payload.audio_url : null,
      };
      setLiveMessages((previous) => [...previous, nextMessage]);
    });
    socket.on("team_chat_typing", (event: unknown) => {
      if (!event || typeof event !== "object") return;
      const payload = event as Record<string, unknown>;
      const sender =
        typeof payload.sender_name === "string" && payload.sender_name.length > 0
          ? payload.sender_name
          : "Equipe";
      setTypingNames((previous) => {
        if (previous.includes(sender)) return previous;
        return [...previous, sender];
      });
      setTypingUntilMs(Date.now() + 3000);
    });
    return () => {
      socket.removeAllListeners();
      socket.disconnect();
      setSocketInstance(null);
      setSocketConnected(false);
    };
  }, [contextId]);

  useEffect(() => {
    if (typingUntilMs === 0) return;
    const timer = setTimeout(() => {
      setTypingNames([]);
      setTypingUntilMs(0);
    }, Math.max(300, typingUntilMs - Date.now()));
    return () => clearTimeout(timer);
  }, [typingUntilMs]);

  const mergedMessages = useMemo(() => {
    const all = [...(messagesQuery.data ?? []), ...liveMessages];
    const dedup = new Map<string, DriverChatMessage>();
    all.forEach((message) => {
      dedup.set(String(message.id), message);
    });
    return [...dedup.values()].sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
  }, [liveMessages, messagesQuery.data]);

  const listInitialScroll = useMemo(
    () =>
      getChatListInitialScroll({
        kind: "driver",
        messages: mergedMessages,
        lastReadAt: unread.lastReadAt,
      }),
    [mergedMessages, unread.lastReadAt]
  );

  const wasLastReadQueryLoading = useRef(true);
  useEffect(() => {
    if (wasLastReadQueryLoading.current && !unread.isLoadingLastRead) {
      setListAnchorKey((k) => k + 1);
    }
    wasLastReadQueryLoading.current = unread.isLoadingLastRead;
  }, [unread.isLoadingLastRead]);

  useFocusEffect(
    useCallback(() => {
      setListAnchorKey((k) => k + 1);
      let cancelled = false;
      const task = InteractionManager.runAfterInteractions(() => {
        if (cancelled) return;
        void unread.markRead(mergedMessages[mergedMessages.length - 1]?.timestamp);
      });
      return () => {
        cancelled = true;
        (task as { cancel?: () => void } | void)?.cancel?.();
      };
    }, [mergedMessages, unread])
  );

  const sendMessage = () => {
    const content = input.trim();
    if (!content) return;
    if (!socketInstance || !socketConnected) {
      setSendError("Socket chat indisponible. Reessayez dans quelques secondes.");
      return;
    }
    socketInstance.emit("team_chat_message", {
      content,
      receiver_id: null,
    });
    setLiveMessages((previous) => [
      ...previous,
      {
        id: `local-${Date.now()}`,
        content,
        sender_role: "DRIVER",
        sender_name: "Moi",
        timestamp: new Date().toISOString(),
      },
    ]);
    setInput("");
    setSendError(null);
  };

  const sendVoiceMessage = (uri: string) => {
    if (!socketInstance || !socketConnected) {
      setSendError("Socket chat indisponible. Reessayez dans quelques secondes.");
      return;
    }
    socketInstance.emit("team_chat_message", {
      content: "Message vocal",
      audio_url: uri,
    });
    setLiveMessages((previous) => [
      ...previous,
      {
        id: `local-voice-${Date.now()}`,
        content: "Message vocal",
        sender_role: "DRIVER",
        sender_name: "Moi",
        timestamp: new Date().toISOString(),
        audio_url: uri,
      },
    ]);
    setSendError(null);
  };

  const emitTyping = useCallback(() => {
    if (!socketInstance || !socketConnected) return;
    socketInstance.emit("team_chat_typing", { surface: "driver" });
  }, [socketConnected, socketInstance]);

  const loadMore = async () => {
    if (!companyId || !contextId || loadingMore) return;
    const oldest = mergedMessages[0]?.timestamp;
    if (!oldest) return;
    setLoadingMore(true);
    try {
      const next = await getDriverMessages(companyId, { before: oldest, limit: 40 });
      setLiveMessages((previous) => [...next, ...previous]);
    } catch (error) {
      setSendError(error instanceof Error ? error.message : "Chargement de messages plus anciens impossible.");
    } finally {
      setLoadingMore(false);
    }
  };

  const pickImageAttachment = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setSendError("Permission galerie refusee.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.7,
      allowsEditing: false,
    });
    if (result.canceled || !result.assets[0]) return;
    const asset = result.assets[0];
    if (!socketInstance || !socketConnected) {
      setSendError("Socket chat indisponible. Piece jointe non envoyee.");
      return;
    }
    socketInstance.emit("team_chat_message", {
      content: input.trim(),
      image_url: asset.uri,
    });
    setLiveMessages((previous) => [
      ...previous,
      {
        id: `local-image-${Date.now()}`,
        content: input.trim(),
        sender_role: "DRIVER",
        sender_name: "Moi",
        timestamp: new Date().toISOString(),
        image_url: asset.uri,
      },
    ]);
    setInput("");
    setSendError(null);
  };

  const pickDocumentAttachment = async () => {
    const result = await DocumentPicker.getDocumentAsync({ multiple: false, copyToCacheDirectory: true });
    if (result.canceled || !result.assets[0]) return;
    const asset = result.assets[0];
    if (!socketInstance || !socketConnected) {
      setSendError("Socket chat indisponible. Piece jointe non envoyee.");
      return;
    }
    socketInstance.emit("team_chat_message", {
      content: input.trim(),
      pdf_url: asset.uri,
      pdf_filename: asset.name ?? "document.pdf",
    });
    setLiveMessages((previous) => [
      ...previous,
      {
        id: `local-doc-${Date.now()}`,
        content: input.trim(),
        sender_role: "DRIVER",
        sender_name: "Moi",
        timestamp: new Date().toISOString(),
        pdf_url: asset.uri,
        pdf_filename: asset.name ?? "document.pdf",
      },
    ]);
    setInput("");
    setSendError(null);
  };

  const keyboardBehavior =
    Platform.OS === "ios" ? "padding" : Platform.OS === "android" ? "height" : undefined;
  const keyboardOffset = Platform.OS === "ios" ? insets.top : 0;

  return (
    <DriverContextGuard>
      <PermissionGuard permission="chat:read">
        <KeyboardAvoidingView
          style={{ flex: 1 }}
          behavior={keyboardBehavior}
          keyboardVerticalOffset={keyboardOffset}
          enabled={Platform.OS !== "web"}
        >
          <View
            style={{
              flex: 1,
              paddingTop: 24,
              paddingHorizontal: 24,
              paddingBottom: Math.max(24, insets.bottom),
              gap: 10,
              position: "relative",
            }}
          >
            <Text style={{ fontSize: 22, fontWeight: "700" }}>Chat chauffeur</Text>
            <Text style={{ color: "#666" }}>
              Canal: {socketConnected ? "socket connecte" : "socket deconnecte"} | Non lus:{" "}
              {unread.unreadCount}
            </Text>
            <View style={{ flex: 1, minHeight: 0 }}>
              <ChatList
                loading={messagesQuery.isLoading}
                messages={mergedMessages.map((message) => ({
                  id: message.id,
                  content: message.content,
                  senderName: message.sender_name,
                  senderRole: message.sender_role,
                  timestamp: message.timestamp,
                  imageUrl: message.image_url,
                  pdfUrl: message.pdf_url,
                  pdfFilename: message.pdf_filename,
                  audioUrl: message.audio_url,
                }))}
                onLoadMore={() => void loadMore()}
                loadingMore={loadingMore}
                loadMoreDisabled={loadingMore || mergedMessages.length === 0}
                listAnchorKey={listAnchorKey}
                initialScroll={listInitialScroll}
              />
            </View>
            <ChatComposer
              value={input}
              onChangeText={(value) => {
                setInput(value);
                emitTyping();
              }}
              onPickImage={() => void pickImageAttachment()}
              onPickPdf={() => void pickDocumentAttachment()}
              onSend={sendMessage}
              onVoiceMessage={sendVoiceMessage}
            />
            {messagesQuery.error ? (
              <Text style={{ color: "#B00020" }}>
                {messagesQuery.error instanceof Error
                  ? messagesQuery.error.message
                  : "Impossible de charger les messages."}
              </Text>
            ) : null}
            {sendError ? <Text style={{ color: "#B00020" }}>{sendError}</Text> : null}
            {typingNames.length > 0 ? (
              <Text style={{ color: "#666" }}>
                {typingNames.join(", ")} {typingNames.length > 1 ? "sont en train d'ecrire..." : "est en train d'ecrire..."}
              </Text>
            ) : null}
          </View>
        </KeyboardAvoidingView>
      </PermissionGuard>
    </DriverContextGuard>
  );
}
