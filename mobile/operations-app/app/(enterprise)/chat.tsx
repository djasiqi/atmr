// app/(enterprise)/chat.tsx
// ✅ Version améliorée WhatsApp-like avec les mêmes composants que le chat chauffeur

import React, {
  useEffect,
  useState,
  useRef,
  useCallback,
  useMemo,
} from "react";
import {
  View,
  Text,
  FlatList,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Keyboard,
  Platform,
  NativeSyntheticEvent,
  NativeScrollEvent,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from "react-native";
import { useFocusEffect } from "@react-navigation/native";
import { useBottomTabBarHeight } from "@react-navigation/bottom-tabs";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";

import { useAuth } from "@/hooks/useAuth";
import { useEnterpriseSocket } from "@/hooks/useEnterpriseSocket";
import { useEnterpriseNotifications } from "@/hooks/useEnterpriseNotifications";
import { useUnreadMessages } from "@/hooks/useUnreadMessages";
import {
  getDispatchMessages,
  sendDispatchMessage,
} from "@/services/enterpriseDispatch";
import { DispatchMessage } from "@/types/enterpriseDispatch";
import { Message } from "@/services/api";
import api from "@/services/api";

import MessageBubble from "@/components/chat/MessageBubble";
import DateSeparator from "@/components/chat/DateSeparator";
import ScrollToBottomButton from "@/components/chat/ScrollToBottomButton";
import TypingIndicator from "@/components/chat/TypingIndicator";
import AttachmentSheet from "@/components/chat/AttachmentSheet";
import ImagePreviewModal from "@/components/chat/ImagePreviewModal";
import PdfPreviewModal from "@/components/chat/PdfPreviewModal";
import { chatStyles } from "@/styles/chatStyles";
import * as ImagePicker from "expo-image-picker";

// ✅ Import conditionnel DocumentPicker
let DocumentPicker: typeof import("expo-document-picker") | null = null;
try {
  if (Platform.OS !== "web") {
    DocumentPicker = require("expo-document-picker");
  }
} catch {
  DocumentPicker = null;
}

// --- Constantes layout ---
const SCROLL_TOLERANCE = 40;
const INPUT_ESTIMATED_HEIGHT = 64;
const CHAT_FETCH_LIMIT = 20;

// Adapter DispatchMessage vers Message pour compatibilité avec MessageBubble
const adaptDispatchMessageToMessage = (msg: DispatchMessage | any): Message => ({
  id: msg.id,
  sender_id: msg.sender_id,
  sender_role: msg.sender_role || "COMPANY",
  sender_name: msg.sender_name,
  content: msg.content,
  timestamp: msg.created_at || msg.timestamp, // Convertir created_at en timestamp
  company_id: undefined,
  receiver_id: null,
  receiver_name: null,
  _localId: msg._localId || null,
  image: msg.image || null,
  image_url: msg.image_url || null,
  pdf: msg.pdf || null,
  pdf_url: msg.pdf_url || null,
  pdf_filename: msg.pdf_filename || null,
  pdf_size: msg.pdf_size || null,
});

// Type pour les items de la liste (message ou séparateur de date)
type ListItem =
  | { type: "message"; message: Message }
  | { type: "dateSeparator"; date: string };

export default function EnterpriseChatScreen() {
  const { enterpriseSession } = useAuth();

  // Activer les notifications push pour l'entreprise
  useEnterpriseNotifications();

  // Gérer les messages non lus
  const { markAsRead, countUnreadInMessages } = useUnreadMessages();

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTeamTyping, setIsTeamTyping] = useState(false);
  const [showAttachment, setShowAttachment] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [pdfPreview, setPdfPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Mesure réelle de la barre d'input
  const [inputContainerHeight, setInputContainerHeight] = useState(
    INPUT_ESTIMATED_HEIGHT
  );

  // État du clavier pour Android (gestion manuelle)
  const [keyboardHeight, setKeyboardHeight] = useState(0);

  // Refs scroll & état
  const flatListRef = useRef<FlatList<ListItem> | null>(null);
  const isMountedRef = useRef(true);
  const hasDoneInitialScrollRef = useRef(false);
  const isAtBottomRef = useRef(true);
  const previousContentHeightRef = useRef(0);
  const contentHeightRef = useRef(0);
  const layoutHeightRef = useRef(0);
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Pagination
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMoreMessages, setHasMoreMessages] = useState(true);
  const lastScrollOffsetRef = useRef(0);

  const insets = useSafeAreaInsets();
  const tabBarHeight = useBottomTabBarHeight();

  // =============== SOCKET ===============

  const socket = useEnterpriseSocket((msg: any) => {
    if (!isMountedRef.current) return;
    // Adapter le message reçu via socket (inclure tous les champs possibles)
    const adapted = adaptDispatchMessageToMessage({
      id: msg.id,
      sender_id: msg.sender_id,
      sender_role: msg.sender_role,
      sender_name: msg.sender_name,
      content: msg.content,
      created_at: msg.timestamp || msg.created_at,
      image_url: msg.image_url,
      pdf_url: msg.pdf_url,
      pdf_filename: msg.pdf_filename,
      pdf_size: msg.pdf_size,
      _localId: msg._localId,
    });

    setMessages((prev) => {
      const exists = prev.some(
        (m) =>
          (m.id && adapted.id && m.id === adapted.id) ||
          (m._localId && adapted._localId && m._localId === adapted._localId)
      );
      if (exists) {
        return prev;
      }
      const updated = [...prev, adapted];
      return updated.sort((a, b) => {
        const timeA = new Date(a.timestamp || 0).getTime();
        const timeB = new Date(b.timestamp || 0).getTime();
        return timeA - timeB;
      });
    });
  });

  // =============== TYPING INDICATOR ===============

  useEffect(() => {
    if (!socket) return;

    socket.on("typing_start", () => setIsTeamTyping(true));
    socket.on("typing_stop", () => setIsTeamTyping(false));

    return () => {
      socket.off("typing_start");
      socket.off("typing_stop");
    };
  }, [socket]);

  const typingTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleTyping = useCallback(
    (text: string) => {
      setInput(text);
      if (!socket) return;

      socket.emit("typing_start");
      if (typingTimeout.current) clearTimeout(typingTimeout.current);
      typingTimeout.current = setTimeout(() => {
        socket.emit("typing_stop");
      }, 900);
    },
    [socket]
  );

  // =============== IMAGE / PDF ENVOI ===============

  const handleSendImage = useCallback(
    async (imageUri: string) => {
      if (!socket || !enterpriseSession?.company?.id) {
        Alert.alert("Erreur", "Connexion non disponible. Veuillez réessayer.");
        return;
      }

      if (!socket.connected) {
        Alert.alert("Erreur", "Connexion au serveur perdue. Veuillez réessayer.");
        return;
      }

      setIsUploading(true);
      try {
        const formData = new FormData();
        formData.append("file", {
          uri: imageUri,
          type: "image/jpeg",
          name: "image.jpg",
        } as any);

        const uploadRes = await api.post("/messages/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        const { url } = uploadRes.data;
        if (!url) {
          throw new Error("URL de l'image non reçue du serveur");
        }

        socket.emit("team_chat_message", {
          content: "",
          image_url: url,
          receiver_id: null,
        });
      } catch (error: any) {
        console.error("[EnterpriseChat] Erreur upload image:", error);
        const errorMessage =
          error?.response?.data?.error ||
          error?.message ||
          "Erreur lors de l'envoi de l'image. Veuillez réessayer.";
        Alert.alert("Erreur", errorMessage);
      } finally {
        setIsUploading(false);
      }
    },
    [socket, enterpriseSession?.company?.id]
  );

  const handleSendPdf = useCallback(
    async (pdfUri: string, filename: string) => {
      if (!socket || !enterpriseSession?.company?.id) {
        Alert.alert("Erreur", "Connexion non disponible. Veuillez réessayer.");
        return;
      }

      if (!socket.connected) {
        Alert.alert("Erreur", "Connexion au serveur perdue. Veuillez réessayer.");
        return;
      }

      setIsUploading(true);
      try {
        const formData = new FormData();
        formData.append("file", {
          uri: pdfUri,
          type: "application/pdf",
          name: filename,
        } as any);

        const uploadRes = await api.post("/messages/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        const { url, filename: uploadedFilename, size_bytes } = uploadRes.data;
        if (!url) {
          throw new Error("URL du PDF non reçue du serveur");
        }

        socket.emit("team_chat_message", {
          content: "",
          pdf_url: url,
          pdf_filename: uploadedFilename,
          pdf_size: size_bytes,
          receiver_id: null,
        });
      } catch (error: any) {
        console.error("[EnterpriseChat] Erreur upload PDF:", error);
        const errorMessage =
          error?.response?.data?.error ||
          error?.message ||
          "Erreur lors de l'envoi du PDF. Veuillez réessayer.";
        Alert.alert("Erreur", errorMessage);
      } finally {
        setIsUploading(false);
      }
    },
    [socket, enterpriseSession?.company?.id]
  );

  // =============== ATTACHMENT HANDLERS ===============

  const handlePickCamera = useCallback(async () => {
    setShowAttachment(false);
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") return;

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.7,
    });

    if (!result.canceled && result.assets[0]) {
      await handleSendImage(result.assets[0].uri);
    }
  }, [handleSendImage]);

  const handlePickGallery = useCallback(async () => {
    setShowAttachment(false);
    // ✅ Demander les permissions pour accéder à la galerie
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission requise",
        "L'application a besoin d'accéder à votre galerie pour envoyer des photos."
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.7,
    });

    if (!result.canceled && result.assets[0]) {
      await handleSendImage(result.assets[0].uri);
    }
  }, [handleSendImage]);

  const handlePickDocument = useCallback(async () => {
    setShowAttachment(false);
    if (!DocumentPicker) {
      console.log("⚠️ DocumentPicker non dispo");
      return;
    }
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "application/pdf",
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets[0]) {
        await handleSendPdf(
          result.assets[0].uri,
          result.assets[0].name || "document.pdf"
        );
      }
    } catch (error) {
      console.log("Erreur sélection PDF:", error);
    }
  }, [handleSendPdf]);

  // =============== LOAD INITIAL MESSAGES ===============

  useEffect(() => {
    const loadInitialMessages = async () => {
      if (!enterpriseSession?.company?.id) return;

      try {
        console.log("📨 load initial messages company_id:", enterpriseSession.company.id);
        const fetched = await getDispatchMessages({
          limit: CHAT_FETCH_LIMIT,
        });

        if (!isMountedRef.current) return;

        // Convertir DispatchMessage en Message et trier
        const adapted = fetched.map(adaptDispatchMessageToMessage);
        const sorted = adapted.sort((a, b) => {
          const timeA = new Date(a.timestamp || 0).getTime();
          const timeB = new Date(b.timestamp || 0).getTime();
          return timeA - timeB; // Tri croissant : plus ancien en premier
        });

        setMessages(sorted);
        hasDoneInitialScrollRef.current = false;
        isAtBottomRef.current = true;
        setShowScrollButton(false);
        setHasMoreMessages(fetched.length >= CHAT_FETCH_LIMIT);

        // Compter les messages non lus parmi les messages chargés
        if (countUnreadInMessages) {
          countUnreadInMessages(sorted);
        }

        // Scroll vers le bas après un délai
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            setTimeout(() => {
              if (flatListRef.current && layoutHeightRef.current > 0) {
                const offset = contentHeightRef.current - layoutHeightRef.current;
                if (offset > 0) {
                  flatListRef.current.scrollToOffset({ offset, animated: false });
                } else {
                  flatListRef.current.scrollToEnd({ animated: false });
                }
                setTimeout(() => {
                  if (flatListRef.current && layoutHeightRef.current > 0) {
                    const offset2 = contentHeightRef.current - layoutHeightRef.current;
                    if (offset2 > 0) {
                      flatListRef.current.scrollToOffset({ offset: offset2, animated: false });
                    } else {
                      flatListRef.current.scrollToEnd({ animated: false });
                    }
                    isAtBottomRef.current = true;
                    setShowScrollButton(false);
                  }
                }, 150);
              }
            }, 100);
          });
        });
      } catch (e) {
        console.error("❌ Erreur chargement messages:", e);
      }
    };

    loadInitialMessages();
  }, [enterpriseSession?.company?.id]);

  // =============== LOAD MORE MESSAGES ===============

  const loadMoreMessages = useCallback(async () => {
    if (!enterpriseSession?.company?.id || isLoadingMore || !hasMoreMessages) return;

    const oldestMessage = messages[0];
    if (!oldestMessage?.timestamp) return;

    setIsLoadingMore(true);
    try {
      const older = await getDispatchMessages({
        before: oldestMessage.timestamp,
        limit: CHAT_FETCH_LIMIT,
      });

      if (!isMountedRef.current || older.length === 0) {
        setHasMoreMessages(false);
        setIsLoadingMore(false);
        return;
      }

      const adapted = older.map(adaptDispatchMessageToMessage);
      setMessages((prev) => {
        const existingIds = new Set(prev.map((m) => m.id).filter((id) => id != null));
        const newMessages = adapted.filter((m) => !m.id || !existingIds.has(m.id));
        const combined = [...prev, ...newMessages];
        return combined.sort((a, b) => {
          const timeA = new Date(a.timestamp || 0).getTime();
          const timeB = new Date(b.timestamp || 0).getTime();
          return timeA - timeB;
        });
      });

      setHasMoreMessages(older.length >= CHAT_FETCH_LIMIT);
    } catch (e) {
      console.error("❌ Erreur chargement messages supplémentaires:", e);
    } finally {
      setIsLoadingMore(false);
    }
  }, [enterpriseSession?.company?.id, isLoadingMore, hasMoreMessages, messages]);

  // =============== SCROLL TO BOTTOM ===============

  const scrollToBottom = useCallback((animated = true) => {
    if (!flatListRef.current) return;
    try {
      if (layoutHeightRef.current > 0 && contentHeightRef.current > 0) {
        const offset = contentHeightRef.current - layoutHeightRef.current;
        if (offset > 0) {
          flatListRef.current.scrollToOffset({ offset, animated });
        } else {
          flatListRef.current.scrollToEnd({ animated });
        }
      } else {
        flatListRef.current.scrollToEnd({ animated });
      }
      isAtBottomRef.current = true;
      setShowScrollButton(false);
    } catch (e) {
      console.log("[EnterpriseChat] scrollToBottom error:", e);
    }
  }, []);

  // =============== HANDLE SCROLL ===============

  const handleScroll = useCallback(
    (event: NativeSyntheticEvent<NativeScrollEvent>) => {
      const { contentOffset, contentSize, layoutMeasurement } = event.nativeEvent;

      const distanceFromBottom =
        contentSize.height - (contentOffset.y + layoutMeasurement.height);

      const isBottom = distanceFromBottom <= SCROLL_TOLERANCE;
      isAtBottomRef.current = isBottom;
      setShowScrollButton(!isBottom);

      // Pagination : détecter le scroll vers le haut
      const currentOffset = contentOffset.y;
      const isScrollingUp = currentOffset < lastScrollOffsetRef.current;
      const distanceFromTop = contentOffset.y;

      if (isScrollingUp && distanceFromTop < 200 && hasMoreMessages && !isLoadingMore) {
        loadMoreMessages();
      }

      lastScrollOffsetRef.current = currentOffset;
    },
    [hasMoreMessages, isLoadingMore, loadMoreMessages]
  );

  // =============== SEND MESSAGE ===============

  const sendMessage = useCallback(() => {
    const content = input.trim();
    if (!content) return;

    if (!socket) {
      Alert.alert("Erreur", "Connexion non disponible. Veuillez réessayer.");
      return;
    }

    if (!socket.connected) {
      Alert.alert("Erreur", "Connexion au serveur perdue. Veuillez réessayer.");
      return;
    }

    try {
      socket.emit("team_chat_message", {
        content,
        receiver_id: null,
      });

      setInput("");
      if (typingTimeout.current) {
        clearTimeout(typingTimeout.current);
        typingTimeout.current = null;
      }
      socket.emit("typing_stop");
    } catch (error: any) {
      console.error("[EnterpriseChat] Erreur envoi message:", error);
      Alert.alert(
        "Erreur",
        error?.message || "Erreur lors de l'envoi du message. Veuillez réessayer."
      );
    }
  }, [input, socket]);

  // =============== FOCUS SCREEN ===============

  useFocusEffect(
    useCallback(() => {
      // Marquer tous les messages comme lus quand l'écran est au focus
      if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        if (lastMessage?.id) {
          const messageId = typeof lastMessage.id === "string"
            ? parseInt(lastMessage.id, 10)
            : lastMessage.id;
          const messageTimestamp = lastMessage.timestamp
            ? new Date(lastMessage.timestamp).getTime()
            : Date.now();
          if (!isNaN(messageId)) {
            markAsRead(messageId, messageTimestamp);
          }
        }
      }

      if (messages.length > 0 && isAtBottomRef.current) {
        const t = setTimeout(() => {
          scrollToBottom(false);
          setTimeout(() => {
            scrollToBottom(false);
          }, 100);
        }, 100);
        return () => clearTimeout(t);
      }
      return () => { };
    }, [messages.length, scrollToBottom, markAsRead])
  );

  // =============== KEYBOARD LISTENERS (Android uniquement) ===============

  useEffect(() => {
    if (Platform.OS !== "android") return;

    const keyboardDidShowListener = Keyboard.addListener(
      "keyboardDidShow",
      (event) => {
        const newKeyboardHeight = event.endCoordinates.height;
        setKeyboardHeight(newKeyboardHeight);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            setTimeout(() => {
              scrollToBottom(true);
              setTimeout(() => {
                scrollToBottom(true);
              }, 100);
            }, 50);
          });
        });
      }
    );

    const keyboardDidHideListener = Keyboard.addListener(
      "keyboardDidHide",
      () => {
        setTimeout(() => {
          setKeyboardHeight(0);
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              setTimeout(() => {
                scrollToBottom(true);
                setTimeout(() => {
                  scrollToBottom(true);
                }, 100);
              }, 50);
            });
          });
        }, 50);
      }
    );

    return () => {
      keyboardDidShowListener.remove();
      keyboardDidHideListener.remove();
    };
  }, [scrollToBottom]);

  // =============== MOUNT / UNMOUNT ===============

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (typingTimeout.current) clearTimeout(typingTimeout.current);
    };
  }, []);

  // =============== FLATLIST PADDING & BOUTON ↓ ===============

  // paddingBottom = hauteur input + tab bar/clavier + safe area + marge verticale
  // Le dernier message doit être juste au-dessus de l'input
  // Sur Android avec clavier ouvert : clavier + input + marge verticale
  const flatListPaddingBottom = useMemo(() => {
    const safeBottom = insets.bottom;
    const messageSpacing = 4; // Marge verticale pour que le dernier message soit bien visible juste au-dessus de l'input

    if (Platform.OS === "android" && keyboardHeight > 0) {
      // Clavier ouvert : clavier + input + marge verticale
      // Le message doit être affiché au-dessus de l'input qui est au-dessus du clavier
      return keyboardHeight + inputContainerHeight + safeBottom + messageSpacing;
    }
    // Clavier fermé : padding pour l'input au-dessus de la tab bar
    return inputContainerHeight + tabBarHeight + safeBottom + messageSpacing;
  }, [inputContainerHeight, tabBarHeight, insets.bottom, keyboardHeight]);

  // offset du bouton ↓ = au-dessus de l'input + tab bar
  const scrollButtonBottom = useMemo(() => {
    const safeBottom = insets.bottom;
    return inputContainerHeight + tabBarHeight + safeBottom + 16;
  }, [inputContainerHeight, tabBarHeight, insets.bottom]);

  const contentContainerStyle = useMemo(
    () => [
      chatStyles.messagesList,
      messages.length === 0 && {
        flexGrow: 1,
        justifyContent: "center" as const,
      },
      { paddingBottom: flatListPaddingBottom },
    ],
    [messages.length, flatListPaddingBottom]
  );

  // =============== FORMAT DATA WITH DATE SEPARATORS ===============

  const listItemsWithDates = useMemo((): ListItem[] => {
    if (messages.length === 0) return [];

    const items: ListItem[] = [];
    let lastDate: string | null = null;

    for (const message of messages) {
      if (!message.timestamp) {
        items.push({ type: "message", message });
        continue;
      }

      const messageDate = new Date(message.timestamp);
      const dateKey = `${messageDate.getFullYear()}-${String(messageDate.getMonth() + 1).padStart(2, "0")}-${String(messageDate.getDate()).padStart(2, "0")}`;

      if (lastDate !== dateKey) {
        items.push({ type: "dateSeparator", date: messageDate.toISOString() });
        lastDate = dateKey;
      }
      items.push({ type: "message", message });
    }
    return items;
  }, [messages]);

  // =============== RENDER ===============

  const renderContent = () => (
    <View style={{ flex: 1 }}>
      <FlatList
        ref={flatListRef}
        data={listItemsWithDates}
        renderItem={({ item }) => {
          if (item.type === "dateSeparator") {
            return <DateSeparator date={item.date} />;
          }
          return (
            <MessageBubble
              message={item.message}
              currentUserId={enterpriseSession?.user?.id || null}
              onPressImage={setImagePreview}
              onPressPdf={setPdfPreview}
            />
          );
        }}
        keyExtractor={(item, index) => {
          if (item.type === "dateSeparator") {
            return `date-${item.date}-${index}`;
          }
          if (item.message?.id != null) {
            return `msg-${item.message.id}`;
          }
          const timestamp = item.message?.timestamp ? new Date(item.message.timestamp).getTime() : Date.now();
          return `msg-${index}-${timestamp}-${Math.random().toString(36).slice(2)}`;
        }}
        contentContainerStyle={contentContainerStyle}
        style={{ flex: 1 }}
        showsVerticalScrollIndicator
        onScroll={handleScroll}
        scrollEventThrottle={16}
        onLayout={(event) => {
          layoutHeightRef.current = event.nativeEvent.layout.height;
        }}
        ListHeaderComponent={
          isLoadingMore ? (
            <View style={{ padding: 16, alignItems: "center" }}>
              <Text style={{ color: "#5F7369", fontSize: 14 }}>Chargement...</Text>
            </View>
          ) : null
        }
        onContentSizeChange={(contentWidth, contentHeight) => {
          if (listItemsWithDates.length === 0) return;

          contentHeightRef.current = contentHeight;

          if (!hasDoneInitialScrollRef.current) {
            hasDoneInitialScrollRef.current = true;
            previousContentHeightRef.current = contentHeight;
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                setTimeout(() => {
                  if (flatListRef.current && layoutHeightRef.current > 0) {
                    const offset = contentHeight - layoutHeightRef.current;
                    if (offset > 0) {
                      flatListRef.current.scrollToOffset({ offset, animated: false });
                    } else {
                      flatListRef.current.scrollToEnd({ animated: false });
                    }
                    setTimeout(() => {
                      if (flatListRef.current && layoutHeightRef.current > 0) {
                        const offset2 = contentHeightRef.current - layoutHeightRef.current;
                        if (offset2 > 0) {
                          flatListRef.current.scrollToOffset({ offset: offset2, animated: false });
                        } else {
                          flatListRef.current.scrollToEnd({ animated: false });
                        }
                        isAtBottomRef.current = true;
                        setShowScrollButton(false);
                      }
                    }, 100);
                  }
                }, 100);
              });
            });
            return;
          }

          // Vérifier si c'est un changement de taille dû au padding ou à un nouveau message
          const heightDifference = contentHeight - previousContentHeightRef.current;
          previousContentHeightRef.current = contentHeight;

          // Si la différence est petite (< 100px), c'est probablement juste un changement de padding
          // On ne scroll pas dans ce cas pour éviter le scroll inversé
          if (Math.abs(heightDifference) < 100) {
            return;
          }

          // Si la différence est grande et qu'on est en bas, c'est un nouveau message
          // Scroller vers le bas pour montrer le nouveau message
          if (heightDifference > 0 && isAtBottomRef.current) {
            requestAnimationFrame(() => {
              scrollToBottom(true);
            });
          }
        }}
        onScrollBeginDrag={() => {
          // Sur Android, ne pas fermer le clavier automatiquement pour éviter les conflits
          // Le clavier peut se fermer automatiquement si nécessaire
          if (Platform.OS === "ios") {
            try {
              Keyboard.dismiss();
            } catch (e) {
              console.log("[EnterpriseChat] Keyboard dismiss error:", e);
            }
          }
        }}
        keyboardShouldPersistTaps="handled"
        ListEmptyComponent={() => (
          <View style={chatStyles.emptyContainer}>
            <Ionicons name="chatbubble-ellipses-outline" size={48} color="#5F7369" />
            <Text style={chatStyles.emptyText}>
              Aucun message pour le moment.{"\n"}
              Commencez la conversation avec votre équipe !
            </Text>
          </View>
        )}
      />

      {/* Indicateur "équipe écrit" */}
      {isTeamTyping && <TypingIndicator />}

      {/* Bouton ↓ flottant */}
      <ScrollToBottomButton
        visible={showScrollButton}
        onPress={() => scrollToBottom(true)}
        bottomOffset={scrollButtonBottom}
      />

      {/* Barre d'input - Position dynamique sur Android selon le clavier */}
      <View
        style={[
          chatStyles.inputContainer,
          Platform.OS === "android"
            ? {
              // Android : Position absolue pour suivre le clavier
              position: "absolute" as const,
              bottom:
                keyboardHeight > 0
                  ? keyboardHeight // Au-dessus du clavier quand ouvert
                  : tabBarHeight, // Au-dessus de la tab bar quand fermé
              left: 0,
              right: 0,
              paddingBottom: insets.bottom,
              // Éviter les re-renders qui font perdre le focus
              pointerEvents: "auto" as const,
            }
            : {
              // iOS : Dans le flux normal (géré par KeyboardAvoidingView)
              paddingBottom: insets.bottom,
              marginBottom: tabBarHeight,
            },
        ]}
        onLayout={(e) => {
          // Ne mesurer que si le clavier est fermé pour éviter les re-renders
          if (Platform.OS === "android" && keyboardHeight === 0) {
            setInputContainerHeight(e.nativeEvent.layout.height);
          } else if (Platform.OS !== "android") {
            setInputContainerHeight(e.nativeEvent.layout.height);
          }
        }}
      >
        <TouchableOpacity
          onPress={() => setShowAttachment(true)}
          style={{ marginRight: 8 }}
          disabled={isUploading}
        >
          {isUploading ? (
            <ActivityIndicator size="small" color="#0A7F59" />
          ) : (
            <Ionicons
              name="attach"
              size={22}
              color="#0A7F59"
              style={{ transform: [{ rotate: "45deg" }] }}
            />
          )}
        </TouchableOpacity>

        <TextInput
          value={input}
          onChangeText={handleTyping}
          placeholder="Écrire un message..."
          placeholderTextColor={chatStyles.inputPlaceholder.color}
          style={chatStyles.input}
          multiline={false}
          maxLength={1000}
          onSubmitEditing={sendMessage}
          returnKeyType="send"
          onFocus={() => {
            // Pas de scroll automatique au focus sur Android
            // Le clavier va s'ouvrir et le padding va s'ajuster automatiquement
            // On laisse le comportement natif gérer le scroll
          }}
        />
        <TouchableOpacity
          onPress={sendMessage}
          style={chatStyles.sendButton}
          disabled={isUploading || !input.trim()}
        >
          {isUploading ? (
            <ActivityIndicator size="small" color="#FFFFFF" />
          ) : (
            <Ionicons name="send" size={20} color="#FFFFFF" />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={chatStyles.container}>
      {Platform.OS === "android" ? (
        // Android : Gestion manuelle du clavier via listeners (pas de KeyboardAvoidingView)
        // Cela évite les marges supplémentaires générées par KeyboardAvoidingView
        renderContent()
      ) : (
        // iOS : KeyboardAvoidingView avec behavior="padding"
        <KeyboardAvoidingView
          style={{ flex: 1 }}
          behavior="padding"
          keyboardVerticalOffset={tabBarHeight}
        >
          {renderContent()}
        </KeyboardAvoidingView>
      )}

      {/* -------- ATTACHMENT SHEET -------- */}
      <AttachmentSheet
        visible={showAttachment}
        onClose={() => setShowAttachment(false)}
        onPickCamera={handlePickCamera}
        onPickGallery={handlePickGallery}
        onPickDocument={handlePickDocument}
      />

      {/* -------- IMAGE PREVIEW -------- */}
      <ImagePreviewModal
        visible={imagePreview !== null}
        uri={imagePreview}
        onClose={() => setImagePreview(null)}
      />

      {/* -------- PDF PREVIEW -------- */}
      <PdfPreviewModal
        visible={pdfPreview !== null}
        pdfUrl={pdfPreview}
        onClose={() => setPdfPreview(null)}
      />
    </View>
  );
}
