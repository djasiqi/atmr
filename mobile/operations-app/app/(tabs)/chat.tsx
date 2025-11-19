// app/(tabs)/chat.tsx
// ✅ Version simplifiée & stable – scroll WhatsApp-like + tab bar + clavier OK

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
} from "react-native";
import { useFocusEffect } from "@react-navigation/native";
import { useBottomTabBarHeight } from "@react-navigation/bottom-tabs";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";

import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";
import api, { Message } from "@/services/api";

import ChatHeader from "@/components/dashboard/ChatHeader";
import { chatStyles } from "@/styles/chatStyles";
import ScrollToBottomButton from "@/components/chat/ScrollToBottomButton";
import MessageBubble from "@/components/chat/MessageBubble";
import TypingIndicator from "@/components/chat/TypingIndicator";
import AttachmentSheet from "@/components/chat/AttachmentSheet";
import ImagePreviewModal from "@/components/chat/ImagePreviewModal";
import PdfPreviewModal from "@/components/chat/PdfPreviewModal";
import DateSeparator from "@/components/chat/DateSeparator";
import * as ImagePicker from "expo-image-picker";

// ✅ Import conditionnel DocumentPicker (bare / dev / prod)
let DocumentPicker: typeof import("expo-document-picker") | null = null;
try {
  if (Platform.OS !== "web") {
    DocumentPicker = require("expo-document-picker");
  }
} catch {
  DocumentPicker = null;
}

// --- Constantes layout ---
const SCROLL_TOLERANCE = 40; // marge pour considérer "en bas"
const INPUT_ESTIMATED_HEIGHT = 64; // estimation initiale, sera mesurée

export default function ChatScreen() {
  const { driver } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTeamTyping, setIsTeamTyping] = useState(false);
  const [showAttachment, setShowAttachment] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [pdfPreview, setPdfPreview] = useState<string | null>(null);

  // Mesure réelle de la barre d'input
  const [inputContainerHeight, setInputContainerHeight] = useState(
    INPUT_ESTIMATED_HEIGHT
  );

  // État du clavier pour Android (gestion manuelle)
  const [keyboardHeight, setKeyboardHeight] = useState(0);

  // Type pour les items de la liste (message ou séparateur de date)
  type ListItem =
    | { type: "message"; message: Message }
    | { type: "dateSeparator"; date: string };

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

  const socket = useSocket(undefined, (msg: Message) => {
    if (!isMountedRef.current) return;
    // Vérifier que le message n'existe pas déjà pour éviter les doublons
    setMessages((prev) => {
      // Vérifier si le message existe déjà (par ID ou par _localId)
      const exists = prev.some(
        (m) =>
          (m.id && msg.id && m.id === msg.id) ||
          (m._localId && msg._localId && m._localId === msg._localId)
      );
      if (exists) {
        return prev; // Ne pas ajouter si déjà présent
      }
      // Ajouter le nouveau message et trier par timestamp (plus ancien en premier)
      const updated = [...prev, msg];
      return updated.sort((a, b) => {
        const timeA = new Date(a.timestamp || 0).getTime();
        const timeB = new Date(b.timestamp || 0).getTime();
        return timeA - timeB; // Tri croissant : plus ancien en premier
      });
    });
    // 👉 pas de scroll direct ici : on laisse onContentSizeChange gérer
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

  // =============== SCROLL TO BOTTOM ===============

  const scrollToBottom = useCallback(
    (animated = true) => {
      if (!flatListRef.current) return;

      try {
        // Calculer l'offset nécessaire pour scroller jusqu'en bas
        // offset = contentHeight - layoutHeight (pour être tout en bas)
        const contentHeight = contentHeightRef.current;
        const layoutHeight = layoutHeightRef.current;

        if (contentHeight > layoutHeight) {
          // Utiliser scrollToOffset avec l'offset calculé pour garantir qu'on va jusqu'au bout
          const offset = contentHeight - layoutHeight;
          flatListRef.current.scrollToOffset({ offset, animated });
        } else {
          // Si le contenu est plus petit que le layout, utiliser scrollToEnd
          flatListRef.current.scrollToEnd({ animated });
        }

        isAtBottomRef.current = true;
        setShowScrollButton(false);
      } catch (e) {
        console.log("[ChatScreen] scrollToBottom error:", e);
        // Fallback : essayer avec scrollToEnd
        try {
          flatListRef.current.scrollToEnd({ animated });
        } catch (e2) {
          console.log("[ChatScreen] scrollToEnd fallback error:", e2);
        }
      }
    },
    []
  );

  // =============== LOAD MORE MESSAGES (pagination) ===============

  const loadMoreMessages = useCallback(async () => {
    if (!driver?.company_id || isLoadingMore || !hasMoreMessages) {
      return;
    }

    // Trouver le message le plus ancien dans la liste actuelle
    const oldestMessage = messages[messages.length - 1];
    if (!oldestMessage || !oldestMessage.timestamp) {
      setHasMoreMessages(false);
      return;
    }

    setIsLoadingMore(true);
    try {
      console.log("📨 load more messages before:", oldestMessage.timestamp);
      // Charger les messages plus anciens que le timestamp du message le plus ancien
      const res = await api.get(`/messages/${driver.company_id}`, {
        params: { limit: 20, before: oldestMessage.timestamp },
      });
      const loaded = res.data as Message[];

      if (!isMountedRef.current || loaded.length === 0) {
        setHasMoreMessages(false);
        setIsLoadingMore(false);
        return;
      }

      // L'API retourne déjà les messages triés du plus ancien au plus récent
      // Ajouter les nouveaux messages et trier par timestamp pour garantir l'ordre chronologique
      // Filtrer les doublons en vérifiant les IDs
      setMessages((prev) => {
        const existingIds = new Set(prev.map((m) => m.id).filter((id) => id != null));
        const newMessages = loaded.filter((m) => !m.id || !existingIds.has(m.id));
        // Combiner et trier par timestamp (plus ancien en premier)
        const combined = [...prev, ...newMessages];
        return combined.sort((a, b) => {
          const timeA = new Date(a.timestamp || 0).getTime();
          const timeB = new Date(b.timestamp || 0).getTime();
          return timeA - timeB; // Tri croissant : plus ancien en premier
        });
      });

      // Si on a moins de 20 messages, il n'y a plus de messages à charger
      setHasMoreMessages(loaded.length >= 20);
    } catch (e) {
      console.error("❌ Erreur chargement messages supplémentaires:", e);
    } finally {
      setIsLoadingMore(false);
    }
  }, [driver?.company_id, isLoadingMore, hasMoreMessages, messages]);

  // =============== HANDLE SCROLL ===============

  const handleScroll = useCallback(
    (event: NativeSyntheticEvent<NativeScrollEvent>) => {
      const { contentOffset, contentSize, layoutMeasurement } =
        event.nativeEvent;

      const distanceFromBottom =
        contentSize.height - (contentOffset.y + layoutMeasurement.height);

      const isBottom = distanceFromBottom <= SCROLL_TOLERANCE;

      isAtBottomRef.current = isBottom;
      setShowScrollButton(!isBottom);

      // Pagination : détecter le scroll vers le haut pour charger les messages plus anciens
      const currentOffset = contentOffset.y;
      const isScrollingUp = currentOffset < lastScrollOffsetRef.current;
      const distanceFromTop = contentOffset.y;

      // Si on scroll vers le haut et qu'on est proche du début (< 200px), charger plus de messages
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
    if (!content || !socket) return;

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
  }, [input, socket]);

  // =============== IMAGE / PDF ENVOI ===============

  const handleSendImage = useCallback(
    async (imageUri: string) => {
      if (!socket || !driver?.company_id) return;

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
        socket.emit("team_chat_message", {
          content: "",
          image_url: url,
          receiver_id: null,
        });
      } catch (error) {
        console.log("[ChatScreen] Erreur upload image:", error);
      }
    },
    [socket, driver?.company_id]
  );

  const handleSendPdf = useCallback(
    async (pdfUri: string, filename: string) => {
      if (!socket || !driver?.company_id) return;

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
        socket.emit("team_chat_message", {
          content: "",
          pdf_url: url,
          pdf_filename: uploadedFilename,
          pdf_size: size_bytes,
          receiver_id: null,
        });
      } catch (error) {
        console.log("[ChatScreen] Erreur upload PDF:", error);
      }
    },
    [socket, driver?.company_id]
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
      console.log("⚠️ DocumentPicker non dispo (Expo Go / rebuild natif)");
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

  // =============== LOAD INITIAL MESSAGES (derniers messages uniquement) ===============

  useEffect(() => {
    const loadInitialMessages = async () => {
      if (!driver?.company_id) return;

      try {
        console.log("📨 load initial messages company_id:", driver.company_id);
        // Charger seulement les 20 derniers messages
        const res = await api.get(`/messages/${driver.company_id}`, {
          params: { limit: 20 },
        });
        const loaded = res.data as Message[];

        if (!isMountedRef.current) return;

        // L'API retourne déjà les messages triés du plus ancien au plus récent
        // Mais on s'assure qu'ils sont bien triés par timestamp pour garantir l'ordre chronologique
        const sorted = loaded.sort((a, b) => {
          const timeA = new Date(a.timestamp || 0).getTime();
          const timeB = new Date(b.timestamp || 0).getTime();
          return timeA - timeB; // Tri croissant : plus ancien en premier
        });
        setMessages(sorted);
        hasDoneInitialScrollRef.current = false;
        isAtBottomRef.current = true;
        setShowScrollButton(false);

        // Si on a moins de 20 messages, il n'y a plus de messages à charger
        setHasMoreMessages(loaded.length >= 20);

        // Forcer le scroll vers le bas après un délai pour s'assurer que tout est rendu
        // Utiliser requestAnimationFrame pour une transition plus fluide
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
                // Une seule vérification après un court délai pour garantir le scroll complet
                setTimeout(() => {
                  if (flatListRef.current && layoutHeightRef.current > 0) {
                    const offset2 = contentHeightRef.current - layoutHeightRef.current;
                    if (offset2 > 0) {
                      flatListRef.current.scrollToOffset({ offset: offset2, animated: false });
                    } else {
                      flatListRef.current.scrollToEnd({ animated: false });
                    }
                    // S'assurer qu'on est bien en bas
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
  }, [driver?.company_id]);

  // =============== FOCUS SCREEN (ex: retour sur l'onglet) ===============

  useFocusEffect(
    useCallback(() => {
      if (messages.length > 0 && isAtBottomRef.current) {
        // Utiliser plusieurs délais pour s'assurer que le layout est prêt
        const t = setTimeout(() => {
          scrollToBottom(false);
          // Double vérification pour s'assurer que le scroll est bien effectué
          setTimeout(() => {
            scrollToBottom(false);
          }, 100);
        }, 100);
        return () => clearTimeout(t);
      }
      return () => { };
    }, [messages.length, scrollToBottom])
  );

  // =============== KEYBOARD LISTENERS (Android uniquement) ===============

  useEffect(() => {
    if (Platform.OS !== "android") return;

    const keyboardDidShowListener = Keyboard.addListener(
      "keyboardDidShow",
      (event) => {
        const newKeyboardHeight = event.endCoordinates.height;
        // Mettre à jour la hauteur du clavier
        setKeyboardHeight(newKeyboardHeight);

        // Toujours scroller vers le bas quand le clavier s'ouvre pour montrer le dernier message
        // Le padding va augmenter (clavier + input), donc on doit scroller pour compenser
        // Utiliser requestAnimationFrame pour une transition plus fluide
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            setTimeout(() => {
              scrollToBottom(true);
              // Une seule vérification après un court délai pour garantir le scroll complet
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
        // Délai pour s'assurer que le clavier est bien fermé avant de réinitialiser
        setTimeout(() => {
          setKeyboardHeight(0);
          // Scroller vers le bas pour revenir à l'état initial (dernier message visible)
          // Le padding va diminuer, donc on doit scroller pour compenser et montrer le dernier message
          // Utiliser requestAnimationFrame pour une transition plus fluide
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              setTimeout(() => {
                scrollToBottom(true);
                // Une seule vérification après un court délai pour garantir le scroll complet
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
    const messageSpacing = 4; // Marge verticale augmentée pour que le dernier message soit bien visible juste au-dessus de l'input

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

  // Transformer les messages en liste avec séparateurs de date
  const listItemsWithDates = useMemo((): ListItem[] => {
    if (messages.length === 0) return [];

    const items: ListItem[] = [];
    let lastDate: string | null = null;

    for (const message of messages) {
      if (!message.timestamp) {
        // Si pas de timestamp, ajouter le message sans séparateur
        items.push({ type: "message", message });
        continue;
      }

      // Extraire la date au format YYYY-MM-DD
      const messageDate = new Date(message.timestamp);
      const dateKey = `${messageDate.getFullYear()}-${String(messageDate.getMonth() + 1).padStart(2, "0")}-${String(messageDate.getDate()).padStart(2, "0")}`;

      // Si c'est une nouvelle date, ajouter un séparateur
      if (lastDate !== dateKey) {
        items.push({ type: "dateSeparator", date: messageDate.toISOString() });
        lastDate = dateKey;
      }

      // Ajouter le message
      items.push({ type: "message", message });
    }

    return items;
  }, [messages]);

  // =============== RENDER ===============

  // Contenu commun (FlatList + Input)
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
              currentUserId={driver?.user?.id || driver?.user_id || null}
              onPressImage={setImagePreview}
              onPressPdf={setPdfPreview}
            />
          );
        }}
        keyExtractor={(item, index) => {
          if (item.type === "dateSeparator") {
            return `date-${item.date}-${index}`;
          }
          // Utiliser l'ID du message s'il existe, sinon utiliser l'index + timestamp pour garantir l'unicité
          if (item.message?.id != null) {
            return `msg-${item.message.id}`;
          }
          // Fallback : index + timestamp pour garantir l'unicité même si plusieurs messages n'ont pas d'ID
          const timestamp = item.message?.timestamp ? new Date(item.message.timestamp).getTime() : Date.now();
          return `msg-${index}-${timestamp}-${Math.random().toString(36).slice(2)}`;
        }}
        contentContainerStyle={contentContainerStyle}
        style={{ flex: 1 }}
        showsVerticalScrollIndicator
        onScroll={handleScroll}
        scrollEventThrottle={16}
        onLayout={(event) => {
          // Stocker la hauteur du layout pour calculer l'offset de scroll
          layoutHeightRef.current = event.nativeEvent.layout.height;
        }}
        // Pagination gérée dans handleScroll (scroll vers le haut)
        ListHeaderComponent={
          isLoadingMore ? (
            <View style={{ padding: 16, alignItems: "center" }}>
              <Text style={{ color: "#5F7369", fontSize: 14 }}>Chargement...</Text>
            </View>
          ) : null
        }
        onContentSizeChange={(contentWidth, contentHeight) => {
          if (listItemsWithDates.length === 0) return;

          // Stocker la hauteur du contenu pour calculer l'offset de scroll
          contentHeightRef.current = contentHeight;

          // 1er rendu après chargement → scroll instantané et invisible vers le dernier message
          if (!hasDoneInitialScrollRef.current) {
            hasDoneInitialScrollRef.current = true;
            previousContentHeightRef.current = contentHeight;
            // Scroll immédiat et invisible (sans animation) pour afficher directement le dernier message
            // Utiliser plusieurs requestAnimationFrame et setTimeout pour s'assurer que le layout et le padding sont prêts
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                setTimeout(() => {
                  if (flatListRef.current && layoutHeightRef.current > 0) {
                    // Calculer l'offset nécessaire pour scroller jusqu'en bas
                    const offset = contentHeight - layoutHeightRef.current;
                    if (offset > 0) {
                      flatListRef.current.scrollToOffset({ offset, animated: false });
                    } else {
                      flatListRef.current.scrollToEnd({ animated: false });
                    }
                    // Une seule vérification après un court délai pour garantir le scroll complet
                    setTimeout(() => {
                      if (flatListRef.current && layoutHeightRef.current > 0) {
                        const offset2 = contentHeightRef.current - layoutHeightRef.current;
                        if (offset2 > 0) {
                          flatListRef.current.scrollToOffset({ offset: offset2, animated: false });
                        } else {
                          flatListRef.current.scrollToEnd({ animated: false });
                        }
                        // S'assurer qu'on est bien en bas
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

          // Nouveau message & on était déjà en bas → rester collé
          if (isAtBottomRef.current && heightDifference > 0) {
            // Scroll animé pour suivre le nouveau message
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
              console.log("[ChatScreen] Keyboard dismiss error:", e);
            }
          }
        }}
        keyboardShouldPersistTaps="handled"
        ListEmptyComponent={() => (
          <View style={chatStyles.emptyContainer}>
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
        >
          <Ionicons
            name="attach"
            size={22}
            color="#0A7F59"
            style={{ transform: [{ rotate: "45deg" }] }}
          />
        </TouchableOpacity>

        <TextInput
          value={input}
          onChangeText={handleTyping}
          placeholder="Écrire un message..."
          placeholderTextColor={chatStyles.inputPlaceholder.color}
          style={chatStyles.input}
          multiline={false}
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
        >
          <Ionicons name="send" size={20} color="#FFFFFF" />
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={chatStyles.container}>
      <ChatHeader />

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
