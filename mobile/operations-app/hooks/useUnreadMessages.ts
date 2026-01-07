// hooks/useUnreadMessages.ts
import { useState, useEffect, useRef, useCallback } from "react";
import { useFocusEffect } from "@react-navigation/native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useEnterpriseSocket } from "@/hooks/useEnterpriseSocket";
import { useAuth } from "@/hooks/useAuth";

const LAST_READ_MESSAGE_KEY = "enterprise_chat_last_read_message_id";
const LAST_READ_TIMESTAMP_KEY = "enterprise_chat_last_read_timestamp";

export const useUnreadMessages = () => {
  const { enterpriseSession } = useAuth();
  const socket = useEnterpriseSocket();
  const [unreadCount, setUnreadCount] = useState(0);
  const lastReadMessageIdRef = useRef<number | null>(null);
  const lastReadTimestampRef = useRef<number | null>(null);
  const isChatFocusedRef = useRef(false);
  const unreadMessagesRef = useRef<Set<number>>(new Set());

  // Charger le dernier message lu depuis AsyncStorage
  useEffect(() => {
    const loadLastRead = async () => {
      try {
        const lastReadId = await AsyncStorage.getItem(LAST_READ_MESSAGE_KEY);
        const lastReadTimestamp = await AsyncStorage.getItem(LAST_READ_TIMESTAMP_KEY);
        
        if (lastReadId) {
          lastReadMessageIdRef.current = parseInt(lastReadId, 10);
        }
        if (lastReadTimestamp) {
          lastReadTimestampRef.current = parseInt(lastReadTimestamp, 10);
        }
      } catch (error) {
        console.warn("[useUnreadMessages] Erreur lors du chargement du dernier message lu:", error);
      }
    };
    
    loadLastRead();
  }, []);

  // Écouter les nouveaux messages via Socket.IO
  useEffect(() => {
    if (!socket || !enterpriseSession?.user?.id) {
      return;
    }

    const handleNewMessage = (message: any) => {
      // Ignorer les messages envoyés par l'utilisateur actuel
      if (message.sender_id === enterpriseSession.user.id) {
        return;
      }

      const messageId = typeof message.id === "string" 
        ? parseInt(message.id, 10) 
        : message.id;
      const messageTimestamp = message.timestamp 
        ? new Date(message.timestamp).getTime() 
        : Date.now();

      if (isNaN(messageId)) {
        return;
      }

      // Si le chat n'est pas au focus, ajouter le message aux non lus
      if (!isChatFocusedRef.current) {
        // Vérifier si le message est plus récent que le dernier lu
        const shouldMarkAsUnread = 
          !lastReadMessageIdRef.current || 
          messageId > lastReadMessageIdRef.current ||
          (lastReadTimestampRef.current && messageTimestamp > lastReadTimestampRef.current);

        if (shouldMarkAsUnread) {
          unreadMessagesRef.current.add(messageId);
          setUnreadCount(unreadMessagesRef.current.size);
        }
      } else {
        // Si le chat est au focus, marquer le message comme lu immédiatement
        markAsRead(messageId, messageTimestamp);
      }
    };

    socket.on("team_chat_message", handleNewMessage);

    return () => {
      socket.off("team_chat_message", handleNewMessage);
    };
  }, [socket, enterpriseSession, markAsRead]);

  // Marquer tous les messages comme lus
  const markAsRead = useCallback(async (messageId?: number, timestamp?: number) => {
    try {
      const currentTimestamp = timestamp || Date.now();
      const currentMessageId = messageId || (lastReadMessageIdRef.current ?? 0);

      // Mettre à jour les refs
      lastReadMessageIdRef.current = currentMessageId;
      lastReadTimestampRef.current = currentTimestamp;

      // Sauvegarder dans AsyncStorage
      await AsyncStorage.setItem(LAST_READ_MESSAGE_KEY, String(currentMessageId));
      await AsyncStorage.setItem(LAST_READ_TIMESTAMP_KEY, String(currentTimestamp));

      // Réinitialiser le compteur de messages non lus
      unreadMessagesRef.current.clear();
      setUnreadCount(0);
    } catch (error) {
      console.warn("[useUnreadMessages] Erreur lors de la sauvegarde du dernier message lu:", error);
    }
  }, []);

  // Marquer comme lu quand l'écran chat est au focus
  useFocusEffect(
    useCallback(() => {
      isChatFocusedRef.current = true;
      // Marquer tous les messages comme lus quand on ouvre le chat
      markAsRead();
      
      return () => {
        isChatFocusedRef.current = false;
      };
    }, [markAsRead])
  );

  // Fonction pour compter les messages non lus dans une liste de messages
  const countUnreadInMessages = useCallback((messages: Array<{ id: number | string; sender_id?: number | null; timestamp?: string | number }>) => {
    if (!enterpriseSession?.user?.id || !lastReadMessageIdRef.current) {
      return 0;
    }

    let count = 0;
    for (const msg of messages) {
      // Ignorer les messages de l'utilisateur actuel
      if (msg.sender_id === enterpriseSession.user.id) {
        continue;
      }

      const messageId = typeof msg.id === "string" ? parseInt(msg.id, 10) : msg.id;
      if (isNaN(messageId)) {
        continue;
      }

      // Vérifier si le message est plus récent que le dernier lu
      const messageTimestamp = msg.timestamp 
        ? new Date(msg.timestamp).getTime() 
        : Date.now();
      
      const isUnread = 
        messageId > lastReadMessageIdRef.current ||
        (lastReadTimestampRef.current && messageTimestamp > lastReadTimestampRef.current);

      if (isUnread) {
        count++;
        unreadMessagesRef.current.add(messageId);
      }
    }

    if (count > 0) {
      setUnreadCount(unreadMessagesRef.current.size);
    }

    return count;
  }, [enterpriseSession]);

  // Fonction pour réinitialiser le compteur (utile après avoir lu les messages)
  const resetUnreadCount = useCallback(() => {
    unreadMessagesRef.current.clear();
    setUnreadCount(0);
  }, []);

  return {
    unreadCount,
    markAsRead,
    resetUnreadCount,
    countUnreadInMessages,
  };
};

