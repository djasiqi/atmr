import React, { useEffect, useRef, useState } from "react";
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
import { Ionicons } from "@expo/vector-icons";

import { useAuth } from "@/hooks/useAuth";
import {
  getDispatchMessages,
  sendDispatchMessage,
} from "@/services/enterpriseDispatch";
import { DispatchMessage } from "@/types/enterpriseDispatch";

export default function EnterpriseChatScreen() {
  const { enterpriseSession } = useAuth();
  const [messages, setMessages] = useState<DispatchMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const listRef = useRef<FlatList<DispatchMessage>>(null);

  useEffect(() => {
    let isMounted = true;
    let timer: ReturnType<typeof setInterval> | null = null;

    const loadMessages = async () => {
      try {
        const data = await getDispatchMessages();
        if (isMounted) {
          setMessages(data);
          requestAnimationFrame(() =>
            listRef.current?.scrollToEnd({ animated: false })
          );
        }
      } catch {
        // ignorer les erreurs réseau ponctuelles
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    if (enterpriseSession) {
      loadMessages();
      timer = setInterval(loadMessages, 5000);
    }

    return () => {
      isMounted = false;
      if (timer) clearInterval(timer);
    };
  }, [enterpriseSession]);

  const handleSend = async () => {
    const content = input.trim();
    if (!content || sending) return;
    setSending(true);
    try {
      const message = await sendDispatchMessage(content);
      setMessages((prev) => [...prev, message]);
      requestAnimationFrame(() =>
        listRef.current?.scrollToEnd({ animated: true })
      );
      setInput("");
    } finally {
      setSending(false);
    }
  };

  const renderItem = ({ item }: { item: DispatchMessage }) => {
    const isCompany = (item.sender_role || "").toUpperCase() === "COMPANY";
    return (
      <View
        style={[styles.message, isCompany ? styles.outgoing : styles.incoming]}
      >
        <Text style={styles.messageSender}>
          {item.sender_name || (isCompany ? "Vous" : "Dispatch")}
        </Text>
        <Text style={styles.messageContent}>{item.content}</Text>
        <Text style={styles.messageTime}>
          {new Date(item.created_at).toLocaleTimeString("fr-CH", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </Text>
      </View>
    );
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      {loading ? (
        <View style={styles.loading}>
          <ActivityIndicator color="#4D6BFE" />
        </View>
      ) : (
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id.toString()}
          renderItem={renderItem}
          contentContainerStyle={styles.listContent}
        />
      )}

      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Écrire un message…"
          placeholderTextColor="#9AA5CC"
        />
        <TouchableOpacity
          style={styles.sendButton}
          onPress={handleSend}
          disabled={sending || !input.trim()}
        >
          <Ionicons name="send" size={22} color="#FFFFFF" />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B1736",
  },
  loading: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  listContent: {
    padding: 16,
    paddingBottom: 100,
  },
  message: {
    maxWidth: "80%",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  outgoing: {
    alignSelf: "flex-end",
    backgroundColor: "rgba(77,107,254,0.25)",
  },
  incoming: {
    alignSelf: "flex-start",
    backgroundColor: "rgba(255,255,255,0.12)",
  },
  messageSender: {
    color: "#D7DFFF",
    fontSize: 12,
    marginBottom: 4,
  },
  messageContent: {
    color: "#FFFFFF",
    fontSize: 15,
  },
  messageTime: {
    color: "#9AA5CC",
    fontSize: 11,
    marginTop: 6,
    textAlign: "right",
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderColor: "rgba(255,255,255,0.1)",
    backgroundColor: "rgba(10,19,46,0.95)",
  },
  input: {
    flex: 1,
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 24,
    paddingHorizontal: 16,
    paddingVertical: 10,
    color: "#FFFFFF",
    marginRight: 10,
  },
  sendButton: {
    backgroundColor: "#4D6BFE",
    borderRadius: 24,
    padding: 12,
  },
});
