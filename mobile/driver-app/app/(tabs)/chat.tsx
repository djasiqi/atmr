import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
} from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";
import { Ionicons } from "@expo/vector-icons";
import { useBottomTabBarHeight } from "@react-navigation/bottom-tabs";
import api, { Message } from "@/services/api"; // ✅ import du type

export default function ChatScreen() {
  const { driver } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const flatListRef = useRef<FlatList>(null);
  const tabBarHeight = useBottomTabBarHeight();

  const socket = useSocket(undefined, (msg: Message) => {
    setMessages((prev) => [...prev, msg]);
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
  });

  const sendMessage = () => {
    const content = input.trim();
    if (!content || !socket) return;

    socket.emit("team_chat_message", {
      content,
      receiver_id: null,
    });

    setInput("");
  };

  const renderItem = ({ item }: { item: Message }) => (
    <View
      style={[
        styles.messageContainer,
        item.sender_role === "driver"
          ? styles.driverMessage
          : styles.companyMessage,
      ]}
    >
      <Text style={styles.sender}>{item.sender} ({item.sender_role})</Text>
      <Text>{item.content}</Text>
    </View>
  );

  useEffect(() => {
    const loadHistory = async () => {
      if (!driver?.company_id) {
        return;
      }

      try {
        console.log("📨 Récupération des messages pour company_id :", driver.company_id);
        const response = await api.get(`/messages/${driver.company_id}`);
        setMessages(response.data);
        setTimeout(() => flatListRef.current?.scrollToEnd({ animated: false }), 100);
      } catch (error) {
      }
    };

    loadHistory();
  }, [driver?.company_id]);

  return (
    <KeyboardAvoidingView
      style={[styles.container, { paddingBottom: tabBarHeight + 60 }]}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      keyboardVerticalOffset={tabBarHeight + 40}
    >
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={(item) => item.id.toString()}
        contentContainerStyle={{ paddingBottom: 10 }}
      />

      <View style={[styles.inputContainer, { bottom: tabBarHeight }]}>
        <TextInput
          value={input}
          onChangeText={setInput}
          placeholder="Écrire un message..."
          style={styles.input}
        />
        <TouchableOpacity onPress={sendMessage} style={styles.sendButton}>
          <Ionicons name="send" size={24} color="white" />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

// styles identiques
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    padding: 10,
  },
  messageContainer: {
    borderRadius: 8,
    padding: 10,
    marginVertical: 5,
    maxWidth: "80%",
  },
  driverMessage: {
    backgroundColor: "#e1f5fe",
    alignSelf: "flex-end",
  },
  companyMessage: {
    backgroundColor: "#f1f1f1",
    alignSelf: "flex-start",
  },
  sender: {
    fontWeight: "bold",
    marginBottom: 4,
  },
  inputContainer: {
    position: "absolute",
    left: 0,
    right: 0,
    flexDirection: "row",
    padding: 8,
    borderTopWidth: 1,
    borderColor: "#ccc",
    backgroundColor: "#f9f9f9",
    alignItems: "center",
  },
  input: {
    flex: 1,
    padding: 12,
    backgroundColor: "#fff",
    borderRadius: 25,
    borderWidth: 1,
    borderColor: "#ccc",
    marginRight: 8,
  },
  sendButton: {
    backgroundColor: "#007aff",
    borderRadius: 25,
    padding: 12,
    justifyContent: "center",
    alignItems: "center",
  },
});
