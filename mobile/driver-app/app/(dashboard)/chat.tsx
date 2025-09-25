import React, { useState, useEffect } from 'react';
import {
  View,
  FlatList,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useAuth } from '@/hooks/useAuth';
import { useSocket } from '@/hooks/useSocket';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Loader } from '@/components/ui/Loader';
import { Feather } from '@expo/vector-icons';

interface ChatMessage {
  sender: string;
  content: string;
  timestamp: string;
}

export default function ChatScreen() {
  const { driver } = useAuth();
  const socket = useSocket();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');

  useEffect(() => {
    if (!socket || !driver?.company?.id) return;

    const room = `company-${driver.company.id}`;
    socket.emit('joinRoom', { room });

    const handleMessage = (msg: ChatMessage) => {
      setMessages((prevMessages) => [msg, ...prevMessages]);
    };

    socket.on('chatMessage', handleMessage);

    return () => {
      socket.emit('leaveRoom', { room });
      socket.off('chatMessage', handleMessage);
    };
  }, [socket, driver?.company?.id]);

  const sendMessage = () => {
    if (!input.trim() || !socket || !driver?.user || !driver?.company?.id) return;

    const newMessage: ChatMessage = {
      sender: `${driver.first_name} ${driver.last_name}`,
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    socket.emit('chatMessage', {
      room: `company-${driver.company.id}`,
      message: newMessage,
    });

    setMessages((prev) => [newMessage, ...prev]);
    setInput('');
  };

  if (!driver) {
    return (
      <ThemedView className="flex-1 justify-center items-center">
        <Loader />
      </ThemedView>
    );
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      className="flex-1 bg-white dark:bg-black"
    >
      <FlatList
        inverted
        data={messages}
        keyExtractor={(_, index) => index.toString()}
        renderItem={({ item }) => (
          <ThemedView className="px-4 py-2">
            <ThemedText className="text-sm font-semibold">{item.sender}</ThemedText>
            <ThemedText className="text-base mt-1">{item.content}</ThemedText>
            <ThemedText className="text-xs text-gray-500 mt-1">
              {new Date(item.timestamp).toLocaleTimeString()}
            </ThemedText>
          </ThemedView>
        )}
      />

      <View className="flex-row items-center border-t border-gray-200 px-4 py-2 bg-gray-100">
        <TextInput
          className="flex-1 bg-white px-3 py-2 rounded-lg border border-gray-200"
          placeholder="Écrivez votre message..."
          value={input}
          onChangeText={setInput}
        />

        <TouchableOpacity className="ml-2 p-2" onPress={sendMessage}>
          <Feather name="send" size={24} color="#3b82f6" />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}
