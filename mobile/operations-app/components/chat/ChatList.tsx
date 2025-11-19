// components/chat/ChatList.tsx
import React, { useRef, useCallback, useState } from "react";
import {
    View,
    Text,
    FlatList,
    TouchableOpacity,
    NativeScrollEvent,
    NativeSyntheticEvent,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Message } from "@/services/api";
import ChatMessageItem from "./ChatMessageItem";
import ChatTypingIndicator from "./ChatTypingIndicator";
import { chatStyles } from "@/styles/chatStyles";

interface ChatListProps {
    messages: Message[];
    isTyping?: boolean;
    onImagePress?: (uri: string) => void;
    onPdfPress?: (uri: string) => void;
    onScrollToBottom?: () => void;
}

export default function ChatList({
    messages,
    isTyping = false,
    onImagePress,
    onPdfPress,
    onScrollToBottom,
}: ChatListProps) {
    const flatListRef = useRef<FlatList<Message>>(null);
    const isAtBottomRef = useRef(true);
    const [showScrollButton, setShowScrollButton] = useState(false);
    const initialScrollDone = useRef(false);

    const scrollToBottom = useCallback(
        (animated = true) => {
            if (!flatListRef.current) return;
            try {
                flatListRef.current.scrollToEnd({ animated });
                setShowScrollButton(false);
                isAtBottomRef.current = true;
            } catch { }
        },
        []
    );

    const handleScroll = (event: NativeSyntheticEvent<NativeScrollEvent>) => {
        const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
        const padding = 40;
        const isAtBottom =
            layoutMeasurement.height + contentOffset.y >=
            contentSize.height - padding;

        isAtBottomRef.current = isAtBottom;
        setShowScrollButton(!isAtBottom);

        // Si l'utilisateur atteint le bas, réactiver l'auto-scroll
        if (isAtBottom && onScrollToBottom) {
            onScrollToBottom();
        }
    };

    const renderItem = ({ item }: { item: Message }) => (
        <ChatMessageItem
            message={item}
            onImagePress={onImagePress}
            onPdfPress={onPdfPress}
        />
    );

    return (
        <View style={{ flex: 1 }}>
            <FlatList
                ref={flatListRef}
                data={messages}
                renderItem={renderItem}
                keyExtractor={(item) =>
                    item?.id ? String(item.id) : Math.random().toString()
                }
                showsVerticalScrollIndicator={true}
                scrollEventThrottle={16}
                onScroll={handleScroll}
                onContentSizeChange={() => {
                    // Premier chargement → scroll direct
                    if (!initialScrollDone.current) {
                        initialScrollDone.current = true;
                        scrollToBottom(false);
                        return;
                    }

                    // Nouveau message alors qu'on est en bas → rester collé
                    if (isAtBottomRef.current) {
                        scrollToBottom(true);
                    }
                }}
                ListFooterComponent={() => {
                    if (!isTyping) return null;
                    return <ChatTypingIndicator />;
                }}
                ListEmptyComponent={() => (
                    <View style={chatStyles.emptyContainer}>
                        <Text style={chatStyles.emptyText}>
                            Aucun message pour le moment.{"\n"}
                            Commencez la conversation avec votre équipe !
                        </Text>
                    </View>
                )}
            />

            {/* Bouton scroll to bottom */}
            {showScrollButton && (
                <TouchableOpacity
                    style={chatStyles.scrollToBottomButton}
                    onPress={() => scrollToBottom(true)}
                    activeOpacity={0.8}
                >
                    <Ionicons name="chevron-down" size={24} color="#FFFFFF" />
                </TouchableOpacity>
            )}
        </View>
    );
}

