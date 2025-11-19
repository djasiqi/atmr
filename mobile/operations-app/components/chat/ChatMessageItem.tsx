// components/chat/ChatMessageItem.tsx
import React from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Message } from "@/services/api";
import { chatStyles } from "@/styles/chatStyles";

interface ChatMessageItemProps {
    message: Message;
    onImagePress?: (uri: string) => void;
    onPdfPress?: (uri: string) => void;
}

export default function ChatMessageItem({
    message,
    onImagePress,
    onPdfPress,
}: ChatMessageItemProps) {
    const isDriver = String(message.sender_role).toUpperCase() === "DRIVER";
    const textStyle = isDriver
        ? chatStyles.messageTextDriver
        : chatStyles.messageTextCompany;

    // Détecter si le message contient une image ou un PDF
    const hasImage = message.image_url || message.image;
    const hasPdf = message.pdf_url || message.pdf;
    const hasContent = message.content && message.content.trim().length > 0;

    return (
        <View
            style={[
                chatStyles.messageContainer,
                isDriver ? chatStyles.driverMessage : chatStyles.companyMessage,
            ]}
        >
            {/* Nom expéditeur */}
            {message.sender_name && (
                <Text style={[chatStyles.senderName, textStyle]}>
                    {message.sender_name}
                </Text>
            )}

            {/* Contenu texte */}
            {hasContent && <Text style={textStyle}>{message.content}</Text>}

            {/* Image */}
            {hasImage && (() => {
                const imageUri = message.image_url || message.image;
                if (!imageUri) return null;
                return (
                    <TouchableOpacity
                        style={chatStyles.imageMessage}
                        onPress={() => {
                            if (onImagePress) {
                                onImagePress(imageUri);
                            }
                        }}
                        activeOpacity={0.9}
                    >
                        <Image
                            source={{ uri: imageUri }}
                            style={chatStyles.imagePreview}
                            resizeMode="cover"
                        />
                    </TouchableOpacity>
                );
            })()}

            {/* PDF */}
            {hasPdf && (
                <TouchableOpacity
                    style={chatStyles.pdfMessage}
                    onPress={() => {
                        const pdfUri = message.pdf_url || message.pdf;
                        if (pdfUri && onPdfPress) {
                            onPdfPress(pdfUri);
                        }
                    }}
                    activeOpacity={0.7}
                >
                    <Ionicons
                        name="document-text"
                        size={24}
                        color={isDriver ? "#D0F5E2" : "#5F7369"}
                        style={chatStyles.pdfIcon}
                    />
                    <View style={chatStyles.pdfInfo}>
                        <Text
                            style={[
                                chatStyles.pdfFileName,
                                { color: isDriver ? "#D0F5E2" : "#15362B" },
                            ]}
                        >
                            {message.pdf_filename || "Document PDF"}
                        </Text>
                        {message.pdf_size && (
                            <Text
                                style={[
                                    chatStyles.pdfFileSize,
                                    { color: isDriver ? "#D0F5E2" : "#5F7369" },
                                ]}
                            >
                                {formatFileSize(message.pdf_size)}
                            </Text>
                        )}
                    </View>
                </TouchableOpacity>
            )}

            {/* Footer avec timestamp et tick */}
            <View style={chatStyles.footerRow}>
                <Text
                    style={[
                        chatStyles.timestamp,
                        isDriver && { color: "#D0F5E2" },
                    ]}
                >
                    {new Date(message.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                    })}
                </Text>
                {/* Tick pour les messages du chauffeur */}
                {isDriver && (
                    <Ionicons
                        name="checkmark-done"
                        size={14}
                        color="#D0F5E2"
                        style={chatStyles.tickIcon}
                    />
                )}
            </View>
        </View>
    );
}

// Helper pour formater la taille de fichier
function formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

