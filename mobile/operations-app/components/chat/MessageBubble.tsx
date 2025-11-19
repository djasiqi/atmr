// components/chat/MessageBubble.tsx
// ✅ Version WhatsApp Premium Liri avec animations Reanimated

import React from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet } from "react-native";
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withTiming,
    withDelay,
    interpolate,
} from "react-native-reanimated";
import { Ionicons } from "@expo/vector-icons";
import { Message } from "@/services/api";
import Avatar from "./Avatar";

interface Props {
    message: Message;
    currentUserId?: number | null; // ID de l'utilisateur actuel (driver.id)
    onPressImage?: (uri: string) => void;
    onPressPdf?: (uri: string) => void;
}

export default function MessageBubble({
    message,
    currentUserId,
    onPressImage,
    onPressPdf,
}: Props) {
    // Déterminer si le message est envoyé par l'utilisateur actuel
    // Si sender_id correspond à currentUserId, c'est un message envoyé (à droite)
    // Sinon, c'est un message reçu (à gauche)
    // Comparer en convertissant en nombres pour éviter les problèmes de type
    const isOwnMessage =
        currentUserId != null &&
        message.sender_id != null &&
        Number(message.sender_id) === Number(currentUserId);

    // ---- ANIMATION ----
    const opacity = useSharedValue(0);
    const translateY = useSharedValue(15);

    React.useEffect(() => {
        opacity.value = withDelay(50, withTiming(1, { duration: 180 }));
        translateY.value = withDelay(20, withTiming(0, { duration: 180 }));
    }, []);

    const animatedStyle = useAnimatedStyle(() => ({
        opacity: opacity.value,
        transform: [{ translateY: translateY.value }],
    }));

    // ---- PDF SIZE FORMAT ----
    const formatSize = (bytes: number | null | undefined) => {
        if (!bytes) return "";
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    };

    const imageUri = message.image_url || message.image;
    const pdfUri = message.pdf_url || message.pdf;

    return (
        <Animated.View
            style={[
                styles.wrapper,
                isOwnMessage ? styles.rightWrapper : styles.leftWrapper,
                animatedStyle,
            ]}
        >
            {/* Avatar uniquement pour les messages reçus (à gauche) */}
            {!isOwnMessage && (
                <View style={styles.avatarContainer}>
                    <Avatar
                        photo={null} // TODO: Ajouter sender_photo si disponible dans le backend
                        name={message.sender_name || undefined}
                        size={32}
                    />
                </View>
            )}
            <View
                style={[
                    styles.container,
                    isOwnMessage ? styles.rightContainer : styles.leftContainer,
                ]}
            >
                <View
                    style={[
                        styles.bubble,
                        isOwnMessage ? styles.bubbleDriver : styles.bubbleCompany,
                    ]}
                >
                    {/* --- IMAGE --- */}
                    {imageUri && (
                        <TouchableOpacity
                            onPress={() => onPressImage?.(imageUri)}
                            activeOpacity={0.8}
                        >
                            <Image source={{ uri: imageUri }} style={styles.image} />
                        </TouchableOpacity>
                    )}

                    {/* --- PDF --- */}
                    {pdfUri && (
                        <TouchableOpacity
                            onPress={() => onPressPdf?.(pdfUri)}
                            style={styles.pdfContainer}
                            activeOpacity={0.9}
                        >
                            <Ionicons name="document-text-outline" size={28} color="#0A7F59" />
                            <View style={{ flex: 1 }}>
                                <Text style={styles.pdfName} numberOfLines={1}>
                                    {message.pdf_filename || "Document PDF"}
                                </Text>
                                {message.pdf_size && (
                                    <Text style={styles.pdfSize}>
                                        {formatSize(message.pdf_size)}
                                    </Text>
                                )}
                            </View>
                        </TouchableOpacity>
                    )}

                    {/* --- TEXTE --- */}
                    {message.content && (
                        <Text
                            style={[
                                styles.text,
                                isOwnMessage ? styles.textDriver : styles.textCompany,
                            ]}
                        >
                            {message.content}
                        </Text>
                    )}

                    {/* ---- TIMESTAMP ---- */}
                    {message.timestamp && (
                        <Text style={[
                            styles.timestamp,
                            isOwnMessage ? styles.timestampOwn : styles.timestampReceived
                        ]}>
                            {new Date(message.timestamp).toLocaleTimeString("fr-FR", {
                                hour: "2-digit",
                                minute: "2-digit",
                                hour12: false,
                            })}
                        </Text>
                    )}
                </View>
            </View>
        </Animated.View>
    );
}

const styles = StyleSheet.create({
    wrapper: {
        flexDirection: "row",
        marginVertical: 1,
        marginHorizontal: 6,
        alignItems: "flex-end",
    },
    leftWrapper: {
        justifyContent: "flex-start",
    },
    rightWrapper: {
        justifyContent: "flex-end",
    },
    avatarContainer: {
        marginRight: 8,
        marginBottom: 2,
    },
    container: {
        maxWidth: "85%",
    },
    leftContainer: {
        alignSelf: "flex-start",
    },
    rightContainer: {
        alignSelf: "flex-end",
    },
    bubble: {
        paddingHorizontal: 12,
        paddingVertical: 1,
        borderRadius: 16,
        borderWidth: 1,
        shadowColor: "rgba(0,0,0,0.08)",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 2,
    },
    bubbleDriver: {
        backgroundColor: "#0A7F59",
        borderColor: "rgba(0,0,0,0.1)",
    },
    bubbleCompany: {
        backgroundColor: "#ffffff",
        borderColor: "rgba(0,0,0,0.05)",
    },
    text: {
        fontSize: 14,
        lineHeight: 21,
    },
    textDriver: {
        color: "#ffffff",
    },
    textCompany: {
        color: "#15362B",
    },
    timestamp: {
        fontSize: 10,
        alignSelf: "flex-end",
        marginTop: 4,
    },
    timestampOwn: {
        color: "rgba(255,255,255,0.7)", // Plus clair pour les messages envoyés (sur fond vert)
    },
    timestampReceived: {
        color: "rgba(21,54,43,0.45)", // Couleur originale pour les messages reçus
    },
    image: {
        width: 190,
        height: 190,
        borderRadius: 10,
        marginBottom: 6,
        backgroundColor: "#d9d9d9",
    },
    pdfContainer: {
        flexDirection: "row",
        backgroundColor: "rgba(10,127,89,0.08)",
        borderRadius: 12,
        padding: 10,
        alignItems: "center",
        marginBottom: 6,
        borderWidth: 1,
        borderColor: "rgba(10,127,89,0.2)",
    },
    pdfName: {
        fontSize: 14,
        color: "#15362B",
        fontWeight: "600",
        marginBottom: 2,
        marginLeft: 10,
    },
    pdfSize: {
        fontSize: 12,
        color: "#5F7369",
        marginLeft: 10,
    },
});

