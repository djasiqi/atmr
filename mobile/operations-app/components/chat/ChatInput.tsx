// components/chat/ChatInput.tsx
import React, { useState } from "react";
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    Modal,
    Alert,
    Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
// import * as DocumentPicker from "expo-document-picker"; // TODO: Installer expo-document-picker
import { chatStyles } from "@/styles/chatStyles";

interface ChatInputProps {
    value: string;
    onChangeText: (text: string) => void;
    onSend: (content: string) => void;
    onSendImage?: (imageUri: string, base64?: string) => void;
    onSendPdf?: (pdfUri: string, filename: string) => void;
    placeholder?: string;
}

export default function ChatInput({
    value,
    onChangeText,
    onSend,
    onSendImage,
    onSendPdf,
    placeholder = "Écrire un message...",
}: ChatInputProps) {
    const [showAttachMenu, setShowAttachMenu] = useState(false);

    const handleSend = () => {
        const content = value.trim();
        if (!content) return;
        onSend(content);
    };

    const handleCamera = async () => {
        setShowAttachMenu(false);
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== "granted") {
            Alert.alert("Permission requise", "Permission caméra nécessaire");
            return;
        }

        const result = await ImagePicker.launchCameraAsync({
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.7,
            base64: true,
        });

        if (!result.canceled && result.assets[0] && onSendImage) {
            const asset = result.assets[0];
            const base64 = asset.base64
                ? `data:image/jpeg;base64,${asset.base64}`
                : undefined;
            onSendImage(asset.uri, base64);
        }
    };

    const handleGallery = async () => {
        setShowAttachMenu(false);
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ["images"],
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.7,
            base64: true,
        });

        if (!result.canceled && result.assets[0] && onSendImage) {
            const asset = result.assets[0];
            const base64 = asset.base64
                ? `data:image/jpeg;base64,${asset.base64}`
                : undefined;
            onSendImage(asset.uri, base64);
        }
    };

    const handleDocument = async () => {
        setShowAttachMenu(false);
        // TODO: Implémenter avec expo-document-picker une fois installé
        Alert.alert(
            "Fonctionnalité à venir",
            "La sélection de PDF sera disponible prochainement"
        );
        // try {
        //   const result = await DocumentPicker.getDocumentAsync({
        //     type: "application/pdf",
        //     copyToCacheDirectory: true,
        //   });
        //
        //   if (!result.canceled && result.assets[0] && onSendPdf) {
        //     const asset = result.assets[0];
        //     onSendPdf(asset.uri, asset.name || "document.pdf");
        //   }
        // } catch (error) {
        //   console.log("Erreur sélection PDF:", error);
        //   Alert.alert("Erreur", "Impossible de sélectionner le document");
        // }
    };

    return (
        <>
            <View style={chatStyles.inputContainer}>
                {/* Bouton attach */}
                <TouchableOpacity
                    style={styles.attachButton}
                    onPress={() => setShowAttachMenu(true)}
                    hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                >
                    <Ionicons name="attach" size={24} color="#5F7369" />
                </TouchableOpacity>

                {/* Input */}
                <TextInput
                    value={value}
                    onChangeText={onChangeText}
                    placeholder={placeholder}
                    placeholderTextColor={chatStyles.inputPlaceholder.color}
                    style={chatStyles.input}
                    multiline={false}
                    returnKeyType="send"
                    onSubmitEditing={handleSend}
                />

                {/* Bouton send */}
                {value.trim() ? (
                    <TouchableOpacity
                        style={chatStyles.sendButton}
                        onPress={handleSend}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="send" size={20} color="#FFFFFF" />
                    </TouchableOpacity>
                ) : (
                    <TouchableOpacity
                        style={styles.cameraButton}
                        onPress={handleCamera}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="camera" size={24} color="#5F7369" />
                    </TouchableOpacity>
                )}
            </View>

            {/* Menu attach */}
            <Modal
                visible={showAttachMenu}
                transparent={true}
                animationType="fade"
                onRequestClose={() => setShowAttachMenu(false)}
            >
                <TouchableOpacity
                    style={styles.modalOverlay}
                    activeOpacity={1}
                    onPress={() => setShowAttachMenu(false)}
                >
                    <View style={styles.attachMenu}>
                        <TouchableOpacity
                            style={styles.menuItem}
                            onPress={handleCamera}
                        >
                            <View style={[styles.menuIcon, { backgroundColor: "#E8F5E8" }]}>
                                <Ionicons name="camera" size={28} color="#0A7F59" />
                            </View>
                            <Text style={styles.menuText}>Caméra</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.menuItem}
                            onPress={handleGallery}
                        >
                            <View style={[styles.menuIcon, { backgroundColor: "#E8F5E8" }]}>
                                <Ionicons name="images" size={28} color="#0A7F59" />
                            </View>
                            <Text style={styles.menuText}>Galerie</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.menuItem}
                            onPress={handleDocument}
                        >
                            <View style={[styles.menuIcon, { backgroundColor: "#E8F5E8" }]}>
                                <Ionicons name="document-text" size={28} color="#0A7F59" />
                            </View>
                            <Text style={styles.menuText}>PDF</Text>
                        </TouchableOpacity>
                    </View>
                </TouchableOpacity>
            </Modal>
        </>
    );
}

const styles = StyleSheet.create({
    attachButton: {
        width: 40,
        height: 40,
        justifyContent: "center",
        alignItems: "center",
        marginRight: 8,
    },
    cameraButton: {
        width: 50,
        height: 50,
        borderRadius: 25,
        backgroundColor: "#F5F7F6",
        justifyContent: "center",
        alignItems: "center",
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        justifyContent: "flex-end",
    },
    attachMenu: {
        backgroundColor: "#FFFFFF",
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        paddingTop: 20,
        paddingBottom: 40,
        paddingHorizontal: 20,
        flexDirection: "row",
        justifyContent: "space-around",
    },
    menuItem: {
        alignItems: "center",
        flex: 1,
    },
    menuIcon: {
        width: 56,
        height: 56,
        borderRadius: 28,
        justifyContent: "center",
        alignItems: "center",
        marginBottom: 8,
    },
    menuText: {
        fontSize: 13,
        color: "#15362B",
        fontWeight: "600",
    },
});

