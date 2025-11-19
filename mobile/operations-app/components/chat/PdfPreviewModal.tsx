// components/chat/PdfPreviewModal.tsx
// ✅ WebView + BlurView premium

import React from "react";
import { Modal, StyleSheet, View, TouchableOpacity } from "react-native";
import { WebView } from "react-native-webview";
import { BlurView } from "expo-blur";
import { Ionicons } from "@expo/vector-icons";

type Props = {
    visible: boolean;
    pdfUrl: string | null;
    onClose: () => void;
};

export default function PdfPreviewModal({
    visible,
    pdfUrl,
    onClose,
}: Props) {
    if (!pdfUrl) return null;

    return (
        <Modal visible={visible} transparent animationType="fade">
            <BlurView intensity={40} tint="dark" style={StyleSheet.absoluteFill} />

            <View style={styles.container}>
                <View style={styles.viewer}>
                    <WebView
                        source={{ uri: pdfUrl }}
                        style={{ flex: 1 }}
                        javaScriptEnabled
                        domStorageEnabled
                    />
                </View>

                <TouchableOpacity style={styles.closeBtn} onPress={onClose}>
                    <Ionicons name="close" size={30} color="white" />
                </TouchableOpacity>
            </View>
        </Modal>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: "center",
        paddingHorizontal: 16,
    },
    viewer: {
        backgroundColor: "#FFFFFF",
        borderRadius: 18,
        overflow: "hidden",
        height: "85%",
        elevation: 6,
    },
    closeBtn: {
        position: "absolute",
        top: 40,
        right: 20,
        backgroundColor: "#0A7F59",
        width: 48,
        height: 48,
        borderRadius: 24,
        justifyContent: "center",
        alignItems: "center",
        shadowColor: "#0A7F59",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
    },
});
