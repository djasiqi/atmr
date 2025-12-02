// components/version/UpdateRecommendedModal.tsx
// Modal non bloquant affichée lorsque UPDATE_RECOMMENDED

import React, { useState } from "react";
import {
    StyleSheet,
    Text,
    View,
    TouchableOpacity,
    Modal,
    Linking,
    Platform,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import Ionicons from "react-native-vector-icons/Ionicons";
import { useVersion } from "@/contexts/VersionContext";

export function UpdateRecommendedModal() {
    const { versionInfo, status } = useVersion();
    const [dismissed, setDismissed] = useState(false);

    // Ne pas afficher si ce n'est pas UPDATE_RECOMMENDED ou si déjà fermé
    if (status !== "UPDATE_RECOMMENDED" || dismissed) {
        return null;
    }

    const handleUpdate = async () => {
        const storeUrl = versionInfo?.store_url;
        if (storeUrl) {
            try {
                const canOpen = await Linking.canOpenURL(storeUrl);
                if (canOpen) {
                    await Linking.openURL(storeUrl);
                } else {
                    console.warn("Impossible d'ouvrir le store:", storeUrl);
                }
            } catch (error) {
                console.error("Erreur lors de l'ouverture du store:", error);
            }
        } else {
            // Fallback: ouvrir le store par défaut selon la plateforme
            const defaultStoreUrl =
                Platform.OS === "android"
                    ? "https://play.google.com/store/apps/details?id=com.drinjasiqi.atmr"
                    : "https://apps.apple.com/app/id123456789"; // TODO: Remplacer par l'ID réel de l'App Store
            try {
                await Linking.openURL(defaultStoreUrl);
            } catch (error) {
                console.error("Erreur lors de l'ouverture du store par défaut:", error);
            }
        }
    };

    const handleDismiss = () => {
        setDismissed(true);
    };

    return (
        <Modal
            visible={true}
            transparent={true}
            animationType="fade"
            onRequestClose={handleDismiss}
        >
            <View style={styles.overlay}>
                <View style={styles.modalContent}>
                    <LinearGradient
                        colors={["#06100C", "#10261A"]}
                        style={[StyleSheet.absoluteFill, { borderRadius: 20 }]}
                    />
                    <TouchableOpacity
                        style={styles.closeButton}
                        onPress={handleDismiss}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="close" size={24} color="#8AA295" />
                    </TouchableOpacity>

                    <View style={styles.iconContainer}>
                        <Ionicons name="information-circle" size={48} color="#79C59C" />
                    </View>

                    <Text style={styles.title}>Mise à jour disponible</Text>

                    <Text style={styles.message}>
                        {versionInfo?.message ||
                            "Une nouvelle version de l'application est disponible. Nous vous recommandons de mettre à jour pour bénéficier des dernières améliorations."}
                    </Text>

                    {versionInfo && (
                        <View style={styles.versionInfo}>
                            <Text style={styles.versionText}>
                                Version actuelle: {versionInfo.current_version}
                            </Text>
                            <Text style={styles.versionText}>
                                Dernière version: {versionInfo.latest_version}
                            </Text>
                        </View>
                    )}

                    <View style={styles.buttonContainer}>
                        <TouchableOpacity
                            style={styles.dismissButton}
                            onPress={handleDismiss}
                            activeOpacity={0.7}
                        >
                            <Text style={styles.dismissButtonText}>Plus tard</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.updateButton}
                            onPress={handleUpdate}
                            activeOpacity={0.9}
                        >
                            <Ionicons name="download-outline" size={18} color="#FFFFFF" />
                            <Text style={styles.updateButtonText}>Mettre à jour</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        </Modal>
    );
}

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        justifyContent: "center",
        alignItems: "center",
        padding: 20,
    },
    modalContent: {
        width: "100%",
        maxWidth: 400,
        backgroundColor: "#10261A",
        borderRadius: 20,
        padding: 24,
        borderWidth: 1,
        borderColor: "rgba(121,197,156,0.18)",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.3,
        shadowRadius: 20,
        elevation: 10,
    },
    closeButton: {
        position: "absolute",
        top: 16,
        right: 16,
        zIndex: 1,
    },
    iconContainer: {
        alignItems: "center",
        marginBottom: 16,
        marginTop: 8,
    },
    title: {
        fontSize: 22,
        fontWeight: "700",
        color: "#F4FFFA",
        marginBottom: 12,
        textAlign: "center",
    },
    message: {
        fontSize: 15,
        color: "#8AA295",
        textAlign: "center",
        lineHeight: 22,
        marginBottom: 20,
    },
    versionInfo: {
        backgroundColor: "rgba(244,255,250,0.05)",
        borderRadius: 10,
        padding: 12,
        marginBottom: 20,
        borderWidth: 1,
        borderColor: "rgba(121,197,156,0.18)",
    },
    versionText: {
        fontSize: 13,
        color: "#8AA295",
        marginBottom: 4,
    },
    buttonContainer: {
        flexDirection: "row",
        gap: 12,
    },
    dismissButton: {
        flex: 1,
        backgroundColor: "rgba(244,255,250,0.1)",
        borderRadius: 12,
        paddingVertical: 14,
        alignItems: "center",
        borderWidth: 1,
        borderColor: "rgba(121,197,156,0.2)",
    },
    dismissButtonText: {
        color: "#8AA295",
        fontSize: 15,
        fontWeight: "600",
    },
    updateButton: {
        flex: 1,
        backgroundColor: "#00796B",
        borderRadius: 12,
        paddingVertical: 14,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        shadowColor: "#00796B",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.25,
        shadowRadius: 8,
        elevation: 4,
    },
    updateButtonText: {
        color: "#FFFFFF",
        fontSize: 15,
        fontWeight: "600",
    },
});

