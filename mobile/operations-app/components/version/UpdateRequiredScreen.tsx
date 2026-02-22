// components/version/UpdateRequiredScreen.tsx
// Écran bloquant affiché lorsque UPDATE_REQUIRED

import React from "react";
import {
    StyleSheet,
    Text,
    View,
    TouchableOpacity,
    Linking,
    Platform,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import Ionicons from "react-native-vector-icons/Ionicons";
import { useVersion } from "@/contexts/VersionContext";
import { getLogger } from "@/utils/logger";

const log = getLogger("Update");
export function UpdateRequiredScreen() {
    const { versionInfo } = useVersion();

    const handleUpdate = async () => {
        const storeUrl = versionInfo?.store_url;
        if (storeUrl) {
            try {
                const canOpen = await Linking.canOpenURL(storeUrl);
                if (canOpen) {
                    await Linking.openURL(storeUrl);
                } else {
                    log.warn("cannot open store", { storeUrl });
                }
            } catch (error) {
                log.error("open store failed", { error });
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
                log.error("open default store failed", { error });
            }
        }
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={["#06100C", "#10261A", "#06100C"]}
                style={StyleSheet.absoluteFill}
            />
            <View style={styles.content}>
                <View style={styles.iconContainer}>
                    <Ionicons name="alert-circle" size={64} color="#FF6B6B" />
                </View>

                <Text style={styles.title}>Mise à jour requise</Text>

                <Text style={styles.message}>
                    {versionInfo?.message ||
                        "Une nouvelle version de l'application est nécessaire pour continuer. Veuillez mettre à jour depuis le store."}
                </Text>

                {versionInfo && (
                    <View style={styles.versionInfo}>
                        <Text style={styles.versionLabel}>Version actuelle:</Text>
                        <Text style={styles.versionValue}>
                            {versionInfo.current_version}
                        </Text>
                        <Text style={styles.versionLabel}>Version requise:</Text>
                        <Text style={styles.versionValue}>
                            {versionInfo.min_required_version}
                        </Text>
                    </View>
                )}

                <TouchableOpacity
                    style={styles.updateButton}
                    onPress={handleUpdate}
                    activeOpacity={0.9}
                >
                    <Ionicons name="download-outline" size={20} color="#FFFFFF" />
                    <Text style={styles.updateButtonText}>Mettre à jour</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: "#06100C",
    },
    content: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        padding: 28,
    },
    iconContainer: {
        marginBottom: 24,
    },
    title: {
        fontSize: 28,
        fontWeight: "700",
        color: "#F4FFFA",
        marginBottom: 16,
        textAlign: "center",
    },
    message: {
        fontSize: 16,
        color: "#8AA295",
        textAlign: "center",
        lineHeight: 24,
        marginBottom: 32,
    },
    versionInfo: {
        backgroundColor: "rgba(244,255,250,0.05)",
        borderRadius: 12,
        padding: 16,
        marginBottom: 32,
        width: "100%",
        borderWidth: 1,
        borderColor: "rgba(121,197,156,0.18)",
    },
    versionLabel: {
        fontSize: 14,
        color: "#8AA295",
        marginBottom: 4,
    },
    versionValue: {
        fontSize: 16,
        fontWeight: "600",
        color: "#F4FFFA",
        marginBottom: 12,
    },
    updateButton: {
        backgroundColor: "#00796B",
        borderRadius: 14,
        paddingVertical: 16,
        paddingHorizontal: 32,
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        shadowColor: "#00796B",
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.25,
        shadowRadius: 12,
        elevation: 6,
        minWidth: 200,
        justifyContent: "center",
    },
    updateButtonText: {
        color: "#FFFFFF",
        fontSize: 16,
        fontWeight: "600",
        letterSpacing: 0.4,
    },
});

