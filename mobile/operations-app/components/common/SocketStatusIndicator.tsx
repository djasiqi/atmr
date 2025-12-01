// mobile/operations-app/components/common/SocketStatusIndicator.tsx

import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, Modal } from "react-native";
import { useSocketStatus } from "@/hooks/useSocketStatus";

/**
 * Indicateur discret du statut de connexion Socket.IO
 * - Badge discret en overlay (point coloré)
 * - Tap pour voir détails (modal)
 */
export default function SocketStatusIndicator() {
    const { connected, reconnecting, lastConnected } = useSocketStatus();
    const [showDetails, setShowDetails] = React.useState(false);

    // Déterminer la couleur et le texte
    let statusColor = "#F44336"; // Rouge - déconnecté
    let statusText = "Déconnecté";

    if (reconnecting) {
        statusColor = "#FF9800"; // Orange - reconnexion
        statusText = "Reconnexion...";
    } else if (connected) {
        statusColor = "#4CAF50"; // Vert - connecté
        statusText = "Connecté";
    }

    // Formater la date de dernière connexion
    const lastConnectedText = lastConnected
        ? `Dernière connexion: ${lastConnected.toLocaleTimeString("fr-FR")}`
        : "Jamais connecté";

    return (
        <>
            <TouchableOpacity
                style={[styles.indicator, { backgroundColor: statusColor }]}
                onPress={() => setShowDetails(true)}
                accessibilityLabel={`Statut connexion: ${statusText}`}
                accessibilityRole="button"
            >
                <View style={styles.dot} />
            </TouchableOpacity>

            <Modal
                visible={showDetails}
                transparent
                animationType="fade"
                onRequestClose={() => setShowDetails(false)}
            >
                <TouchableOpacity
                    style={styles.modalOverlay}
                    activeOpacity={1}
                    onPress={() => setShowDetails(false)}
                >
                    <View style={styles.modalContent}>
                        <Text style={styles.modalTitle}>Statut de connexion</Text>
                        <View style={styles.modalRow}>
                            <View
                                style={[styles.modalDot, { backgroundColor: statusColor }]}
                            />
                            <Text style={styles.modalText}>{statusText}</Text>
                        </View>
                        <Text style={styles.modalSubtext}>{lastConnectedText}</Text>
                        <TouchableOpacity
                            style={styles.modalButton}
                            onPress={() => setShowDetails(false)}
                        >
                            <Text style={styles.modalButtonText}>Fermer</Text>
                        </TouchableOpacity>
                    </View>
                </TouchableOpacity>
            </Modal>
        </>
    );
}

const styles = StyleSheet.create({
    indicator: {
        position: "absolute",
        top: 12,
        left: 12,
        width: 12,
        height: 12,
        borderRadius: 6,
        zIndex: 1000,
        elevation: 5, // Android shadow
        shadowColor: "#000", // iOS shadow
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
    },
    dot: {
        width: 12,
        height: 12,
        borderRadius: 6,
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        justifyContent: "center",
        alignItems: "center",
    },
    modalContent: {
        backgroundColor: "#fff",
        borderRadius: 12,
        padding: 20,
        width: "80%",
        maxWidth: 300,
        alignItems: "center",
    },
    modalTitle: {
        fontSize: 18,
        fontWeight: "bold",
        marginBottom: 16,
        color: "#333",
    },
    modalRow: {
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 8,
    },
    modalDot: {
        width: 16,
        height: 16,
        borderRadius: 8,
        marginRight: 8,
    },
    modalText: {
        fontSize: 16,
        color: "#333",
    },
    modalSubtext: {
        fontSize: 12,
        color: "#666",
        marginBottom: 16,
    },
    modalButton: {
        backgroundColor: "#0A7F59",
        paddingHorizontal: 24,
        paddingVertical: 10,
        borderRadius: 8,
    },
    modalButtonText: {
        color: "#fff",
        fontSize: 14,
        fontWeight: "600",
    },
});

