import React from "react";
import {
    Modal,
    View,
    Text,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    Pressable,
    StyleSheet,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import type { RideSummary, DriverSuggestion } from "@/types/enterpriseDispatch";

// ✅ Palette professionnelle cohérente avec le dashboard driver
const palette = {
    modalOverlay: "rgba(21,54,43,0.75)",
    modalBackground: "#FFFFFF",
    modalBorder: "rgba(15,54,43,0.12)",
    modalTitle: "#15362B",
    modalText: "#5F7369",
    modalButton: "#0A7F59",
    modalButtonText: "#FFFFFF",
    modalCancelText: "#5F7369",
    loadingText: "#91A59D",
    surfaceBorder: "rgba(15,54,43,0.08)",
};

interface AssignDriverModalProps {
    visible: boolean;
    ride: RideSummary | null;
    suggestions: DriverSuggestion[];
    loading: boolean;
    assigning: boolean;
    allDrivers?: DriverSuggestion[]; // ✅ Tous les chauffeurs disponibles (fallback)
    loadingAllDrivers?: boolean; // ✅ Chargement de tous les chauffeurs
    isManualMode?: boolean; // ✅ Mode manuel (pas de suggestions)
    onClose: () => void;
    onAssign: (driverId: string) => void;
}

export function AssignDriverModal({
    visible,
    ride,
    suggestions,
    loading,
    assigning,
    allDrivers = [],
    loadingAllDrivers = false,
    isManualMode = false,
    onClose,
    onAssign,
}: AssignDriverModalProps) {
    // ✅ Vérifier que onAssign est bien défini
    const handleAssign = React.useCallback((driverId: string) => {
        console.log("[AssignDriverModal] handleAssign appelé:", { driverId, rideId: ride?.id, hasOnAssign: !!onAssign });
        if (!onAssign) {
            console.error("[AssignDriverModal] onAssign n'est pas défini!");
            return;
        }
        if (assigning) {
            console.log("[AssignDriverModal] Assignation déjà en cours, ignoré");
            return;
        }
        onAssign(driverId);
    }, [onAssign, ride?.id, assigning]);

    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <View style={styles.modalOverlay}>
                <View style={styles.modalCard}>
                    <Text style={styles.modalTitle}>Assigner un chauffeur</Text>

                    {ride && (
                        <View style={styles.modalRideInfo}>
                            <Text style={styles.modalRideClient}>{ride.client.name}</Text>
                            <Text style={styles.modalRideRoute} numberOfLines={1}>
                                {ride.route.pickup_address}
                            </Text>
                            <Text style={styles.modalRideRoute} numberOfLines={1}>
                                → {ride.route.dropoff_address}
                            </Text>
                        </View>
                    )}

                    {!loading && !loadingAllDrivers && (suggestions.length > 0 || allDrivers.length > 0) && (
                        <Text style={styles.modalSectionTitle}>
                            {isManualMode || suggestions.length === 0
                                ? "Sélectionner un chauffeur manuellement"
                                : "Choisir un chauffeur"}
                        </Text>
                    )}

                    {loading || loadingAllDrivers ? (
                        <View style={styles.modalLoading}>
                            <ActivityIndicator color={palette.modalButton} />
                            <Text style={styles.modalLoadingText}>
                                {loading ? "Chargement des suggestions..." : "Chargement des chauffeurs..."}
                            </Text>
                        </View>
                    ) : suggestions.length === 0 && allDrivers.length === 0 ? (
                        <View style={styles.modalEmptyContainer}>
                            <Text style={styles.modalEmpty}>
                                {ride?.driver?.id
                                    ? "Aucune suggestion disponible. Vous pouvez réassigner un chauffeur manuellement."
                                    : "Aucun chauffeur disponible pour cette course."}
                            </Text>
                        </View>
                    ) : (
                        <ScrollView
                            style={styles.modalDriverList}
                            nestedScrollEnabled
                            showsVerticalScrollIndicator={false}
                            keyboardShouldPersistTaps="handled"
                        >
                            {/* ✅ En mode manuel, ne jamais afficher les suggestions, seulement tous les chauffeurs */}
                            {((isManualMode || suggestions.length === 0) ? allDrivers : suggestions).map((suggestion: DriverSuggestion) => (
                                <Pressable
                                    key={suggestion.driver_id}
                                    style={({ pressed }) => [
                                        styles.modalDriverOption,
                                        pressed && !assigning && styles.modalDriverOptionPressed,
                                        assigning && styles.modalDriverOptionDisabled,
                                    ]}
                                    onPress={() => handleAssign(suggestion.driver_id)}
                                    disabled={assigning}
                                >
                                    <View style={styles.modalDriverInfo}>
                                        <Text style={styles.modalDriverName}>
                                            {suggestion.driver_name}
                                        </Text>
                                        <Text style={styles.modalDriverMeta}>
                                            Score: {suggestion.score.toFixed(2)}
                                            {suggestion.preferred_match && " • Préféré"}
                                            {suggestion.is_emergency && " • Urgence"}
                                        </Text>
                                        {suggestion.reason && (
                                            <Text style={styles.modalDriverReason}>
                                                {suggestion.reason}
                                            </Text>
                                        )}
                                    </View>
                                    <Ionicons
                                        name="chevron-forward"
                                        size={20}
                                        color={assigning ? palette.modalText : palette.modalButton}
                                    />
                                </Pressable>
                            ))}
                        </ScrollView>
                    )}

                    <View style={styles.modalActions}>
                        <Pressable
                            style={styles.modalCancelButton}
                            onPress={onClose}
                            disabled={assigning}
                        >
                            <Text style={styles.modalCancelText}>Annuler</Text>
                        </Pressable>
                        {ride && (
                            <Pressable
                                style={styles.modalViewDetailsButton}
                                onPress={() => {
                                    onClose();
                                    router.push({
                                        pathname: "/(enterprise)/ride-details",
                                        params: { rideId: ride.id },
                                    } as any);
                                }}
                                disabled={assigning}
                            >
                                <Text style={styles.modalViewDetailsText}>
                                    Voir la fiche complète
                                </Text>
                            </Pressable>
                        )}
                    </View>

                    {assigning && (
                        <View style={styles.modalAssigningOverlay}>
                            <ActivityIndicator color="#FFFFFF" size="large" />
                            <Text style={styles.modalAssigningText}>
                                Assignation en cours...
                            </Text>
                        </View>
                    )}
                </View>
            </View>
        </Modal>
    );
}

const styles = StyleSheet.create({
    modalOverlay: {
        flex: 1,
        backgroundColor: palette.modalOverlay,
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
    },
    modalCard: {
        width: "100%",
        maxWidth: 420,
        backgroundColor: palette.modalBackground,
        borderRadius: 24,
        padding: 24,
        borderWidth: 1,
        borderColor: palette.modalBorder,
        maxHeight: "80%",
        gap: 16,
        shadowColor: "rgba(15,54,43,0.15)",
        shadowOffset: { width: 0, height: 12 },
        shadowOpacity: 1,
        shadowRadius: 24,
        elevation: 8,
    },
    modalTitle: {
        color: palette.modalTitle,
        fontSize: 20,
        fontWeight: "700",
    },
    modalSubtitle: {
        color: palette.modalText,
        fontSize: 14,
        fontWeight: "400",
    },
    modalRideInfo: {
        backgroundColor: "rgba(10,127,89,0.06)",
        borderRadius: 18,
        padding: 18,
        borderWidth: 1.5,
        borderColor: "rgba(10,127,89,0.15)",
        marginBottom: 4,
    },
    modalRideClient: {
        color: palette.modalTitle,
        fontSize: 16,
        fontWeight: "700",
        marginBottom: 8,
        letterSpacing: 0.2,
    },
    modalRideRoute: {
        color: palette.modalText,
        fontSize: 13,
        marginTop: 6,
        lineHeight: 18,
    },
    modalSectionTitle: {
        color: palette.modalTitle,
        fontSize: 14,
        fontWeight: "700",
        textTransform: "uppercase",
        letterSpacing: 0.5,
        marginTop: 8,
        marginBottom: 12,
    },
    modalLoading: {
        alignItems: "center",
        paddingVertical: 40,
        gap: 12,
    },
    modalLoadingText: {
        color: palette.loadingText,
        fontSize: 14,
    },
    modalEmptyContainer: {
        paddingVertical: 40,
        paddingHorizontal: 16,
    },
    modalEmpty: {
        color: palette.modalText,
        fontSize: 14,
        textAlign: "center",
        lineHeight: 20,
    },
    modalDriverList: {
        maxHeight: 400,
        marginBottom: 16,
        marginTop: 4,
    },
    modalDriverOption: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        backgroundColor: "#FFFFFF",
        borderRadius: 16,
        paddingVertical: 10,
        paddingHorizontal: 14,
        marginBottom: 10,
        borderWidth: 1.5,
        borderColor: palette.surfaceBorder,
        shadowColor: "rgba(15,54,43,0.08)",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 1,
        shadowRadius: 6,
        elevation: 2,
    },
    modalDriverOptionPressed: {
        backgroundColor: "rgba(10,127,89,0.08)",
        borderColor: palette.modalButton,
    },
    modalDriverOptionDisabled: {
        opacity: 0.5,
    },
    modalDriverInfo: {
        flex: 1,
        marginRight: 12,
    },
    modalDriverName: {
        color: palette.modalTitle,
        fontSize: 15,
        fontWeight: "600",
        marginBottom: 2,
    },
    modalDriverMeta: {
        color: palette.modalText,
        fontSize: 12,
        marginBottom: 2,
    },
    modalDriverReason: {
        color: palette.loadingText,
        fontSize: 11,
        fontStyle: "italic",
        marginTop: 2,
    },
    modalActions: {
        flexDirection: "row",
        justifyContent: "flex-end",
        gap: 12,
    },
    modalCancelButton: {
        paddingHorizontal: 14,
        paddingVertical: 10,
    },
    modalCancelText: {
        color: palette.modalCancelText,
        fontWeight: "600",
        fontSize: 15,
    },
    modalViewDetailsButton: {
        backgroundColor: palette.modalButton,
        paddingHorizontal: 18,
        paddingVertical: 12,
        borderRadius: 14,
    },
    modalViewDetailsText: {
        color: palette.modalButtonText,
        fontWeight: "700",
        fontSize: 15,
    },
    modalAssigningOverlay: {
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0,0,0,0.8)",
        borderRadius: 24,
        alignItems: "center",
        justifyContent: "center",
    },
    modalAssigningText: {
        color: "#FFFFFF",
        marginTop: 12,
        fontSize: 14,
    },
});

