import React, { useState, useEffect } from "react";
import {
    View,
    Text,
    Modal,
    TouchableOpacity,
    FlatList,
    ActivityIndicator,
    Alert,
    StyleSheet,
    ScrollView,
    Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import {
    fetchPartnershipsForTransfer,
    proposeTransfer,
    Partnership,
} from "@/services/partnershipService";
import { RideSummary } from "@/types/enterpriseDispatch";
import { getLogger } from "@/utils/logger";

const log = getLogger("Transfer");
// Palette de couleurs cohérente avec l'app
const palette = {
    primary: "#0A7F59",
    text: "#15362B",
    textSecondary: "#5F7369",
    border: "rgba(15,54,43,0.08)",
    background: "#F5F7F6",
    danger: "#ef4444",
    success: "#10b981",
    warning: "#f59e0b",
    disabled: "#91A59D",
};

interface TransferRideModalProps {
    visible: boolean;
    onClose: () => void;
    ride: RideSummary | null;
    onSuccess: () => void;
}

/**
 * Modal pour transférer une course à une entreprise partenaire
 * Similaire au composant web TransferBookingModal.jsx
 */
export const TransferRideModal: React.FC<TransferRideModalProps> = ({
    visible,
    onClose,
    ride,
    onSuccess,
}) => {
    const [partnerships, setPartnerships] = useState<Partnership[]>([]);
    const [selectedPartnership, setSelectedPartnership] = useState<Partnership | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadingTransfer, setLoadingTransfer] = useState(false);
    const [error, setError] = useState("");
    const [showConfirmModal, setShowConfirmModal] = useState(false);

    // Charger les partenariats quand le modal s'ouvre
    useEffect(() => {
        if (visible) {
            loadPartnerships();
            setSelectedPartnership(null);
            setError("");
        }
    }, [visible]);

    useEffect(() => {
        log.info("selected partnership state", { selectedPartnership: selectedPartnership?.partner_company_name || "none" });
    }, [selectedPartnership]);

    /**
     * Charge la liste des partenariats disponibles pour le transfert
     */
    const loadPartnerships = async () => {
        try {
            setLoading(true);
            setError("");
            const data = await fetchPartnershipsForTransfer();
            log.info("partnerships loaded", { count: data.length });
            setPartnerships(data);

            if (data.length === 0) {
                setError("Aucun partenariat actif disponible pour le transfert");
            }
        } catch (err: any) {
            log.error("load partnerships failed", { error: err });
            setError(err?.response?.data?.error || "Impossible de charger les partenariats");
        } finally {
            setLoading(false);
        }
    };

    /**
     * Propose le transfert de la course au partenaire sélectionné
     */
    const handleTransfer = async () => {
        log.info("handleTransfer called", { selectedPartnership: selectedPartnership?.partner_company_name });

        if (!selectedPartnership) {
            if (Platform.OS === 'web') {
                window.alert("Veuillez sélectionner une entreprise partenaire");
            } else {
                Alert.alert("Erreur", "Veuillez sélectionner une entreprise partenaire");
            }
            return;
        }

        if (!ride) {
            if (Platform.OS === 'web') {
                window.alert("Aucune course sélectionnée");
            } else {
                Alert.alert("Erreur", "Aucune course sélectionnée");
            }
            return;
        }

        // Afficher le modal de confirmation
        setShowConfirmModal(true);
    };

    /**
     * Confirme et effectue le transfert
     */
    const confirmTransfer = async () => {
        if (!selectedPartnership || !ride) return;

        try {
            setLoadingTransfer(true);
            setError("");
            setShowConfirmModal(false);

            await proposeTransfer(selectedPartnership.id, ride.id);

            if (Platform.OS === 'web') {
                window.alert(`Course transférée avec succès à ${selectedPartnership.partner_company_name}`);
            } else {
                Alert.alert(
                    "Succès",
                    `Course transférée avec succès à ${selectedPartnership.partner_company_name}`,
                    [
                        {
                            text: "OK",
                            onPress: () => {
                                onSuccess();
                                onClose();
                            },
                        },
                    ]
                );
                return;
            }

            onSuccess();
            onClose();
        } catch (err: any) {
            log.error("transfer failed", { error: err });
            const errorMessage =
                err?.response?.data?.error || err?.message || "Erreur lors du transfert";
            setError(errorMessage);

            if (Platform.OS === 'web') {
                window.alert(errorMessage);
            } else {
                Alert.alert("Erreur", errorMessage);
            }
        } finally {
            setLoadingTransfer(false);
        }
    };

    /**
     * Sélectionne un partenariat
     */
    const handleSelectPartnership = (partnership: Partnership) => {
        log.info("partnership selected", { partnerCompanyName: partnership.partner_company_name });
        setSelectedPartnership(partnership);
        setError("");
    };

    /**
     * Rendu d'un item de partenariat
     */
    const renderPartnershipItem = ({ item }: { item: Partnership }) => {
        const isSelected = selectedPartnership?.id === item.id;

        // Formatage de la date en dehors du JSX pour éviter les problèmes de rendering
        let partnershipDetails = "Partenariat actif (date inconnue)";
        if (item.created_at) {
            const formattedDate = new Date(item.created_at).toLocaleDateString("fr-FR");
            partnershipDetails = "Partenariat actif depuis " + formattedDate;
        }

        return (
            <TouchableOpacity
                style={[styles.partnershipItem, isSelected && styles.partnershipItemSelected]}
                onPress={() => handleSelectPartnership(item)}
                activeOpacity={0.7}
                accessible={true}
                accessibilityRole="button"
            >
                <View style={[styles.radioContainer, styles.pointerEventsNone]}>
                    {isSelected ? (
                        <Ionicons name="radio-button-on" size={24} color={palette.primary} />
                    ) : (
                        <Ionicons name="radio-button-off" size={24} color={palette.textSecondary} />
                    )}
                </View>

                <View style={[styles.partnershipInfo, styles.pointerEventsNone]}>
                    <Text style={styles.partnershipName}>{item.partner_company_name}</Text>
                    <Text style={styles.partnershipDetails}>{partnershipDetails}</Text>
                </View>

                {isSelected && (
                    <View style={styles.pointerEventsNone}>
                        <Ionicons name="checkmark-circle" size={20} color={palette.primary} />
                    </View>
                )}
            </TouchableOpacity>
        );
    };

    /**
     * Contenu vide
     */
    const renderEmptyList = () => {
        return (
            <View style={styles.emptyContainer}>
                <Ionicons name="business-outline" size={48} color={palette.textSecondary} />
                <Text style={styles.emptyText}>Aucun partenariat disponible</Text>
                <Text style={styles.emptySubText}>
                    Créez des partenariats avec d'autres entreprises pour pouvoir transférer des courses
                </Text>
            </View>
        );
    };

    return (
        <Modal
            visible={visible}
            animationType="slide"
            transparent={false}
            onRequestClose={onClose}
        >
            <View style={styles.container}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                        <Ionicons name="close" size={28} color={palette.text} />
                    </TouchableOpacity>
                    <Text style={styles.title}>Transférer la course</Text>
                    <View style={styles.placeholder} />
                </View>

                {/* Course Info */}
                {ride && (
                    <View style={styles.rideInfo}>
                        <View style={styles.rideInfoRow}>
                            <View style={styles.rideInfoIcon}>
                                <Ionicons name="location" size={16} color={palette.primary} />
                            </View>
                            <Text style={styles.rideInfoText} numberOfLines={1}>
                                {ride.route.pickup_address}
                            </Text>
                        </View>
                        <View style={styles.rideInfoRow}>
                            <View style={styles.rideInfoIcon}>
                                <Ionicons name="navigate" size={16} color={palette.danger} />
                            </View>
                            <Text style={styles.rideInfoText} numberOfLines={1}>
                                {ride.route.dropoff_address}
                            </Text>
                        </View>
                        <View style={styles.rideInfoRow}>
                            <View style={styles.rideInfoIcon}>
                                <Ionicons name="time-outline" size={16} color={palette.textSecondary} />
                            </View>
                            <Text style={styles.rideInfoText}>
                                {ride.time.pickup_at ? new Date(ride.time.pickup_at).toLocaleString("fr-FR") : "Horaire non défini"}
                            </Text>
                        </View>
                    </View>
                )}

                {/* Instructions */}
                <View style={styles.instructions}>
                    <Text style={styles.instructionsText}>
                        Sélectionnez l'entreprise partenaire à qui transférer cette course :
                    </Text>
                </View>

                {/* Loading */}
                {loading && (
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator size="large" color={palette.primary} />
                        <Text style={styles.loadingText}>Chargement des partenariats...</Text>
                    </View>
                )}

                {/* Error */}
                {error && !loading && (
                    <View style={styles.errorContainer}>
                        <View style={styles.errorIcon}>
                            <Ionicons name="alert-circle" size={24} color={palette.danger} />
                        </View>
                        <Text style={styles.errorText}>{error}</Text>
                    </View>
                )}

                {/* Liste des partenariats */}
                {!loading && !error && (
                    <FlatList
                        data={partnerships}
                        keyExtractor={(item) => item.id}
                        renderItem={renderPartnershipItem}
                        style={styles.list}
                        contentContainerStyle={styles.listContainer}
                        ListEmptyComponent={renderEmptyList}
                        nestedScrollEnabled={true}
                    />
                )}

                {/* Boutons */}
                <View style={styles.footer}>
                    <TouchableOpacity
                        style={[styles.button, styles.cancelButton]}
                        onPress={onClose}
                        disabled={loadingTransfer}
                    >
                        <Text style={styles.cancelButtonText}>Annuler</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={[
                            styles.button,
                            styles.transferButton,
                            (!selectedPartnership || loadingTransfer) && styles.transferButtonDisabled,
                        ]}
                        onPress={handleTransfer}
                        disabled={!selectedPartnership || loadingTransfer}
                    >
                        {loadingTransfer ? (
                            <ActivityIndicator size="small" color="#fff" />
                        ) : (
                            <View style={styles.transferButtonContent}>
                                <View style={styles.transferButtonIcon}>
                                    <Ionicons name="swap-horizontal" size={20} color="#fff" />
                                </View>
                                <Text style={styles.transferButtonText}>Transférer</Text>
                            </View>
                        )}
                    </TouchableOpacity>
                </View>
            </View>

            {/* Modal de confirmation */}
            <Modal
                visible={showConfirmModal}
                transparent={true}
                animationType="fade"
                onRequestClose={() => setShowConfirmModal(false)}
            >
                <View style={styles.confirmModalOverlay}>
                    <View style={styles.confirmModalCard}>
                        <Text style={styles.confirmModalTitle}>Confirmer le transfert</Text>
                        <Text style={styles.confirmModalText}>
                            Transférer cette course à {selectedPartnership?.partner_company_name} ?
                        </Text>
                        <View style={styles.confirmModalActions}>
                            <TouchableOpacity
                                style={[styles.confirmButton, styles.cancelButton]}
                                onPress={() => setShowConfirmModal(false)}
                            >
                                <Text style={styles.cancelButtonText}>Annuler</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                                style={[styles.confirmButton, styles.transferButton]}
                                onPress={confirmTransfer}
                            >
                                <Text style={styles.transferButtonText}>Transférer</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            </Modal>
        </Modal>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: palette.background,
    },
    header: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: palette.border,
        backgroundColor: "#fff",
    },
    closeButton: {
        padding: 4,
    },
    title: {
        fontSize: 18,
        fontWeight: "600",
        color: palette.text,
    },
    placeholder: {
        width: 36,
    },
    rideInfo: {
        backgroundColor: "#fff",
        padding: 16,
        marginTop: 8,
        marginHorizontal: 16,
        borderRadius: 8,
    },
    rideInfoRow: {
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 8,
    },
    rideInfoIcon: {
        marginRight: 8,
    },
    rideInfoText: {
        flex: 1,
        fontSize: 14,
        color: palette.text,
    },
    instructions: {
        padding: 16,
    },
    instructionsText: {
        fontSize: 14,
        color: palette.textSecondary,
        lineHeight: 20,
    },
    loadingContainer: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
    },
    loadingText: {
        fontSize: 14,
        color: palette.textSecondary,
        marginTop: 12,
    },
    errorContainer: {
        flexDirection: "row",
        alignItems: "center",
        padding: 16,
        marginHorizontal: 16,
        backgroundColor: "#fee",
        borderRadius: 8,
        borderWidth: 1,
        borderColor: palette.danger,
    },
    errorIcon: {
        marginRight: 12,
    },
    errorText: {
        flex: 1,
        fontSize: 14,
        color: palette.danger,
    },
    list: {
        flex: 1,
    },
    listContainer: {
        padding: 16,
    },
    partnershipItem: {
        flexDirection: "row",
        alignItems: "center",
        padding: 16,
        backgroundColor: "#fff",
        borderRadius: 8,
        borderWidth: 1,
        borderColor: palette.border,
        marginBottom: 12,
    },
    partnershipItemSelected: {
        borderColor: palette.primary,
        borderWidth: 2,
        backgroundColor: "#f0f8ff",
    },
    radioContainer: {
        marginRight: 12,
    },
    partnershipInfo: {
        flex: 1,
    },
    partnershipName: {
        fontSize: 16,
        fontWeight: "600",
        color: palette.text,
        marginBottom: 4,
    },
    partnershipDetails: {
        fontSize: 12,
        color: palette.textSecondary,
    },
    emptyContainer: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        padding: 32,
    },
    emptyText: {
        fontSize: 16,
        fontWeight: "600",
        color: palette.text,
        marginTop: 16,
        marginBottom: 8,
    },
    emptySubText: {
        fontSize: 14,
        color: palette.textSecondary,
        textAlign: "center",
        lineHeight: 20,
    },
    footer: {
        flexDirection: "row",
        padding: 16,
        borderTopWidth: 1,
        borderTopColor: palette.border,
        backgroundColor: "#fff",
    },
    button: {
        flex: 1,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 14,
        borderRadius: 8,
        marginHorizontal: 6,
    },
    cancelButton: {
        backgroundColor: "#f5f5f5",
    },
    cancelButtonText: {
        fontSize: 16,
        fontWeight: "600",
        color: palette.text,
    },
    transferButton: {
        backgroundColor: palette.primary,
    },
    transferButtonDisabled: {
        backgroundColor: palette.disabled,
    },
    transferButtonText: {
        fontSize: 16,
        fontWeight: "600",
        color: "#fff",
    },
    transferButtonContent: {
        flexDirection: "row",
        alignItems: "center",
        marginLeft: 4,
    },
    transferButtonIcon: {
        marginRight: 8,
    },
    confirmModalOverlay: {
        flex: 1,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        justifyContent: "center",
        alignItems: "center",
        padding: 20,
    },
    confirmModalCard: {
        backgroundColor: "#fff",
        borderRadius: 16,
        padding: 24,
        width: "100%",
        maxWidth: 400,
    },
    confirmModalTitle: {
        fontSize: 20,
        fontWeight: "700",
        color: palette.text,
        marginBottom: 12,
    },
    confirmModalText: {
        fontSize: 16,
        color: palette.textSecondary,
        marginBottom: 24,
        lineHeight: 24,
    },
    confirmModalActions: {
        flexDirection: "row",
        justifyContent: "flex-end",
    },
    confirmButton: {
        paddingVertical: 12,
        paddingHorizontal: 20,
        borderRadius: 8,
        marginLeft: 12,
    },
    pointerEventsNone: {
        pointerEvents: "none",
    },
});
