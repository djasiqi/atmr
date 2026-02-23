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
import { createShadow } from "@/styles/shadowStyles";

const log = getLogger("Transfer");

const BRAND = "#00796B";
const palette = {
    primary: BRAND,
    primaryLight: "rgba(0,121,107,0.08)",
    primaryMedium: "rgba(0,121,107,0.15)",
    text: "#1E293B",
    textSecondary: "#64748B",
    textMuted: "#94A3B8",
    border: "rgba(0,121,107,0.08)",
    background: "#f4f7fc",
    card: "#FFFFFF",
    danger: "#dc3545",
    dangerLight: "rgba(220,53,69,0.06)",
    dangerBorder: "rgba(220,53,69,0.15)",
    success: "#10b981",
    disabled: "rgba(0,121,107,0.4)",
};

interface TransferRideModalProps {
    visible: boolean;
    onClose: () => void;
    ride: RideSummary | null;
    onSuccess: () => void;
}

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

    useEffect(() => {
        if (visible) {
            loadPartnerships();
            setSelectedPartnership(null);
            setError("");
        }
    }, [visible]);

    useEffect(() => {
        log.info("selected partnership state", {
            selectedPartnership: selectedPartnership?.partner_company_name || "none",
        });
    }, [selectedPartnership]);

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

    const handleTransfer = async () => {
        log.info("handleTransfer called", {
            selectedPartnership: selectedPartnership?.partner_company_name,
        });

        if (!selectedPartnership) {
            if (Platform.OS === "web") {
                window.alert("Veuillez sélectionner une entreprise partenaire");
            } else {
                Alert.alert("Erreur", "Veuillez sélectionner une entreprise partenaire");
            }
            return;
        }

        if (!ride) {
            if (Platform.OS === "web") {
                window.alert("Aucune course sélectionnée");
            } else {
                Alert.alert("Erreur", "Aucune course sélectionnée");
            }
            return;
        }

        setShowConfirmModal(true);
    };

    const confirmTransfer = async () => {
        if (!selectedPartnership || !ride) return;

        try {
            setLoadingTransfer(true);
            setError("");
            setShowConfirmModal(false);

            await proposeTransfer(selectedPartnership.id, ride.id);

            if (Platform.OS === "web") {
                window.alert(
                    `Course transférée avec succès à ${selectedPartnership.partner_company_name}`,
                );
            } else {
                Alert.alert(
                    "Succès",
                    `Course transférée avec succès à ${selectedPartnership.partner_company_name}`,
                    [{ text: "OK", onPress: () => { onSuccess(); onClose(); } }],
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

            if (Platform.OS === "web") {
                window.alert(errorMessage);
            } else {
                Alert.alert("Erreur", errorMessage);
            }
        } finally {
            setLoadingTransfer(false);
        }
    };

    const handleSelectPartnership = (partnership: Partnership) => {
        log.info("partnership selected", {
            partnerCompanyName: partnership.partner_company_name,
        });
        setSelectedPartnership(partnership);
        setError("");
    };

    const renderPartnershipItem = ({ item }: { item: Partnership }) => {
        const isSelected = selectedPartnership?.id === item.id;

        let partnershipDetails = "Partenariat actif";
        if (item.created_at) {
            const formattedDate = new Date(item.created_at).toLocaleDateString("fr-FR");
            partnershipDetails = "Depuis le " + formattedDate;
        }

        return (
            <TouchableOpacity
                style={[styles.partnershipItem, isSelected && styles.partnershipItemSelected]}
                onPress={() => handleSelectPartnership(item)}
                activeOpacity={0.7}
                accessible={true}
                accessibilityRole="button"
            >
                <View style={[styles.partnershipRadio, styles.pointerEventsNone]}>
                    <View
                        style={[
                            styles.radioOuter,
                            isSelected && styles.radioOuterSelected,
                        ]}
                    >
                        {isSelected && <View style={styles.radioInner} />}
                    </View>
                </View>

                <View style={[styles.partnershipBody, styles.pointerEventsNone]}>
                    <View style={styles.partnershipIconWrap}>
                        <Ionicons name="business" size={18} color={isSelected ? palette.primary : palette.textSecondary} />
                    </View>
                    <View style={styles.partnershipTexts}>
                        <Text style={[styles.partnershipName, isSelected && styles.partnershipNameSelected]}>
                            {item.partner_company_name}
                        </Text>
                        <Text style={styles.partnershipDetails}>{partnershipDetails}</Text>
                    </View>
                </View>

                {isSelected && (
                    <View style={[styles.partnershipCheck, styles.pointerEventsNone]}>
                        <Ionicons name="checkmark-circle" size={22} color={palette.primary} />
                    </View>
                )}
            </TouchableOpacity>
        );
    };

    const renderEmptyList = () => (
        <View style={styles.emptyContainer}>
            <View style={styles.emptyIconWrap}>
                <Ionicons name="business-outline" size={32} color={palette.primary} />
            </View>
            <Text style={styles.emptyText}>Aucun partenariat disponible</Text>
            <Text style={styles.emptySubText}>
                {"Créez des partenariats avec d'autres entreprises pour pouvoir transférer des courses"}
            </Text>
        </View>
    );

    const formattedPickupTime = ride?.time.pickup_at
        ? new Date(ride.time.pickup_at).toLocaleString("fr-FR", {
              day: "2-digit",
              month: "long",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
          })
        : "Horaire non défini";

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
                    <TouchableOpacity onPress={onClose} style={styles.closeButton} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
                        <Ionicons name="close" size={22} color={palette.textSecondary} />
                    </TouchableOpacity>
                    <View style={styles.headerCenter}>
                        <View style={styles.headerIconWrap}>
                            <Ionicons name="swap-horizontal" size={16} color={palette.primary} />
                        </View>
                        <Text style={styles.headerTitle}>Transférer la course</Text>
                    </View>
                    <View style={styles.headerSpacer} />
                </View>

                {/* Route card */}
                {ride && (
                    <View style={styles.routeCard}>
                        <View style={styles.routeTimeline}>
                            <View style={styles.routeDotPickup} />
                            <View style={styles.routeLine} />
                            <View style={styles.routeDotDropoff} />
                        </View>
                        <View style={styles.routeAddresses}>
                            <View style={styles.routeAddressBlock}>
                                <Text style={styles.routeLabel}>Prise en charge</Text>
                                <Text style={styles.routeAddress} numberOfLines={2}>
                                    {ride.route.pickup_address}
                                </Text>
                            </View>
                            <View style={styles.routeAddressBlock}>
                                <Text style={styles.routeLabel}>Destination</Text>
                                <Text style={styles.routeAddress} numberOfLines={2}>
                                    {ride.route.dropoff_address}
                                </Text>
                            </View>
                        </View>
                        <View style={styles.routeTimeRow}>
                            <View style={styles.routeTimeBadge}>
                                <Ionicons name="time-outline" size={13} color={palette.primary} />
                                <Text style={styles.routeTimeText}>{formattedPickupTime}</Text>
                            </View>
                        </View>
                    </View>
                )}

                {/* Section title */}
                <View style={styles.sectionHeader}>
                    <Text style={styles.sectionTitle}>Entreprise partenaire</Text>
                    <Text style={styles.sectionSubtitle}>
                        {"Sélectionnez l'entreprise à qui transférer cette course"}
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
                {!!error && !loading && (
                    <View style={styles.errorContainer}>
                        <View style={styles.errorIconWrap}>
                            <Ionicons name="alert-circle" size={20} color={palette.danger} />
                        </View>
                        <Text style={styles.errorText}>{error}</Text>
                    </View>
                )}

                {/* Partnership list */}
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

                {/* Footer */}
                <View style={styles.footer}>
                    <TouchableOpacity
                        style={styles.cancelButton}
                        onPress={onClose}
                        disabled={loadingTransfer}
                    >
                        <Text style={styles.cancelButtonText}>Annuler</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={[
                            styles.transferButton,
                            (!selectedPartnership || loadingTransfer) && styles.transferButtonDisabled,
                        ]}
                        onPress={handleTransfer}
                        disabled={!selectedPartnership || loadingTransfer}
                    >
                        {loadingTransfer ? (
                            <ActivityIndicator size="small" color="#fff" />
                        ) : (
                            <>
                                <Ionicons name="swap-horizontal" size={18} color="#fff" />
                                <Text style={styles.transferButtonText}>Transférer</Text>
                            </>
                        )}
                    </TouchableOpacity>
                </View>
            </View>

            {/* Confirmation modal */}
            <Modal
                visible={showConfirmModal}
                transparent={true}
                animationType="fade"
                onRequestClose={() => setShowConfirmModal(false)}
            >
                <View style={styles.confirmOverlay}>
                    <View style={styles.confirmCard}>
                        <View style={styles.confirmIconWrap}>
                            <Ionicons name="swap-horizontal" size={28} color={palette.primary} />
                        </View>
                        <Text style={styles.confirmTitle}>Confirmer le transfert</Text>
                        <Text style={styles.confirmText}>
                            {"Transférer cette course à "}
                            <Text style={styles.confirmHighlight}>
                                {selectedPartnership?.partner_company_name}
                            </Text>
                            {" ?"}
                        </Text>
                        <Text style={styles.confirmHint}>
                            {"Cette action notifiera l'entreprise partenaire."}
                        </Text>
                        <View style={styles.confirmActions}>
                            <TouchableOpacity
                                style={styles.confirmCancelBtn}
                                onPress={() => setShowConfirmModal(false)}
                            >
                                <Text style={styles.confirmCancelText}>Annuler</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                                style={styles.confirmTransferBtn}
                                onPress={confirmTransfer}
                            >
                                <Ionicons name="checkmark" size={18} color="#fff" />
                                <Text style={styles.confirmTransferText}>Confirmer</Text>
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

    // ── Header ──
    header: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        paddingHorizontal: 16,
        paddingVertical: 14,
        backgroundColor: palette.card,
        borderBottomWidth: 1,
        borderBottomColor: palette.border,
        ...createShadow({
            shadowColor: "#000",
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.04,
            shadowRadius: 8,
            elevation: 2,
        }),
    },
    closeButton: {
        width: 36,
        height: 36,
        borderRadius: 10,
        backgroundColor: palette.background,
        alignItems: "center",
        justifyContent: "center",
    },
    headerCenter: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
    },
    headerIconWrap: {
        width: 28,
        height: 28,
        borderRadius: 8,
        backgroundColor: palette.primaryLight,
        alignItems: "center",
        justifyContent: "center",
    },
    headerTitle: {
        fontSize: 17,
        fontWeight: "700",
        color: palette.text,
        letterSpacing: -0.3,
    },
    headerSpacer: {
        width: 36,
    },

    // ── Route card ──
    routeCard: {
        backgroundColor: palette.card,
        marginHorizontal: 16,
        marginTop: 16,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: palette.border,
        padding: 16,
        ...createShadow({
            shadowColor: "#000",
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.04,
            shadowRadius: 8,
            elevation: 2,
        }),
    },
    routeTimeline: {
        position: "absolute",
        left: 24,
        top: 24,
        bottom: 56,
        width: 12,
        alignItems: "center",
    },
    routeDotPickup: {
        width: 10,
        height: 10,
        borderRadius: 5,
        backgroundColor: palette.primary,
    },
    routeLine: {
        flex: 1,
        width: 2,
        backgroundColor: palette.border,
        marginVertical: 2,
    },
    routeDotDropoff: {
        width: 10,
        height: 10,
        borderRadius: 5,
        backgroundColor: palette.danger,
    },
    routeAddresses: {
        marginLeft: 28,
        gap: 16,
    },
    routeAddressBlock: {
        gap: 2,
    },
    routeLabel: {
        fontSize: 11,
        fontWeight: "700",
        color: palette.textMuted,
        textTransform: "uppercase",
        letterSpacing: 0.5,
    },
    routeAddress: {
        fontSize: 14,
        fontWeight: "500",
        color: palette.text,
        lineHeight: 20,
    },
    routeTimeRow: {
        marginTop: 14,
        marginLeft: 28,
    },
    routeTimeBadge: {
        flexDirection: "row",
        alignItems: "center",
        alignSelf: "flex-start",
        gap: 6,
        backgroundColor: palette.primaryLight,
        paddingHorizontal: 10,
        paddingVertical: 5,
        borderRadius: 8,
    },
    routeTimeText: {
        fontSize: 13,
        fontWeight: "600",
        color: palette.primary,
    },

    // ── Section header ──
    sectionHeader: {
        paddingHorizontal: 20,
        paddingTop: 20,
        paddingBottom: 8,
    },
    sectionTitle: {
        fontSize: 16,
        fontWeight: "700",
        color: palette.text,
        letterSpacing: -0.2,
    },
    sectionSubtitle: {
        fontSize: 13,
        color: palette.textSecondary,
        marginTop: 2,
        lineHeight: 18,
    },

    // ── Loading ──
    loadingContainer: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        gap: 12,
    },
    loadingText: {
        fontSize: 14,
        color: palette.textSecondary,
    },

    // ── Error ──
    errorContainer: {
        flexDirection: "row",
        alignItems: "center",
        marginHorizontal: 16,
        marginTop: 8,
        padding: 14,
        backgroundColor: palette.dangerLight,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: palette.dangerBorder,
        gap: 12,
    },
    errorIconWrap: {
        width: 32,
        height: 32,
        borderRadius: 8,
        backgroundColor: "rgba(220,53,69,0.1)",
        alignItems: "center",
        justifyContent: "center",
    },
    errorText: {
        flex: 1,
        fontSize: 13,
        fontWeight: "500",
        color: palette.text,
        lineHeight: 18,
    },

    // ── Partnership list ──
    list: {
        flex: 1,
    },
    listContainer: {
        padding: 16,
        gap: 10,
    },
    partnershipItem: {
        flexDirection: "row",
        alignItems: "center",
        padding: 14,
        backgroundColor: palette.card,
        borderRadius: 14,
        borderWidth: 1,
        borderColor: palette.border,
        ...createShadow({
            shadowColor: "#000",
            shadowOffset: { width: 0, height: 1 },
            shadowOpacity: 0.03,
            shadowRadius: 4,
            elevation: 1,
        }),
    },
    partnershipItemSelected: {
        borderColor: palette.primary,
        borderWidth: 2,
        backgroundColor: palette.primaryLight,
        ...createShadow({
            shadowColor: BRAND,
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.12,
            shadowRadius: 8,
            elevation: 3,
        }),
    },
    partnershipRadio: {
        marginRight: 12,
    },
    radioOuter: {
        width: 22,
        height: 22,
        borderRadius: 11,
        borderWidth: 2,
        borderColor: palette.textMuted,
        alignItems: "center",
        justifyContent: "center",
    },
    radioOuterSelected: {
        borderColor: palette.primary,
    },
    radioInner: {
        width: 12,
        height: 12,
        borderRadius: 6,
        backgroundColor: palette.primary,
    },
    partnershipBody: {
        flex: 1,
        flexDirection: "row",
        alignItems: "center",
        gap: 12,
    },
    partnershipIconWrap: {
        width: 36,
        height: 36,
        borderRadius: 10,
        backgroundColor: palette.primaryLight,
        alignItems: "center",
        justifyContent: "center",
    },
    partnershipTexts: {
        flex: 1,
    },
    partnershipName: {
        fontSize: 15,
        fontWeight: "600",
        color: palette.text,
    },
    partnershipNameSelected: {
        color: palette.primary,
        fontWeight: "700",
    },
    partnershipDetails: {
        fontSize: 12,
        color: palette.textSecondary,
        marginTop: 2,
    },
    partnershipCheck: {
        marginLeft: 8,
    },

    // ── Empty state ──
    emptyContainer: {
        alignItems: "center",
        paddingVertical: 40,
        paddingHorizontal: 32,
    },
    emptyIconWrap: {
        width: 56,
        height: 56,
        borderRadius: 16,
        backgroundColor: palette.primaryLight,
        alignItems: "center",
        justifyContent: "center",
        marginBottom: 16,
    },
    emptyText: {
        fontSize: 15,
        fontWeight: "700",
        color: palette.text,
        marginBottom: 6,
    },
    emptySubText: {
        fontSize: 13,
        color: palette.textSecondary,
        textAlign: "center",
        lineHeight: 18,
    },

    // ── Footer ──
    footer: {
        flexDirection: "row",
        padding: 16,
        gap: 10,
        backgroundColor: palette.card,
        borderTopWidth: 1,
        borderTopColor: palette.border,
        ...createShadow({
            shadowColor: "#000",
            shadowOffset: { width: 0, height: -2 },
            shadowOpacity: 0.04,
            shadowRadius: 8,
            elevation: 2,
        }),
    },
    cancelButton: {
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 14,
        borderRadius: 12,
        backgroundColor: palette.background,
        borderWidth: 1,
        borderColor: palette.border,
    },
    cancelButtonText: {
        fontSize: 15,
        fontWeight: "600",
        color: palette.text,
    },
    transferButton: {
        flex: 1,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 14,
        borderRadius: 12,
        backgroundColor: palette.primary,
        gap: 8,
        ...createShadow({
            shadowColor: BRAND,
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.2,
            shadowRadius: 6,
            elevation: 3,
        }),
    },
    transferButtonDisabled: {
        backgroundColor: palette.disabled,
        ...createShadow({
            shadowColor: "transparent",
            shadowOffset: { width: 0, height: 0 },
            shadowOpacity: 0,
            shadowRadius: 0,
            elevation: 0,
        }),
    },
    transferButtonText: {
        fontSize: 15,
        fontWeight: "700",
        color: "#fff",
    },

    // ── Confirm modal ──
    confirmOverlay: {
        flex: 1,
        backgroundColor: "rgba(5,22,16,0.55)",
        justifyContent: "center",
        alignItems: "center",
        padding: 24,
    },
    confirmCard: {
        width: "100%",
        maxWidth: 380,
        backgroundColor: palette.card,
        borderRadius: 20,
        padding: 28,
        alignItems: "center",
        ...createShadow({
            shadowColor: "rgba(10,127,89,0.2)",
            shadowOffset: { width: 0, height: 12 },
            shadowOpacity: 1,
            shadowRadius: 28,
            elevation: 12,
        }),
    },
    confirmIconWrap: {
        width: 56,
        height: 56,
        borderRadius: 16,
        backgroundColor: palette.primaryLight,
        alignItems: "center",
        justifyContent: "center",
        marginBottom: 16,
    },
    confirmTitle: {
        fontSize: 18,
        fontWeight: "700",
        color: palette.text,
        letterSpacing: -0.3,
        marginBottom: 8,
    },
    confirmText: {
        fontSize: 15,
        color: palette.textSecondary,
        textAlign: "center",
        lineHeight: 22,
    },
    confirmHighlight: {
        fontWeight: "700",
        color: palette.primary,
    },
    confirmHint: {
        fontSize: 12,
        color: palette.textMuted,
        textAlign: "center",
        marginTop: 4,
        marginBottom: 24,
    },
    confirmActions: {
        flexDirection: "row",
        gap: 10,
        width: "100%",
    },
    confirmCancelBtn: {
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 13,
        borderRadius: 12,
        backgroundColor: palette.background,
        borderWidth: 1,
        borderColor: palette.border,
    },
    confirmCancelText: {
        fontSize: 15,
        fontWeight: "600",
        color: palette.text,
    },
    confirmTransferBtn: {
        flex: 1,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 13,
        borderRadius: 12,
        backgroundColor: palette.primary,
        gap: 6,
        ...createShadow({
            shadowColor: BRAND,
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.2,
            shadowRadius: 6,
            elevation: 3,
        }),
    },
    confirmTransferText: {
        fontSize: 15,
        fontWeight: "700",
        color: "#fff",
    },

    pointerEventsNone: {
        pointerEvents: "none",
    },
});
