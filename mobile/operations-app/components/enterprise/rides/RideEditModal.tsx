import React, { useState, useEffect } from "react";
import {
    Modal,
    View,
    Text,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    StyleSheet,
    Alert,
    KeyboardAvoidingView,
    Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import { RideSummary, RideEditPayload } from "@/types/enterpriseDispatch";
import { useRideEdit } from "@/hooks/useRideEdit";
import { AddressSelector } from "./AddressSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { NotesEditor } from "./NotesEditor";
import { shadowPresets } from "@/styles/shadowStyles";

const palette = {
    modalOverlay: "rgba(21,54,43,0.75)",
    modalBackground: "#FFFFFF",
    modalBorder: "rgba(15,54,43,0.12)",
    modalTitle: "#15362B",
    modalText: "#5F7369",
    modalButton: "#0A7F59",
    modalButtonText: "#FFFFFF",
    modalCancelText: "#5F7369",
    sectionBg: "rgba(10,127,89,0.06)",
    sectionBorder: "rgba(10,127,89,0.15)",
    divider: "rgba(15,54,43,0.08)",
};

interface RideEditModalProps {
    visible: boolean;
    ride: RideSummary | null;
    onClose: () => void;
    onSuccess?: () => Promise<void>;
}

export const RideEditModal: React.FC<RideEditModalProps> = ({
    visible,
    ride,
    onClose,
    onSuccess,
}) => {
    const { rideDetail, loading, loadingDetail, loadRideDetail, update, clear } =
        useRideEdit(onSuccess);

    const [pickupAddress, setPickupAddress] = useState("");
    const [dropoffAddress, setDropoffAddress] = useState("");
    const [scheduledTime, setScheduledTime] = useState<Date | null>(null);
    const [notes, setNotes] = useState("");
    const [priority, setPriority] = useState<"LOW" | "NORMAL" | "HIGH">("NORMAL");

    useEffect(() => {
        if (visible && ride) {
            loadRideDetail(ride.id);
        } else if (!visible) {
            clear();
        }
    }, [visible, ride, loadRideDetail, clear]);

    useEffect(() => {
        // ✅ Utiliser rideDetail si disponible, sinon utiliser les données de base de ride
        if (rideDetail?.summary) {
            const summary = rideDetail.summary;
            setPickupAddress(summary.route.pickup_address || "");
            setDropoffAddress(summary.route.dropoff_address || "");
            setScheduledTime(
                summary.time.pickup_at ? dayjs(summary.time.pickup_at).toDate() : null
            );
            setNotes(rideDetail.notes?.[0] || "");
            setPriority(summary.client.priority || "NORMAL");
        } else if (ride && !loadingDetail) {
            // ✅ Fallback : utiliser les données de base de ride (RideSummary) si rideDetail n'est pas disponible
            // Cela permet d'éditer même si le chargement des détails a échoué
            console.log("[RideEditModal] Utilisation des données de base de ride (fallback)");
            setPickupAddress(ride.route?.pickup_address || "");
            setDropoffAddress(ride.route?.dropoff_address || "");
            setScheduledTime(
                ride.time?.pickup_at ? dayjs(ride.time.pickup_at).toDate() : null
            );
            setNotes(""); // Les notes ne sont pas disponibles dans RideSummary
            setPriority(ride.client?.priority || "NORMAL");
        }
    }, [rideDetail, ride, loadingDetail]);

    const handleSave = async () => {
        if (!ride) return;

        const payload: RideEditPayload = {
            pickup_address: pickupAddress,
            dropoff_address: dropoffAddress,
            scheduled_time: scheduledTime ? (() => {
                // Utiliser format() au lieu de toISOString() pour préserver l'heure locale
                // Le backend utilise parse_local_naive qui attend un format ISO sans timezone
                const localISO = dayjs(scheduledTime).format("YYYY-MM-DDTHH:mm:ss");
                console.log("[RideEditModal] scheduledTime Date:", scheduledTime);
                console.log("[RideEditModal] scheduledTime dayjs:", dayjs(scheduledTime).format("DD.MM.YYYY HH:mm"));
                console.log("[RideEditModal] scheduledTime local ISO (sans timezone):", localISO);
                return localISO;
            })() : undefined,
            notes: notes || undefined,
            priority: priority,
        };

        try {
            await update(ride.id, payload);
            onClose();
        } catch (error) {
            // L'erreur est déjà gérée dans le hook
        }
    };

    const hasChanges = () => {
        // ✅ Utiliser rideDetail si disponible, sinon utiliser ride (RideSummary)
        const summary = rideDetail?.summary || ride;
        if (!summary) return false;
        // Comparer les dates en format local (sans timezone) pour éviter les problèmes de timezone
        const currentTimeStr = scheduledTime
            ? dayjs(scheduledTime).format("YYYY-MM-DDTHH:mm:ss")
            : null;
        const originalTimeStr = summary.time.pickup_at
            ? dayjs(summary.time.pickup_at).format("YYYY-MM-DDTHH:mm:ss")
            : null;
        return (
            pickupAddress !== (summary.route?.pickup_address || "") ||
            dropoffAddress !== (summary.route?.dropoff_address || "") ||
            currentTimeStr !== originalTimeStr ||
            notes !== (rideDetail?.notes?.[0] || "") ||
            priority !== (summary.client?.priority || "NORMAL")
        );
    };

    if (!ride) return null;

    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <KeyboardAvoidingView
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                style={styles.modalOverlay}
            >
                <View style={styles.modalCard}>
                    <View style={styles.modalHeader}>
                        <View>
                            <Text style={styles.modalTitle}>Modifier la course</Text>
                            <Text style={styles.modalSubtitle}>Course #{ride.id.slice(-4)}</Text>
                        </View>
                        <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                            <Ionicons name="close" size={24} color={palette.modalText} />
                        </TouchableOpacity>
                    </View>

                    {loadingDetail ? (
                        <View style={styles.loadingContainer}>
                            <ActivityIndicator color={palette.modalButton} size="large" />
                            <Text style={styles.loadingText}>Chargement des détails...</Text>
                        </View>
                    ) : (
                        <View style={styles.modalBody}>
                            {!rideDetail?.summary && ride && (
                                // ✅ Afficher un avertissement si les détails n'ont pas pu être chargés
                                // mais permettre quand même l'édition avec les données de base
                                <View style={styles.warningContainer}>
                                    <Ionicons name="warning-outline" size={20} color="#F59E0B" />
                                    <Text style={styles.warningText}>
                                        Les détails complets n'ont pas pu être chargés. Vous pouvez quand même modifier la course avec les informations disponibles.
                                    </Text>
                                </View>
                            )}
                            <ScrollView
                                style={styles.modalScroll}
                                contentContainerStyle={styles.modalContent}
                                showsVerticalScrollIndicator={false}
                                keyboardShouldPersistTaps="handled"
                            >
                                {/* Section Adresses */}
                                <View style={styles.section}>
                                    <View style={styles.sectionHeader}>
                                        <Ionicons name="location" size={18} color={palette.modalButton} />
                                        <Text style={styles.sectionTitle}>ADRESSES</Text>
                                    </View>
                                    <AddressSelector
                                        label="Adresse de départ"
                                        value={pickupAddress}
                                        onChange={(address) => setPickupAddress(address)}
                                        icon="location-outline"
                                    />
                                    <AddressSelector
                                        label="Adresse d'arrivée"
                                        value={dropoffAddress}
                                        onChange={(address) => setDropoffAddress(address)}
                                        icon="flag-outline"
                                    />
                                </View>

                                {/* Section Horaires */}
                                <View style={styles.section}>
                                    <View style={styles.sectionHeader}>
                                        <Ionicons name="time-outline" size={18} color={palette.modalButton} />
                                        <Text style={styles.sectionTitle}>HORAIRES</Text>
                                    </View>
                                    <TimeDatePicker
                                        label="Prise en charge"
                                        value={scheduledTime}
                                        onChange={setScheduledTime}
                                        mode="datetime"
                                    />
                                </View>

                                {/* Section Informations */}
                                <View style={styles.section}>
                                    <View style={styles.sectionHeader}>
                                        <Ionicons name="information-circle-outline" size={18} color={palette.modalButton} />
                                        <Text style={styles.sectionTitle}>INFORMATIONS</Text>
                                    </View>
                                    <NotesEditor
                                        label="Notes internes"
                                        value={notes}
                                        onChange={setNotes}
                                        placeholder="Ajouter des notes..."
                                    />
                                    <View style={styles.priorityContainer}>
                                        <Text style={styles.priorityLabel}>Priorité</Text>
                                        <View style={styles.priorityButtons}>
                                            {(["LOW", "NORMAL", "HIGH"] as const).map((p) => (
                                                <TouchableOpacity
                                                    key={p}
                                                    style={[
                                                        styles.priorityButton,
                                                        priority === p && styles.priorityButtonActive,
                                                    ]}
                                                    onPress={() => setPriority(p)}
                                                >
                                                    <Text
                                                        style={[
                                                            styles.priorityButtonText,
                                                            priority === p && styles.priorityButtonTextActive,
                                                        ]}
                                                    >
                                                        {p === "LOW" ? "Basse" : p === "NORMAL" ? "Normale" : "Haute"}
                                                    </Text>
                                                </TouchableOpacity>
                                            ))}
                                        </View>
                                    </View>
                                </View>
                            </ScrollView>
                        </View>
                    )}

                    <View style={styles.modalActions}>
                        <TouchableOpacity
                            style={styles.modalCancel}
                            onPress={onClose}
                            disabled={loading}
                        >
                            <Text style={styles.modalCancelText}>Annuler</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[
                                styles.modalSave,
                                (!hasChanges() || loading) && styles.modalSaveDisabled,
                            ]}
                            onPress={handleSave}
                            disabled={!hasChanges() || loading}
                        >
                            {loading ? (
                                <ActivityIndicator color="#FFFFFF" size="small" />
                            ) : (
                                <Text style={styles.modalSaveText}>Enregistrer</Text>
                            )}
                        </TouchableOpacity>
                    </View>
                </View>
            </KeyboardAvoidingView>
        </Modal>
    );
};

const styles = StyleSheet.create({
    modalOverlay: {
        flex: 1,
        backgroundColor: palette.modalOverlay,
        justifyContent: "center",
        alignItems: "center",
        padding: 20,
    },
    modalCard: {
        width: "100%",
        maxWidth: 500,
        maxHeight: "90%",
        backgroundColor: palette.modalBackground,
        borderRadius: 24,
        borderWidth: 1,
        borderColor: palette.modalBorder,
        flexDirection: "column",
        ...shadowPresets.large,
    },
    modalHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "flex-start",
        padding: 24,
        paddingBottom: 16,
        borderBottomWidth: 1,
        borderBottomColor: palette.divider,
    },
    modalTitle: {
        color: palette.modalTitle,
        fontSize: 20,
        fontWeight: "700",
    },
    modalSubtitle: {
        color: palette.modalText,
        fontSize: 13,
        marginTop: 4,
    },
    closeButton: {
        padding: 4,
    },
    loadingContainer: {
        padding: 40,
        alignItems: "center",
        gap: 12,
    },
    loadingText: {
        color: palette.modalText,
        fontSize: 14,
    },
    warningContainer: {
        flexDirection: "row",
        alignItems: "flex-start",
        gap: 12,
        padding: 20,
        margin: 20,
        backgroundColor: "rgba(245, 158, 11, 0.1)",
        borderRadius: 12,
        borderWidth: 1,
        borderColor: "rgba(245, 158, 11, 0.3)",
    },
    warningText: {
        flex: 1,
        color: "#92400E",
        fontSize: 13,
        lineHeight: 18,
    },
    modalBody: {
        flex: 1,
        minHeight: 200,
    },
    modalScroll: {
        flex: 1,
    },
    modalContent: {
        padding: 24,
        paddingBottom: 32,
        gap: 20,
        flexGrow: 1,
    },
    section: {
        backgroundColor: palette.sectionBg,
        borderRadius: 18,
        padding: 18,
        borderWidth: 1.5,
        borderColor: palette.sectionBorder,
        gap: 16,
    },
    sectionHeader: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        marginBottom: 4,
    },
    sectionTitle: {
        color: palette.modalTitle,
        fontSize: 13,
        fontWeight: "700",
        textTransform: "uppercase",
        letterSpacing: 0.5,
    },
    priorityContainer: {
        gap: 8,
    },
    priorityLabel: {
        color: palette.modalTitle,
        fontSize: 14,
        fontWeight: "600",
    },
    priorityButtons: {
        flexDirection: "row",
        gap: 8,
    },
    priorityButton: {
        flex: 1,
        paddingVertical: 10,
        paddingHorizontal: 14,
        borderRadius: 12,
        backgroundColor: palette.modalBackground,
        borderWidth: 1.5,
        borderColor: palette.modalBorder,
        alignItems: "center",
    },
    priorityButtonActive: {
        backgroundColor: palette.modalButton,
        borderColor: palette.modalButton,
    },
    priorityButtonText: {
        color: palette.modalText,
        fontSize: 13,
        fontWeight: "600",
    },
    priorityButtonTextActive: {
        color: palette.modalButtonText,
    },
    modalActions: {
        flexDirection: "row",
        justifyContent: "flex-end",
        gap: 12,
        padding: 24,
        paddingTop: 16,
        borderTopWidth: 1,
        borderTopColor: palette.divider,
    },
    modalCancel: {
        paddingHorizontal: 18,
        paddingVertical: 12,
    },
    modalCancelText: {
        color: palette.modalCancelText,
        fontSize: 15,
        fontWeight: "600",
    },
    modalSave: {
        backgroundColor: palette.modalButton,
        paddingHorizontal: 24,
        paddingVertical: 12,
        borderRadius: 14,
        minWidth: 120,
        alignItems: "center",
    },
    modalSaveDisabled: {
        backgroundColor: "rgba(10,127,89,0.4)",
    },
    modalSaveText: {
        color: palette.modalButtonText,
        fontSize: 15,
        fontWeight: "700",
    },
});

