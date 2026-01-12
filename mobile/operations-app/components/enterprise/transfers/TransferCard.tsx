import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Transfer } from "@/services/partnershipService";
import { shadowPresets } from "@/styles/shadowStyles";

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

interface TransferCardProps {
    transfer: Transfer;
    onAccept: (transfer: Transfer) => void;
    onReject: (transfer: Transfer) => void;
    onViewDetails?: (transfer: Transfer) => void;
    type: "incoming" | "outgoing";
}

/**
 * Carte pour afficher un transfert (entrant ou sortant)
 * Similaire au composant web mais adapté pour React Native
 */
export const TransferCard: React.FC<TransferCardProps> = ({
    transfer,
    onAccept,
    onReject,
    onViewDetails,
    type,
}) => {
    const isPending = transfer.status === "PENDING";
    const isAccepted = transfer.status === "ACCEPTED";
    const isRejected = transfer.status === "REJECTED";
    const isIncoming = type === "incoming";

    // Déterminer le statut et sa couleur
    const getStatusConfig = () => {
        if (isPending) {
            return {
                label: "En attente",
                icon: "time-outline" as const,
                color: palette.warning || "#f59e0b",
                bg: "#fef3c7",
            };
        }
        if (isAccepted) {
            return {
                label: "Accepté",
                icon: "checkmark-circle" as const,
                color: palette.success || "#10b981",
                bg: "#d1fae5",
            };
        }
        return {
            label: "Refusé",
            icon: "close-circle" as const,
            color: palette.danger || "#ef4444",
            bg: "#fee",
        };
    };

    const statusConfig = getStatusConfig();

    // Formater la date
    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleString("fr-FR", {
            day: "2-digit",
            month: "short",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    // Obtenir le nom de l'entreprise partenaire
    const partnerName = transfer.partnership
        ? isIncoming
            ? transfer.partnership.owner_company_name
            : transfer.partnership.partner_company_name
        : "Entreprise partenaire";

    return (
        <View style={styles.card}>
            {/* Header avec statut */}
            <View style={styles.header}>
                <View style={styles.headerLeft}>
                    <Ionicons
                        name={isIncoming ? "arrow-down-circle" : "arrow-up-circle"}
                        size={20}
                        color={isIncoming ? palette.primary : palette.textSecondary}
                    />
                    <Text style={styles.typeLabel}>
                        {isIncoming ? "Transfert reçu" : "Transfert proposé"}
                    </Text>
                </View>
                <View style={[styles.statusBadge, { backgroundColor: statusConfig.bg }]}>
                    <Ionicons name={statusConfig.icon} size={14} color={statusConfig.color} />
                    <Text style={[styles.statusText, { color: statusConfig.color }]}>
                        {statusConfig.label}
                    </Text>
                </View>
            </View>

            {/* Informations sur le partenaire */}
            <View style={styles.partnerSection}>
                <Ionicons name="business-outline" size={16} color={palette.textSecondary} />
                <Text style={styles.partnerLabel}>
                    {isIncoming ? `De : ${partnerName}` : `À : ${partnerName}`}
                </Text>
            </View>

            {/* Informations sur la course */}
            {transfer.booking && (
                <View style={styles.bookingSection}>
                    <View style={styles.bookingRow}>
                        <Ionicons name="location" size={14} color={palette.primary} />
                        <Text style={styles.bookingText} numberOfLines={1}>
                            {transfer.booking.pickup_location}
                        </Text>
                    </View>
                    <View style={styles.bookingRow}>
                        <Ionicons name="navigate" size={14} color={palette.danger} />
                        <Text style={styles.bookingText} numberOfLines={1}>
                            {transfer.booking.dropoff_location}
                        </Text>
                    </View>
                    {transfer.booking.scheduled_time && (
                        <View style={styles.bookingRow}>
                            <Ionicons name="time-outline" size={14} color={palette.textSecondary} />
                            <Text style={styles.bookingText}>
                                {formatDate(transfer.booking.scheduled_time)}
                            </Text>
                        </View>
                    )}
                    {transfer.booking.client_name && (
                        <View style={styles.bookingRow}>
                            <Ionicons name="person-outline" size={14} color={palette.textSecondary} />
                            <Text style={styles.bookingText}>{transfer.booking.client_name}</Text>
                        </View>
                    )}
                </View>
            )}

            {/* Raison du refus (si applicable) */}
            {isRejected && transfer.reason && (
                <View style={styles.reasonSection}>
                    <Ionicons name="information-circle-outline" size={16} color={palette.textSecondary} />
                    <Text style={styles.reasonText}>Raison : {transfer.reason}</Text>
                </View>
            )}

            {/* Date de proposition */}
            <View style={styles.dateSection}>
                <Ionicons name="calendar-outline" size={12} color={palette.textSecondary} />
                <Text style={styles.dateText}>Proposé le {formatDate(transfer.proposed_at)}</Text>
            </View>

            {/* Boutons d'action (uniquement pour les transferts entrants en attente) */}
            {isIncoming && isPending && (
                <View style={styles.actionsRow}>
                    <TouchableOpacity
                        style={[styles.actionButton, styles.rejectButton]}
                        onPress={() => onReject(transfer)}
                        activeOpacity={0.7}
                    >
                        <Ionicons name="close-circle" size={18} color="#fff" />
                        <Text style={styles.actionButtonText}>Refuser</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={[styles.actionButton, styles.acceptButton]}
                        onPress={() => onAccept(transfer)}
                        activeOpacity={0.7}
                    >
                        <Ionicons name="checkmark-circle" size={18} color="#fff" />
                        <Text style={styles.actionButtonText}>Accepter</Text>
                    </TouchableOpacity>
                </View>
            )}

            {/* Bouton "Voir détails" (optionnel) */}
            {onViewDetails && (
                <TouchableOpacity
                    style={styles.detailsButton}
                    onPress={() => onViewDetails(transfer)}
                    activeOpacity={0.7}
                >
                    <Text style={styles.detailsButtonText}>Voir la course</Text>
                    <Ionicons name="chevron-forward" size={16} color={palette.primary} />
                </TouchableOpacity>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    card: {
        backgroundColor: "#fff",
        borderRadius: 12,
        padding: 16,
        marginBottom: 12,
        borderWidth: 1,
        borderColor: palette.border,
        ...shadowPresets.small, // ✅ Utilise createShadow() pour compatibilité web/native
    },
    header: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 12,
    },
    headerLeft: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
    },
    typeLabel: {
        fontSize: 14,
        fontWeight: "600",
        color: palette.text,
    },
    statusBadge: {
        flexDirection: "row",
        alignItems: "center",
        gap: 4,
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 12,
    },
    statusText: {
        fontSize: 12,
        fontWeight: "600",
    },
    partnerSection: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        marginBottom: 12,
        paddingBottom: 12,
        borderBottomWidth: 1,
        borderBottomColor: palette.border,
    },
    partnerLabel: {
        fontSize: 14,
        color: palette.textSecondary,
    },
    partnerName: {
        fontWeight: "600",
        color: palette.text,
    },
    bookingSection: {
        gap: 8,
        marginBottom: 12,
    },
    bookingRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
    },
    bookingText: {
        flex: 1,
        fontSize: 13,
        color: palette.text,
    },
    reasonSection: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        padding: 10,
        backgroundColor: "#fee",
        borderRadius: 8,
        marginBottom: 12,
    },
    reasonText: {
        flex: 1,
        fontSize: 13,
        color: palette.danger,
        fontStyle: "italic",
    },
    dateSection: {
        flexDirection: "row",
        alignItems: "center",
        gap: 6,
        marginBottom: 12,
    },
    dateText: {
        fontSize: 12,
        color: palette.textSecondary,
    },
    actionsRow: {
        flexDirection: "row",
        gap: 10,
        marginTop: 4,
    },
    actionButton: {
        flex: 1,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 12,
        borderRadius: 8,
        gap: 6,
    },
    acceptButton: {
        backgroundColor: palette.success || "#10b981",
    },
    rejectButton: {
        backgroundColor: palette.danger || "#ef4444",
    },
    actionButtonText: {
        fontSize: 14,
        fontWeight: "600",
        color: "#fff",
    },
    detailsButton: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 10,
        gap: 6,
        borderTopWidth: 1,
        borderTopColor: palette.border,
        marginTop: 8,
    },
    detailsButtonText: {
        fontSize: 14,
        fontWeight: "600",
        color: palette.primary,
    },
});
