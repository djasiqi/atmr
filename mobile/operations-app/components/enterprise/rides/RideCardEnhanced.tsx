import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { EnterpriseCard } from "../cards/EnterpriseCard";
import { RideSummary } from "@/types/enterpriseDispatch";
import dayjs from "dayjs";

// ✅ Couleurs de statut alignées avec le dashboard et RideSnippetCard
const statusColors = {
    pending: {
        bg: "#fef3c7", // --warning-bg
        text: "#f59e0b", // --warning-primary
    },
    accepted: {
        bg: "#dbeafe", // --info-bg
        text: "#3b82f6", // --info-primary
    },
    assigned: {
        bg: "#dbeafe", // --info-bg
        text: "#3b82f6", // --info-primary
    },
    en_route: {
        bg: "#fef3c7", // --warning-bg (orange)
        text: "#f59e0b", // --warning-primary (orange)
    },
    in_progress: {
        bg: "#fef3c7", // --warning-bg (orange)
        text: "#f59e0b", // --warning-primary (orange)
    },
    completed: {
        bg: "#dcfce7", // --success-bg
        text: "#16a34a", // --success-primary
    },
    return_completed: {
        bg: "#dcfce7", // --success-bg
        text: "#16a34a", // --success-primary
    },
    cancelled: {
        bg: "#f3f4f6", // --bg-hover
        text: "#6b7280", // --text-tertiary
    },
    canceled: {
        bg: "#f3f4f6", // --bg-hover
        text: "#6b7280", // --text-tertiary
    },
};

// ✅ Fonction pour obtenir les couleurs selon le statut
const getStatusColors = (status?: string) => {
    if (!status) return statusColors.pending;
    const normalizedStatus = status.toLowerCase();
    return statusColors[normalizedStatus as keyof typeof statusColors] || statusColors.pending;
};

const palette = {
    time: "#0A7F59",
    timeUndefined: "#F59E0B",
    client: "#15362B",
    text: "#5F7369",
    accent: "#0A7F59",
    chevron: "#91A59D",
    actionBg: "rgba(10,127,89,0.08)",
    actionBorder: "rgba(15,54,43,0.12)",
    divider: "rgba(15,54,43,0.08)",
};

interface RideCardEnhancedProps {
    ride: RideSummary;
    expanded?: boolean;
    onToggle?: () => void;
    onEdit?: (ride: RideSummary) => void;
    onSchedule?: (ride: RideSummary) => void;
    onAssign?: (ride: RideSummary) => void;
    onUrgent?: (ride: RideSummary) => void;
    onViewDetails?: (ride: RideSummary) => void;
    onCancel?: (ride: RideSummary) => void;
    style?: { marginBottom?: number };
}

export const RideCardEnhanced: React.FC<RideCardEnhancedProps> = ({
    ride,
    expanded = false,
    onToggle,
    onEdit,
    onSchedule,
    onAssign,
    onUrgent,
    onViewDetails,
    onCancel,
    style,
}) => {
    // Calculer la largeur de l'écran pour adapter le nombre de colonnes et les marges
    const screenWidth = Dimensions.get("window").width;
    const isTablet = screenWidth >= 768; // iPad et tablettes
    const isSmallScreen = screenWidth <= 375; // iPhone SE et petits écrans
    const buttonsPerRow = isTablet ? 6 : 3; // 6 boutons sur tablette, 3 sur mobile
    const buttonFlexBasis = `${100 / buttonsPerRow}%` as const;

    // Ajuster les marges pour les petits écrans
    const headerGap = isSmallScreen ? 6 : 12;
    const timeContainerWidth = isSmallScreen ? 40 : 54;

    const pickupTime = ride.time.pickup_at
        ? (() => {
            const time = dayjs(ride.time.pickup_at);
            // Si l'heure est à minuit (00:00), c'est probablement une heure non définie
            // Afficher une icône d'horloge au lieu de "00h00"
            if (time.hour() === 0 && time.minute() === 0) {
                return null; // null affichera l'icône d'horloge
            }
            return time.format("HH[h]mm");
        })()
        : null;

    // ✅ Ne pas calculer le retard pour les courses terminées
    const isCompleted = ride.status === "completed" || ride.status === "return_completed";

    const delayMinutes =
        !isCompleted && ride.time.pickup_at && ride.driver?.name
            ? (() => {
                const scheduled = dayjs(ride.time.pickup_at);
                const now = dayjs();
                if (scheduled.isValid() && scheduled.isBefore(now)) {
                    return Math.max(0, now.diff(scheduled, "minute"));
                }
                return null;
            })()
            : null;

    const isDelayed = delayMinutes !== null && delayMinutes > 0;
    const isLongDelay = isDelayed && delayMinutes >= 15;

    return (
        <EnterpriseCard style={[styles.card, style]}>
            <TouchableOpacity
                style={[styles.header, { gap: headerGap }]}
                onPress={onToggle}
                activeOpacity={0.85}
            >
                <View style={[styles.timeContainer, { width: timeContainerWidth }]}>
                    {pickupTime ? (
                        <Text style={styles.time}>{pickupTime}</Text>
                    ) : (
                        <Ionicons name="time-outline" size={18} color={palette.timeUndefined} />
                    )}
                </View>
                <View style={styles.clientContainer}>
                    <Text style={styles.client} numberOfLines={1}>
                        {ride.client.name}
                    </Text>
                </View>
                <View style={styles.badgeContainer}>
                    {(() => {
                        // ✅ Obtenir les couleurs selon le statut (aligné avec le dashboard)
                        const statusColors = getStatusColors(ride.status);

                        // ✅ Si terminée, toujours afficher les couleurs de statut (vert) SANS retard
                        if (isCompleted) {
                            return (
                                <View
                                    style={[
                                        styles.badge,
                                        {
                                            backgroundColor: statusColors.bg,
                                            borderColor: statusColors.text + "40", // 40 = 25% opacity en hex
                                        },
                                    ]}
                                >
                                    <Text
                                        style={[
                                            styles.badgeText,
                                            {
                                                color: statusColors.text,
                                            },
                                        ]}
                                        numberOfLines={1}
                                    >
                                        {ride.driver?.name ? ride.driver.name.toUpperCase() : "Terminée"}
                                    </Text>
                                </View>
                            );
                        }

                        // ✅ Sinon, logique normale avec retard et couleurs de statut
                        if (ride.driver?.name) {
                            // ✅ Utiliser les couleurs de retard si présent, sinon les couleurs de statut
                            const badgeBg = isDelayed
                                ? isLongDelay
                                    ? "#fee2e2" // --error-bg pour retard long (rouge)
                                    : "#fef3c7" // --warning-bg pour retard court (jaune)
                                : statusColors.bg;
                            const badgeTextColor = isDelayed
                                ? isLongDelay
                                    ? "#ef4444" // --error-primary pour retard long (rouge)
                                    : "#f59e0b" // --warning-primary pour retard court (jaune)
                                : statusColors.text;

                            return (
                                <View
                                    style={[
                                        styles.badge,
                                        {
                                            backgroundColor: badgeBg,
                                            borderColor: badgeTextColor + "40", // 40 = 25% opacity en hex
                                        },
                                        isDelayed && isLongDelay && styles.badgeDanger,
                                        isDelayed && !isLongDelay && styles.badgeWarning,
                                    ]}
                                >
                                    <Text
                                        style={[
                                            styles.badgeText,
                                            {
                                                color: badgeTextColor,
                                            },
                                            isDelayed && isLongDelay && styles.badgeTextDanger,
                                            isDelayed && !isLongDelay && styles.badgeTextWarning,
                                        ]}
                                        numberOfLines={1}
                                    >
                                        {isDelayed
                                            ? `${ride.driver.name.toUpperCase()} ${delayMinutes}min`
                                            : ride.driver.name.toUpperCase()}
                                    </Text>
                                </View>
                            );
                        }

                        // ✅ Non assignée - utiliser les couleurs de statut "pending"
                        return (
                            <View
                                style={[
                                    styles.badge,
                                    {
                                        backgroundColor: statusColors.bg,
                                        borderColor: statusColors.text + "40", // 40 = 25% opacity en hex
                                    },
                                    styles.badgeUnassigned,
                                ]}
                            >
                                <Text
                                    style={[
                                        styles.badgeText,
                                        {
                                            color: statusColors.text,
                                        },
                                    ]}
                                >
                                    Non Assigné
                                </Text>
                            </View>
                        );
                    })()}
                </View>
                <View style={styles.chevronContainer}>
                    <Ionicons
                        name={expanded ? "chevron-up-outline" : "chevron-down-outline"}
                        size={16}
                        color={palette.chevron}
                    />
                </View>
            </TouchableOpacity>

            {expanded && (
                <>
                    <View style={styles.route}>
                        <View style={styles.routeRow}>
                            <Ionicons name="location-outline" size={16} color={palette.accent} />
                            <Text style={styles.routeText} numberOfLines={1}>
                                {ride.route.pickup_address}
                            </Text>
                        </View>
                        <View style={styles.routeDivider} />
                        <View style={styles.routeRow}>
                            <Ionicons name="flag-outline" size={16} color={palette.accent} />
                            <Text style={styles.routeText} numberOfLines={1}>
                                {ride.route.dropoff_address}
                            </Text>
                        </View>
                    </View>

                    <View style={styles.divider} />

                    <View style={styles.actions}>
                        {onEdit && (
                            <TouchableOpacity
                                style={styles.actionButton}
                                onPress={() => onEdit(ride)}
                            >
                                <Ionicons name="create-outline" size={16} color={palette.accent} />
                                <Text style={styles.actionText}>Éditer</Text>
                            </TouchableOpacity>
                        )}
                        {onSchedule && (
                            <TouchableOpacity
                                style={[styles.actionButton, { flexBasis: buttonFlexBasis }]}
                                onPress={() => onSchedule(ride)}
                            >
                                <Ionicons name="time-outline" size={16} color={palette.accent} />
                                <Text style={styles.actionText}>Planifier</Text>
                            </TouchableOpacity>
                        )}
                        {onAssign && !ride.driver?.name && (
                            <TouchableOpacity
                                style={[styles.actionButton, { flexBasis: buttonFlexBasis }]}
                                onPress={() => onAssign(ride)}
                            >
                                <Ionicons name="person-add-outline" size={16} color={palette.accent} />
                                <Text style={styles.actionText}>Assigner</Text>
                            </TouchableOpacity>
                        )}
                        {onUrgent && ride.client.priority !== "HIGH" && (
                            <TouchableOpacity
                                style={[styles.actionButton, { flexBasis: buttonFlexBasis }]}
                                onPress={() => onUrgent(ride)}
                            >
                                <Ionicons name="flame-outline" size={16} color="#F59E0B" />
                                <Text style={[styles.actionText, { color: "#F59E0B" }]}>Urgent</Text>
                            </TouchableOpacity>
                        )}
                        {onViewDetails && (
                            <TouchableOpacity
                                style={[styles.actionButton, { flexBasis: buttonFlexBasis }]}
                                onPress={() => onViewDetails(ride)}
                            >
                                <Ionicons name="open-outline" size={16} color={palette.text} />
                                <Text style={[styles.actionText, { color: palette.text }]}>Détails</Text>
                            </TouchableOpacity>
                        )}
                        {onCancel && ride.status !== "cancelled" && (
                            <TouchableOpacity
                                style={[styles.actionButton, { flexBasis: buttonFlexBasis }]}
                                onPress={() => onCancel(ride)}
                            >
                                <Ionicons name="close-circle-outline" size={16} color="#EF4444" />
                                <Text style={[styles.actionText, { color: "#EF4444" }]}>Annuler</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                </>
            )}
        </EnterpriseCard>
    );
};

const styles = StyleSheet.create({
    card: {
        padding: 16,
        gap: 12,
    },
    header: {
        flexDirection: "row",
        alignItems: "center",
        gap: 12,
    },
    timeContainer: {
        width: 54,
        alignItems: "flex-start",
    },
    time: {
        color: palette.time,
        fontWeight: "700",
        fontSize: 16,
        letterSpacing: 0.2,
    },
    clientContainer: {
        flex: 1,
        minWidth: 0,
        marginHorizontal: 0, // Pas de marge supplémentaire, le gap du header suffit
    },
    client: {
        color: palette.client,
        fontSize: 16,
        fontWeight: "600",
    },
    badgeContainer: {
        flex: 1,
        alignItems: "flex-end",
    },
    chevronContainer: {
        width: 24,
        alignItems: "center",
        marginLeft: 2,
    },
    badge: {
        borderRadius: 12,
        paddingHorizontal: 10,
        paddingVertical: 6,
        maxWidth: 120,
        borderWidth: 1,
    },
    badgeText: {
        fontSize: 11,
        fontWeight: "600",
        color: palette.client,
        letterSpacing: 0.3,
        textTransform: "uppercase",
    },
    badgeUnassigned: {
        backgroundColor: "rgba(95,115,105,0.12)",
        borderColor: palette.actionBorder,
    },
    badgeAssigned: {
        backgroundColor: "rgba(10,127,89,0.12)",
        borderColor: palette.actionBorder,
    },
    badgeWarning: {
        backgroundColor: "rgba(251,191,36,0.12)",
        borderColor: "#F59E0B",
    },
    badgeDanger: {
        backgroundColor: "rgba(239,68,68,0.12)",
        borderColor: "#EF4444",
    },
    badgeTextWarning: {
        color: "#F59E0B",
    },
    badgeTextDanger: {
        color: "#EF4444",
    },
    route: {
        marginTop: 14,
        gap: 4,
    },
    routeRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        marginBottom: 4,
    },
    routeDivider: {
        height: 1,
        backgroundColor: palette.divider,
        marginVertical: 6,
        marginLeft: 24,
    },
    routeText: {
        flex: 1,
        color: palette.text,
        fontSize: 13,
        maxWidth: 280,
    },
    divider: {
        height: 1,
        backgroundColor: palette.divider,
        marginVertical: 4,
    },
    actions: {
        flexDirection: "row",
        flexWrap: "wrap",
        gap: 8,
        marginTop: 4,
    },
    actionButton: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 12,
        backgroundColor: palette.actionBg,
        borderWidth: 1,
        borderColor: palette.actionBorder,
    },
    actionText: {
        color: palette.accent,
        fontSize: 12,
        fontWeight: "600",
    },
});

