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
    Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import type { RideSummary, DriverSuggestion } from "@/types/enterpriseDispatch";
import { shadowPresets } from "@/styles/shadowStyles";
import { getLogger } from "@/utils/logger";

const log = getLogger("AssignDriver");

const BRAND = "#00796B";
const BRAND_DARK = "#00695C";

const palette = {
    overlay: "rgba(15,23,42,0.6)",
    card: "#FFFFFF",
    border: "rgba(0,121,107,0.10)",
    borderLight: "rgba(0,121,107,0.06)",
    text: "#1E293B",
    secondary: "#64748B",
    placeholder: "#94A3B8",
    accent: BRAND,
    accentDark: BRAND_DARK,
    accentBg: "rgba(0,121,107,0.06)",
    accentBorder: "rgba(0,121,107,0.15)",
    surface: "#F8FAFB",
    danger: "#dc3545",
    white: "#FFFFFF",
} as const;

const containerShadow =
    Platform.OS === "web"
        ? { boxShadow: "0 24px 48px rgba(0,0,0,0.18)" }
        : {
              shadowColor: "#000",
              shadowOffset: { width: 0, height: 24 },
              shadowOpacity: 0.18,
              shadowRadius: 48,
              elevation: 12,
          };

interface AssignDriverModalProps {
    visible: boolean;
    ride: RideSummary | null;
    suggestions: DriverSuggestion[];
    loading: boolean;
    assigning: boolean;
    allDrivers?: DriverSuggestion[];
    loadingAllDrivers?: boolean;
    isManualMode?: boolean;
    onClose: () => void;
    onAssign: (driverId: string) => void;
}

function getInitials(name: string): string {
    const parts = name.trim().split(/\s+/);
    if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    return (parts[0]?.[0] ?? "?").toUpperCase();
}

function formatScore(score: number): string {
    if (score >= 0.9) return "Excellent";
    if (score >= 0.7) return "Très bon";
    if (score >= 0.5) return "Bon";
    return "Compatible";
}

function getScoreColor(score: number): string {
    if (score >= 0.9) return "#059669";
    if (score >= 0.7) return BRAND;
    if (score >= 0.5) return "#D97706";
    return palette.secondary;
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
    const handleAssign = React.useCallback(
        (driverId: string) => {
            log.info("handleAssign called", {
                driverId,
                rideId: ride?.id,
                hasOnAssign: !!onAssign,
            });
            if (!onAssign || assigning) return;
            onAssign(driverId);
        },
        [onAssign, ride?.id, assigning]
    );

    const driverList =
        isManualMode || suggestions.length === 0 ? allDrivers : suggestions;
    const isLoading = loading || loadingAllDrivers;
    const isEmpty = !isLoading && driverList.length === 0;

    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <View style={s.overlay}>
                <View style={s.card}>
                    {/* Header */}
                    <View style={s.header}>
                        <View style={s.headerLeft}>
                            <View style={s.headerIcon}>
                                <Ionicons
                                    name="person-add"
                                    size={18}
                                    color={palette.accent}
                                />
                            </View>
                            <Text style={s.title}>Assigner un chauffeur</Text>
                        </View>
                        <TouchableOpacity
                            style={s.closeBtn}
                            onPress={onClose}
                            disabled={assigning}
                            hitSlop={8}
                        >
                            <Ionicons
                                name="close"
                                size={20}
                                color={palette.secondary}
                            />
                        </TouchableOpacity>
                    </View>

                    {/* Ride info */}
                    {ride && (
                        <View style={s.rideCard}>
                            <View style={s.rideHeader}>
                                <Ionicons
                                    name="car-outline"
                                    size={14}
                                    color={palette.accent}
                                />
                                <Text style={s.rideLabel}>COURSE</Text>
                            </View>
                            <Text style={s.rideClient}>{ride.client.name}</Text>

                            {/* Timeline route */}
                            <View style={s.routeWrap}>
                                <View style={s.timeline}>
                                    <View style={s.dotPickup} />
                                    <View style={s.connector} />
                                    <View style={s.dotDropoff} />
                                </View>
                                <View style={s.routeAddresses}>
                                    <Text
                                        style={s.routeText}
                                        numberOfLines={1}
                                    >
                                        {ride.route.pickup_address}
                                    </Text>
                                    <Text
                                        style={[s.routeText, s.routeTextDropoff]}
                                        numberOfLines={1}
                                    >
                                        {ride.route.dropoff_address}
                                    </Text>
                                </View>
                            </View>
                        </View>
                    )}

                    {/* Section title */}
                    {!isLoading && driverList.length > 0 && (
                        <Text style={s.sectionTitle}>
                            {isManualMode || suggestions.length === 0
                                ? "Sélectionner un chauffeur"
                                : "Suggestions"}
                        </Text>
                    )}

                    {/* Content */}
                    {isLoading ? (
                        <View style={s.loadingWrap}>
                            <ActivityIndicator color={palette.accent} size="small" />
                            <Text style={s.loadingText}>
                                Chargement des chauffeurs...
                            </Text>
                        </View>
                    ) : isEmpty ? (
                        <View style={s.emptyWrap}>
                            <View style={s.emptyIcon}>
                                <Ionicons
                                    name="people-outline"
                                    size={28}
                                    color={palette.placeholder}
                                />
                            </View>
                            <Text style={s.emptyTitle}>Aucun chauffeur disponible</Text>
                            <Text style={s.emptySubtitle}>
                                {ride?.driver?.id
                                    ? "Vous pouvez réassigner un chauffeur manuellement depuis la fiche complète."
                                    : "Aucun chauffeur n'est disponible pour cette course actuellement."}
                            </Text>
                        </View>
                    ) : (
                        <ScrollView
                            style={s.driverList}
                            nestedScrollEnabled
                            showsVerticalScrollIndicator={false}
                            keyboardShouldPersistTaps="handled"
                        >
                            {driverList.map((driver: DriverSuggestion) => (
                                <Pressable
                                    key={driver.driver_id}
                                    style={({ pressed }) => [
                                        s.driverRow,
                                        pressed && !assigning && s.driverRowPressed,
                                        assigning && s.driverRowDisabled,
                                    ]}
                                    onPress={() => handleAssign(driver.driver_id)}
                                    disabled={assigning}
                                >
                                    {/* Avatar initials */}
                                    <View style={s.avatar}>
                                        <Text style={s.avatarText}>
                                            {getInitials(driver.driver_name)}
                                        </Text>
                                    </View>

                                    {/* Driver info */}
                                    <View style={s.driverInfo}>
                                        <View style={s.driverNameRow}>
                                            <Text
                                                style={s.driverName}
                                                numberOfLines={1}
                                            >
                                                {driver.driver_name}
                                            </Text>
                                            {driver.preferred_match && (
                                                <Ionicons
                                                    name="star"
                                                    size={12}
                                                    color="#F59E0B"
                                                />
                                            )}
                                        </View>

                                        {/* Tags */}
                                        <View style={s.tagRow}>
                                            <View
                                                style={[
                                                    s.tag,
                                                    {
                                                        backgroundColor: `${getScoreColor(driver.score)}12`,
                                                        borderColor: `${getScoreColor(driver.score)}30`,
                                                    },
                                                ]}
                                            >
                                                <Text
                                                    style={[
                                                        s.tagText,
                                                        {
                                                            color: getScoreColor(
                                                                driver.score
                                                            ),
                                                        },
                                                    ]}
                                                >
                                                    {formatScore(driver.score)}
                                                </Text>
                                            </View>
                                            {driver.is_emergency && (
                                                <View style={s.tagEmergency}>
                                                    <Ionicons
                                                        name="flash"
                                                        size={10}
                                                        color="#DC2626"
                                                    />
                                                    <Text style={s.tagEmergencyText}>
                                                        Urgence
                                                    </Text>
                                                </View>
                                            )}
                                        </View>

                                        {driver.reason ? (
                                            <Text
                                                style={s.driverReason}
                                                numberOfLines={1}
                                            >
                                                {driver.reason}
                                            </Text>
                                        ) : null}
                                    </View>

                                    {/* Assign arrow */}
                                    <View style={s.assignArrow}>
                                        <Ionicons
                                            name="arrow-forward-circle"
                                            size={22}
                                            color={
                                                assigning
                                                    ? palette.placeholder
                                                    : palette.accent
                                            }
                                        />
                                    </View>
                                </Pressable>
                            ))}
                        </ScrollView>
                    )}

                    {/* Footer */}
                    <View style={s.footer}>
                        <TouchableOpacity
                            style={s.cancelBtn}
                            onPress={onClose}
                            disabled={assigning}
                        >
                            <Text style={s.cancelText}>Annuler</Text>
                        </TouchableOpacity>
                        {ride && (
                            <TouchableOpacity
                                style={s.detailsBtn}
                                onPress={() => {
                                    onClose();
                                    router.push({
                                        pathname: "/(enterprise)/ride-details",
                                        params: { rideId: ride.id },
                                    } as any);
                                }}
                                disabled={assigning}
                            >
                                <Ionicons
                                    name="document-text-outline"
                                    size={15}
                                    color={palette.white}
                                />
                                <Text style={s.detailsText}>Fiche complète</Text>
                            </TouchableOpacity>
                        )}
                    </View>

                    {/* Assigning overlay */}
                    {assigning && (
                        <View style={s.assigningOverlay}>
                            <View style={s.assigningCard}>
                                <ActivityIndicator color={palette.white} size="large" />
                                <Text style={s.assigningText}>
                                    Assignation en cours...
                                </Text>
                            </View>
                        </View>
                    )}
                </View>
            </View>
        </Modal>
    );
}

const s = StyleSheet.create({
    // ─── Overlay & Card ───
    overlay: {
        flex: 1,
        backgroundColor: palette.overlay,
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
    },
    card: {
        width: "100%",
        maxWidth: 420,
        backgroundColor: palette.card,
        borderRadius: 20,
        maxHeight: "85%",
        overflow: "hidden",
        ...containerShadow,
    },

    // ─── Header ───
    header: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        paddingHorizontal: 20,
        paddingTop: 20,
        paddingBottom: 16,
        borderBottomWidth: 1,
        borderBottomColor: palette.borderLight,
    },
    headerLeft: {
        flexDirection: "row",
        alignItems: "center",
        gap: 10,
        flex: 1,
    },
    headerIcon: {
        width: 36,
        height: 36,
        borderRadius: 10,
        backgroundColor: palette.accentBg,
        alignItems: "center",
        justifyContent: "center",
    },
    title: {
        fontSize: 17,
        fontWeight: "700",
        color: palette.text,
        letterSpacing: -0.2,
    },
    closeBtn: {
        width: 32,
        height: 32,
        borderRadius: 8,
        backgroundColor: palette.surface,
        alignItems: "center",
        justifyContent: "center",
    },

    // ─── Ride Card ───
    rideCard: {
        marginHorizontal: 16,
        marginTop: 16,
        backgroundColor: palette.accentBg,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: palette.accentBorder,
        overflow: "hidden",
    },
    rideHeader: {
        flexDirection: "row",
        alignItems: "center",
        gap: 6,
        paddingHorizontal: 14,
        paddingVertical: 8,
        backgroundColor: "rgba(0,121,107,0.04)",
        borderBottomWidth: 1,
        borderBottomColor: palette.accentBorder,
    },
    rideLabel: {
        fontSize: 10,
        fontWeight: "700",
        color: palette.accent,
        letterSpacing: 0.5,
    },
    rideClient: {
        fontSize: 15,
        fontWeight: "700",
        color: palette.text,
        paddingHorizontal: 14,
        paddingTop: 10,
        paddingBottom: 6,
        letterSpacing: -0.1,
    },
    routeWrap: {
        flexDirection: "row",
        paddingHorizontal: 14,
        paddingBottom: 12,
        gap: 10,
    },
    timeline: {
        alignItems: "center",
        paddingTop: 4,
        width: 14,
    },
    dotPickup: {
        width: 10,
        height: 10,
        borderRadius: 5,
        borderWidth: 2.5,
        borderColor: palette.accent,
        backgroundColor: palette.white,
    },
    connector: {
        width: 2,
        height: 14,
        backgroundColor: "rgba(0,121,107,0.18)",
        marginVertical: 2,
    },
    dotDropoff: {
        width: 10,
        height: 10,
        borderRadius: 5,
        borderWidth: 2.5,
        borderColor: palette.text,
        backgroundColor: palette.text,
    },
    routeAddresses: {
        flex: 1,
        gap: 8,
    },
    routeText: {
        fontSize: 13,
        fontWeight: "500",
        color: palette.text,
        lineHeight: 18,
    },
    routeTextDropoff: {
        color: palette.secondary,
    },

    // ─── Section Title ───
    sectionTitle: {
        fontSize: 11,
        fontWeight: "700",
        color: palette.secondary,
        textTransform: "uppercase",
        letterSpacing: 0.5,
        paddingHorizontal: 20,
        paddingTop: 16,
        paddingBottom: 8,
    },

    // ─── Loading ───
    loadingWrap: {
        alignItems: "center",
        paddingVertical: 40,
        gap: 12,
    },
    loadingText: {
        color: palette.placeholder,
        fontSize: 13,
        fontWeight: "500",
    },

    // ─── Empty ───
    emptyWrap: {
        alignItems: "center",
        paddingVertical: 32,
        paddingHorizontal: 24,
    },
    emptyIcon: {
        width: 52,
        height: 52,
        borderRadius: 14,
        backgroundColor: palette.surface,
        alignItems: "center",
        justifyContent: "center",
        marginBottom: 14,
    },
    emptyTitle: {
        fontSize: 15,
        fontWeight: "700",
        color: palette.text,
        textAlign: "center",
        marginBottom: 6,
    },
    emptySubtitle: {
        fontSize: 13,
        color: palette.secondary,
        textAlign: "center",
        lineHeight: 19,
        maxWidth: 280,
    },

    // ─── Driver List ───
    driverList: {
        maxHeight: 340,
        paddingHorizontal: 16,
    },
    driverRow: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: palette.card,
        borderRadius: 12,
        padding: 12,
        marginBottom: 8,
        borderWidth: 1,
        borderColor: palette.border,
        ...shadowPresets.small,
    },
    driverRowPressed: {
        backgroundColor: palette.accentBg,
        borderColor: palette.accent,
    },
    driverRowDisabled: {
        opacity: 0.45,
    },

    // ─── Avatar ───
    avatar: {
        width: 40,
        height: 40,
        borderRadius: 12,
        backgroundColor: palette.accentBg,
        borderWidth: 1,
        borderColor: palette.accentBorder,
        alignItems: "center",
        justifyContent: "center",
        marginRight: 12,
    },
    avatarText: {
        fontSize: 14,
        fontWeight: "700",
        color: palette.accent,
        letterSpacing: 0.3,
    },

    // ─── Driver Info ───
    driverInfo: {
        flex: 1,
        marginRight: 8,
    },
    driverNameRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 5,
        marginBottom: 4,
    },
    driverName: {
        fontSize: 14,
        fontWeight: "600",
        color: palette.text,
        letterSpacing: -0.1,
        flexShrink: 1,
    },
    tagRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 6,
        flexWrap: "wrap",
    },
    tag: {
        paddingHorizontal: 7,
        paddingVertical: 2,
        borderRadius: 6,
        borderWidth: 1,
    },
    tagText: {
        fontSize: 10,
        fontWeight: "700",
        letterSpacing: 0.2,
    },
    tagEmergency: {
        flexDirection: "row",
        alignItems: "center",
        gap: 3,
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 6,
        backgroundColor: "rgba(220,38,38,0.08)",
        borderWidth: 1,
        borderColor: "rgba(220,38,38,0.20)",
    },
    tagEmergencyText: {
        fontSize: 10,
        fontWeight: "700",
        color: "#DC2626",
    },
    driverReason: {
        fontSize: 11,
        color: palette.placeholder,
        marginTop: 3,
        fontStyle: "italic",
    },
    assignArrow: {
        marginLeft: 4,
    },

    // ─── Footer ───
    footer: {
        flexDirection: "row",
        alignItems: "center",
        gap: 10,
        paddingHorizontal: 16,
        paddingVertical: 14,
        borderTopWidth: 1,
        borderTopColor: palette.borderLight,
    },
    cancelBtn: {
        flex: 1,
        paddingVertical: 12,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: palette.border,
        backgroundColor: palette.card,
        alignItems: "center",
    },
    cancelText: {
        fontSize: 14,
        fontWeight: "600",
        color: palette.secondary,
    },
    detailsBtn: {
        flex: 1,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        paddingVertical: 12,
        borderRadius: 12,
        backgroundColor: palette.accent,
        ...(Platform.OS === "web"
            ? { boxShadow: "0 4px 8px rgba(0,121,107,0.2)" }
            : {
                  shadowColor: BRAND,
                  shadowOffset: { width: 0, height: 4 },
                  shadowOpacity: 0.2,
                  shadowRadius: 8,
                  elevation: 4,
              }),
    },
    detailsText: {
        fontSize: 14,
        fontWeight: "700",
        color: palette.white,
        letterSpacing: 0.1,
    },

    // ─── Assigning Overlay ───
    assigningOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.7)",
        borderRadius: 20,
        alignItems: "center",
        justifyContent: "center",
    },
    assigningCard: {
        alignItems: "center",
        gap: 14,
        padding: 24,
        borderRadius: 16,
        backgroundColor: "rgba(0,0,0,0.5)",
    },
    assigningText: {
        color: palette.white,
        fontSize: 14,
        fontWeight: "600",
    },
});
