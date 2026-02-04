import React, { useState } from "react";
import {
    View,
    Text,
    Modal,
    TouchableOpacity,
    StyleSheet,
    ScrollView,
    Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

type CancelOption = {
    id: string;
    label: string;
    description: string;
    isClientFault: boolean; // true = faute client → facturation, false = faute entreprise → pas facturation
};

// Aligné sur backend application/bookings/cancellation_rules.py :
// Facturables (BILLABLE_REASONS) : LAST_MINUTE, NO_SHOW, CLIENT_REQUEST
// Non facturables : COMPANY_ISSUE, MAJOR_DELAY, VEHICLE_ISSUE, OTHER
const CANCEL_OPTIONS: CancelOption[] = [
    {
        id: "LAST_MINUTE",
        label: "Annulation dernière minute",
        description: "Annulation à la dernière minute (côté client ou organisation)",
        isClientFault: true,
    },
    {
        id: "CLIENT_NO_SHOW",
        label: "Client ne s'est pas présenté",
        description: "Le client n'était pas au lieu de rendez-vous",
        isClientFault: true,
    },
    {
        id: "CLIENT_REQUEST",
        label: "Client a demandé l'annulation",
        description: "Le client a demandé d'annuler la course",
        isClientFault: true,
    },
    {
        id: "COMPANY_ISSUE",
        label: "Problème entreprise",
        description: "Problème technique ou organisationnel de notre côté",
        isClientFault: false,
    },
    {
        id: "DELAY",
        label: "Retard important",
        description: "Retard trop important pour honorer la course",
        isClientFault: false,
    },
    {
        id: "VEHICLE_ISSUE",
        label: "Problème véhicule",
        description: "Panne ou problème mécanique",
        isClientFault: false,
    },
    {
        id: "OTHER",
        label: "Autre raison",
        description: "Autre raison nécessitant une justification",
        isClientFault: false,
    },
];

type Props = {
    visible: boolean;
    onClose: () => void;
    onConfirm: (reason: string, isClientFault: boolean) => void;
};

export default function CancelJustificationModal({
    visible,
    onClose,
    onConfirm,
}: Props) {
    const [selectedOption, setSelectedOption] = useState<string | null>(null);

    const handleConfirm = () => {
        if (!selectedOption) return;
        const option = CANCEL_OPTIONS.find((opt) => opt.id === selectedOption);
        if (option) {
            onConfirm(option.id, option.isClientFault);
            setSelectedOption(null);
        }
    };

    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <View style={styles.overlay}>
                <View style={styles.modalContainer}>
                    <View style={styles.header}>
                        <Text style={styles.title}>Justifier l'annulation</Text>
                        <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                            <Ionicons name="close" size={24} color="#5F7369" />
                        </TouchableOpacity>
                    </View>

                    <Text style={styles.subtitle}>
                        Veuillez sélectionner la raison de l'annulation
                    </Text>

                    <ScrollView style={styles.optionsContainer} showsVerticalScrollIndicator={false}>
                        {CANCEL_OPTIONS.map((option) => {
                            const isSelected = selectedOption === option.id;
                            return (
                                <TouchableOpacity
                                    key={option.id}
                                    style={[
                                        styles.optionCard,
                                        isSelected && styles.optionCardSelected,
                                        option.isClientFault && styles.optionCardClientFault,
                                    ]}
                                    onPress={() => setSelectedOption(option.id)}
                                >
                                    <View style={styles.optionContent}>
                                        <View style={styles.optionHeader}>
                                            <Text
                                                style={[
                                                    styles.optionLabel,
                                                    isSelected && styles.optionLabelSelected,
                                                ]}
                                            >
                                                {option.label}
                                            </Text>
                                            <View
                                                style={[
                                                    styles.checkbox,
                                                    isSelected && styles.checkboxSelected,
                                                ]}
                                            >
                                                {isSelected && (
                                                    <Ionicons name="checkmark" size={16} color="#FFFFFF" />
                                                )}
                                            </View>
                                        </View>
                                        <Text
                                            style={[
                                                styles.optionDescription,
                                                isSelected && styles.optionDescriptionSelected,
                                            ]}
                                        >
                                            {option.description}
                                        </Text>
                                        {option.isClientFault && (
                                            <View style={styles.billingBadge}>
                                                <Ionicons name="card" size={12} color="#8B6914" />
                                                <Text style={styles.billingText}>
                                                    Facturation prévue
                                                </Text>
                                            </View>
                                        )}
                                        {!option.isClientFault && (
                                            <View style={styles.noBillingBadge}>
                                                <Ionicons name="card-outline" size={12} color="#0A7F59" />
                                                <Text style={styles.noBillingText}>
                                                    Pas de facturation
                                                </Text>
                                            </View>
                                        )}
                                    </View>
                                </TouchableOpacity>
                            );
                        })}
                    </ScrollView>

                    <View style={styles.footer}>
                        <TouchableOpacity
                            style={[styles.button, styles.cancelButton]}
                            onPress={onClose}
                        >
                            <Text style={styles.cancelButtonText}>Annuler</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[
                                styles.button,
                                styles.confirmButton,
                                !selectedOption && styles.confirmButtonDisabled,
                            ]}
                            onPress={handleConfirm}
                            disabled={!selectedOption}
                        >
                            <Text
                                style={[
                                    styles.confirmButtonText,
                                    !selectedOption && styles.confirmButtonTextDisabled,
                                ]}
                            >
                                Confirmer l'annulation
                            </Text>
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
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        justifyContent: "center",
        alignItems: "center",
        padding: 20,
    },
    modalContainer: {
        backgroundColor: "#FFFFFF",
        borderRadius: 24,
        width: "100%",
        maxHeight: "80%",
        ...(Platform.OS === "web"
            ? { boxShadow: "0 8px 24px rgba(0,0,0,0.25)" }
            : {
                  shadowColor: "#000",
                  shadowOffset: { width: 0, height: 8 },
                  shadowOpacity: 0.25,
                  shadowRadius: 16,
                  elevation: 10,
              }),
    },
    header: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        padding: 20,
        borderBottomWidth: 1,
        borderBottomColor: "rgba(15,54,43,0.08)",
    },
    title: {
        fontSize: 20,
        fontWeight: "700",
        color: "#15362B",
        letterSpacing: -0.3,
    },
    closeButton: {
        padding: 4,
    },
    subtitle: {
        fontSize: 14,
        color: "#5F7369",
        paddingHorizontal: 20,
        paddingTop: 12,
        paddingBottom: 16,
    },
    optionsContainer: {
        maxHeight: 400,
        paddingHorizontal: 20,
    },
    optionCard: {
        backgroundColor: "#F5F7F6",
        borderRadius: 16,
        padding: 16,
        marginBottom: 12,
        borderWidth: 2,
        borderColor: "transparent",
    },
    optionCardSelected: {
        backgroundColor: "rgba(10,127,89,0.08)",
        borderColor: "#0A7F59",
    },
    optionCardClientFault: {
        borderLeftWidth: 4,
        borderLeftColor: "#FFC107",
    },
    optionContent: {
        flex: 1,
    },
    optionHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "flex-start",
        marginBottom: 6,
    },
    optionLabel: {
        fontSize: 15,
        fontWeight: "600",
        color: "#15362B",
        flex: 1,
        marginRight: 12,
    },
    optionLabelSelected: {
        color: "#0A7F59",
    },
    checkbox: {
        width: 24,
        height: 24,
        borderRadius: 12,
        borderWidth: 2,
        borderColor: "#5F7369",
        justifyContent: "center",
        alignItems: "center",
    },
    checkboxSelected: {
        backgroundColor: "#0A7F59",
        borderColor: "#0A7F59",
    },
    optionDescription: {
        fontSize: 13,
        color: "#5F7369",
        marginBottom: 8,
        lineHeight: 18,
    },
    optionDescriptionSelected: {
        color: "#15362B",
    },
    billingBadge: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "rgba(255,193,7,0.15)",
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 8,
        alignSelf: "flex-start",
        marginTop: 4,
    },
    billingText: {
        fontSize: 11,
        fontWeight: "600",
        color: "#8B6914",
        marginLeft: 4,
    },
    noBillingBadge: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "rgba(10,127,89,0.12)",
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 8,
        alignSelf: "flex-start",
        marginTop: 4,
    },
    noBillingText: {
        fontSize: 11,
        fontWeight: "600",
        color: "#0A7F59",
        marginLeft: 4,
    },
    footer: {
        flexDirection: "row",
        padding: 20,
        borderTopWidth: 1,
        borderTopColor: "rgba(15,54,43,0.08)",
        gap: 12,
    },
    button: {
        flex: 1,
        paddingVertical: 14,
        borderRadius: 16,
        alignItems: "center",
        justifyContent: "center",
    },
    cancelButton: {
        backgroundColor: "#F5F7F6",
        borderWidth: 1,
        borderColor: "#E0E0E0",
    },
    cancelButtonText: {
        fontSize: 15,
        fontWeight: "600",
        color: "#5F7369",
    },
    confirmButton: {
        backgroundColor: "#0A7F59",
    },
    confirmButtonDisabled: {
        backgroundColor: "#91A59D",
        opacity: 0.5,
    },
    confirmButtonText: {
        fontSize: 15,
        fontWeight: "600",
        color: "#FFFFFF",
    },
    confirmButtonTextDisabled: {
        color: "#FFFFFF",
    },
});

