import React, { useState, useEffect } from "react";
import {
    Modal,
    View,
    Text,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    StyleSheet,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import { RideCreatePayload, ClientOption, AddressSuggestion } from "@/types/enterpriseDispatch";
import { useRideCreate } from "@/hooks/useRideCreate";
import { useEnterpriseContext } from "@/context/EnterpriseContext";
import { AddressSelector } from "./AddressSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { ClientSelector } from "./ClientSelector";
import { NotesEditor } from "./NotesEditor";

const palette = {
    modalOverlay: "rgba(21,54,43,0.75)",
    modalBackground: "#FFFFFF",
    modalBorder: "rgba(15,54,43,0.12)",
    modalTitle: "#15362B",
    modalText: "#5F7369",
    modalButton: "#0A7F59",
    modalButtonText: "#FFFFFF",
    modalCancelText: "#5F7369",
    stepActive: "#0A7F59",
    stepInactive: "#91A59D",
    sectionBg: "rgba(10,127,89,0.06)",
    sectionBorder: "rgba(10,127,89,0.15)",
    divider: "rgba(15,54,43,0.08)",
};

type Step = 1 | 2 | 3 | 4;

interface RideCreateModalProps {
    visible: boolean;
    onClose: () => void;
    onSuccess?: () => Promise<void>;
    onOpenClientCreate?: () => void;
    onClientCreated?: (client: ClientOption) => void;
}

export const RideCreateModal: React.FC<RideCreateModalProps> = ({
    visible,
    onClose,
    onSuccess,
    onOpenClientCreate,
    onClientCreated,
}) => {
    const { setSelectedDate } = useEnterpriseContext();
    const { loading, create } = useRideCreate(onSuccess);
    const [currentStep, setCurrentStep] = useState<Step>(1);

    // Étape 1: Client
    const [client, setClient] = useState<ClientOption | null>(null);
    const [customerName, setCustomerName] = useState("");

    // Étape 2: Adresses
    const [pickupAddress, setPickupAddress] = useState("");
    const [pickupSuggestion, setPickupSuggestion] = useState<AddressSuggestion | undefined>();
    const [dropoffAddress, setDropoffAddress] = useState("");
    const [dropoffSuggestion, setDropoffSuggestion] = useState<AddressSuggestion | undefined>();

    // Pré-remplir l'adresse de départ quand un client est sélectionné
    useEffect(() => {
        if (client?.domicile_address) {
            // ✅ Construire l'adresse complète avec code postal et ville si disponibles
            let fullAddress = client.domicile_address;

            // Si l'adresse ne contient pas déjà le code postal et la ville,
            // les ajouter depuis domicile_zip et domicile_city
            if (client.domicile_zip || client.domicile_city) {
                const addressParts = [client.domicile_address];
                if (client.domicile_zip) {
                    addressParts.push(client.domicile_zip);
                }
                if (client.domicile_city) {
                    addressParts.push(client.domicile_city);
                }
                fullAddress = addressParts.join(", ");
            }

            setPickupAddress(fullAddress);
            if (client.domicile_lat !== null && client.domicile_lat !== undefined &&
                client.domicile_lon !== null && client.domicile_lon !== undefined) {
                const suggestion: AddressSuggestion = {
                    label: fullAddress, // ✅ Utiliser l'adresse complète
                    address: fullAddress, // ✅ Utiliser l'adresse complète
                    lat: client.domicile_lat,
                    lon: client.domicile_lon,
                };
                setPickupSuggestion(suggestion);
            }
        } else if (!client) {
            // Réinitialiser si le client est désélectionné
            setPickupAddress("");
            setPickupSuggestion(undefined);
        }
    }, [client]);

    // Étape 3: Horaire
    const [scheduledTime, setScheduledTime] = useState<Date | null>(null);
    const [isReturn, setIsReturn] = useState(false);
    const [returnTime, setReturnTime] = useState<Date | null>(null);

    // Étape 4: Détails
    const [notes, setNotes] = useState("");
    const [priority, setPriority] = useState<"LOW" | "NORMAL" | "HIGH">("NORMAL");
    const [amount, setAmount] = useState("");
    const [wheelchairClientHas, setWheelchairClientHas] = useState(false);
    const [wheelchairNeed, setWheelchairNeed] = useState(false);

    useEffect(() => {
        if (!visible) {
            // Reset form when modal closes
            setCurrentStep(1);
            setClient(null);
            setCustomerName("");
            setPickupAddress("");
            setPickupSuggestion(undefined);
            setDropoffAddress("");
            setDropoffSuggestion(undefined);
            setScheduledTime(null);
            setIsReturn(false);
            setReturnTime(null);
            setNotes("");
            setPriority("NORMAL");
            setAmount("");
            setWheelchairClientHas(false);
            setWheelchairNeed(false);
        }
    }, [visible]);

    const canGoNext = () => {
        switch (currentStep) {
            case 1:
                return client !== null || customerName.trim().length > 0;
            case 2:
                return pickupAddress.trim().length > 0 && dropoffAddress.trim().length > 0;
            case 3:
                return scheduledTime !== null;
            case 4:
                return true;
            default:
                return false;
        }
    };

    const handleNext = () => {
        if (currentStep < 4 && canGoNext()) {
            setCurrentStep((prev) => (prev + 1) as Step);
        }
    };

    const handlePrevious = () => {
        if (currentStep > 1) {
            setCurrentStep((prev) => (prev - 1) as Step);
        }
    };

    const handleCreate = async () => {
        if (!canGoNext()) return;

        const payload: RideCreatePayload = {
            // Si un client est sélectionné, on envoie seulement client_id
            // Sinon, on envoie customer_name
            ...(client?.id ? { client_id: client.id } : { customer_name: customerName || "" }),
            pickup_address: pickupAddress,
            dropoff_address: dropoffAddress,
            pickup_lat: pickupSuggestion?.lat,
            pickup_lon: pickupSuggestion?.lon,
            dropoff_lat: dropoffSuggestion?.lat,
            dropoff_lon: dropoffSuggestion?.lon,
            scheduled_time: scheduledTime ? (() => {
                // Utiliser format() au lieu de toISOString() pour préserver l'heure locale
                // Le backend utilise parse_local_naive qui attend un format ISO sans timezone
                const localISO = dayjs(scheduledTime).format("YYYY-MM-DDTHH:mm:ss");
                console.log("[RideCreateModal] scheduledTime Date:", scheduledTime);
                console.log("[RideCreateModal] scheduledTime dayjs:", dayjs(scheduledTime).format("DD.MM.YYYY HH:mm"));
                console.log("[RideCreateModal] scheduledTime local ISO (sans timezone):", localISO);
                return localISO;
            })() : undefined,
            notes: notes || undefined,
            priority: priority,
            amount: amount ? parseFloat(amount) : undefined,
            is_return: isReturn,
            return_time: returnTime ? (() => {
                // Si l'heure est à minuit, c'est que l'heure n'est pas définie
                // On envoie seulement la date (sans heure) pour indiquer "heure à définir"
                const returnDayjs = dayjs(returnTime);
                if (returnDayjs.hour() === 0 && returnDayjs.minute() === 0) {
                    // Heure non définie : envoyer seulement la date
                    const dateOnly = returnDayjs.format("YYYY-MM-DD");
                    console.log("[RideCreateModal] return_time (date seulement, heure non définie):", dateOnly);
                    return dateOnly;
                } else {
                    // Heure définie : envoyer date + heure en format local (sans timezone)
                    const localISO = returnDayjs.format("YYYY-MM-DDTHH:mm:ss");
                    console.log("[RideCreateModal] return_time (avec heure):", localISO);
                    return localISO;
                }
            })() : undefined,
            wheelchair_client_has: wheelchairClientHas || undefined,
            wheelchair_need: wheelchairNeed || undefined,
        };

        try {
            console.log("[RideCreateModal] Payload avant envoi:", JSON.stringify(payload, null, 2));
            const created = await create(payload);
            console.log("[RideCreateModal] Course créée:", created);
            if (payload.scheduled_time) {
                const courseDate = dayjs(payload.scheduled_time).format("YYYY-MM-DD");
                console.log("[RideCreateModal] Date de la course créée:", courseDate);
                // Naviguer automatiquement vers la date de la course créée
                setSelectedDate(courseDate);
                console.log("[RideCreateModal] Navigation vers la date:", courseDate);
            }
            // Attendre un peu pour que le backend termine le commit
            await new Promise((resolve) => setTimeout(resolve, 500));
            // Rafraîchir la liste
            if (onSuccess) {
                console.log("[RideCreateModal] Rafraîchissement de la liste...");
                await onSuccess();
            }
            onClose();
        } catch (error: any) {
            console.error("[RideCreateModal] Erreur lors de la création:", error);
            console.error("[RideCreateModal] Erreur response:", error?.response?.data);
            // L'erreur est déjà gérée dans le hook
        }
    };

    const renderStepIndicator = () => {
        return (
            <View style={styles.stepIndicator}>
                {[1, 2, 3, 4].map((step) => (
                    <React.Fragment key={step}>
                        <View
                            style={[
                                styles.stepCircle,
                                currentStep >= step && styles.stepCircleActive,
                            ]}
                        >
                            {currentStep > step ? (
                                <Ionicons name="checkmark" size={16} color="#FFFFFF" />
                            ) : (
                                <Text
                                    style={[
                                        styles.stepNumber,
                                        currentStep >= step && styles.stepNumberActive,
                                    ]}
                                >
                                    {step}
                                </Text>
                            )}
                        </View>
                        {step < 4 && (
                            <View
                                style={[
                                    styles.stepLine,
                                    currentStep > step && styles.stepLineActive,
                                ]}
                            />
                        )}
                    </React.Fragment>
                ))}
            </View>
        );
    };

    const renderStepContent = () => {
        switch (currentStep) {
            case 1:
                return (
                    <View style={styles.stepContent}>
                        <Text style={styles.stepTitle}>Informations client</Text>
                        <Text style={styles.stepDescription}>
                            Sélectionnez un client existant ou créez-en un nouveau
                        </Text>
                        <ClientSelector
                            label="Client"
                            value={client}
                            onChange={setClient}
                            onNewClient={() => {
                                if (onOpenClientCreate) {
                                    onOpenClientCreate();
                                }
                            }}
                        />
                        {!client && (
                            <View style={styles.inputGroup}>
                                <Text style={styles.inputLabel}>Nom du client</Text>
                                <View style={styles.textInputContainer}>
                                    <Ionicons name="person-outline" size={18} color={palette.modalButton} />
                                    <TextInput
                                        style={styles.textInput}
                                        value={customerName}
                                        onChangeText={setCustomerName}
                                        placeholder="Nom complet"
                                        placeholderTextColor={palette.modalText}
                                        editable={true}
                                    />
                                </View>
                            </View>
                        )}
                        {client && (
                            <View style={styles.inputGroup}>
                                <Text style={styles.inputLabel}>Nom du client</Text>
                                <View style={[styles.textInputContainer, styles.textInputContainerDisabled]}>
                                    <Ionicons name="person-outline" size={18} color={palette.modalText} />
                                    <TextInput
                                        style={[styles.textInput, styles.textInputDisabled]}
                                        value={client.name}
                                        placeholder="Nom complet"
                                        placeholderTextColor={palette.modalText}
                                        editable={false}
                                    />
                                </View>
                            </View>
                        )}
                    </View>
                );

            case 2:
                return (
                    <View style={styles.stepContent}>
                        <Text style={styles.stepTitle}>Adresses</Text>
                        <Text style={styles.stepDescription}>
                            Définissez les adresses de départ et d'arrivée
                        </Text>
                        <AddressSelector
                            label="Adresse de départ"
                            value={pickupAddress}
                            onChange={(address, suggestion) => {
                                setPickupAddress(address);
                                setPickupSuggestion(suggestion);
                            }}
                            icon="location-outline"
                        />
                        <AddressSelector
                            label="Adresse d'arrivée"
                            value={dropoffAddress}
                            onChange={(address, suggestion) => {
                                setDropoffAddress(address);
                                setDropoffSuggestion(suggestion);
                            }}
                            icon="flag-outline"
                        />
                    </View>
                );

            case 3:
                return (
                    <View style={styles.stepContent}>
                        <Text style={styles.stepTitle}>Horaire & Planification</Text>
                        <Text style={styles.stepDescription}>
                            Définissez la date et l'heure de la course
                        </Text>
                        <TimeDatePicker
                            label="Date et heure de départ"
                            value={scheduledTime}
                            onChange={setScheduledTime}
                            mode="datetime"
                        />
                        <TouchableOpacity
                            style={styles.returnCheckboxContainer}
                            onPress={() => {
                                const newIsReturn = !isReturn;
                                setIsReturn(newIsReturn);
                                if (newIsReturn && scheduledTime) {
                                    // Pré-remplir uniquement la date de retour (sans l'heure)
                                    const dateOnly = dayjs(scheduledTime).startOf("day").toDate();
                                    setReturnTime(dateOnly);
                                } else if (!newIsReturn) {
                                    setReturnTime(null);
                                }
                            }}
                        >
                            <Ionicons
                                name={isReturn ? "checkbox" : "square-outline"}
                                size={20}
                                color={isReturn ? palette.modalButton : palette.modalText}
                            />
                            <Text style={styles.returnCheckboxLabel}>Course aller-retour</Text>
                        </TouchableOpacity>
                        {isReturn && (
                            <View style={styles.returnSection}>
                                <TimeDatePicker
                                    label="Date de retour"
                                    value={returnTime}
                                    onChange={(date) => {
                                        // Si on change la date, on garde l'heure si elle existe, sinon on met juste la date
                                        if (date && returnTime) {
                                            // Garder l'heure existante
                                            const newDate = dayjs(date)
                                                .hour(dayjs(returnTime).hour())
                                                .minute(dayjs(returnTime).minute())
                                                .toDate();
                                            setReturnTime(newDate);
                                        } else {
                                            setReturnTime(date);
                                        }
                                    }}
                                    mode="date"
                                />
                                <TimeDatePicker
                                    label="Heure de retour (optionnel)"
                                    value={returnTime || (scheduledTime ? dayjs(scheduledTime).startOf("day").toDate() : null)}
                                    onChange={(date) => {
                                        if (date) {
                                            // Si on définit une heure, on garde la date de retour ou on utilise la date de départ
                                            const baseDate = returnTime || scheduledTime;
                                            if (baseDate) {
                                                const dateOnly = dayjs(baseDate).startOf("day");
                                                const newDate = dateOnly
                                                    .hour(dayjs(date).hour())
                                                    .minute(dayjs(date).minute())
                                                    .toDate();
                                                setReturnTime(newDate);
                                            } else {
                                                // Si aucune date de base, utiliser la date de départ ou aujourd'hui
                                                const base = scheduledTime || new Date();
                                                const dateOnly = dayjs(base).startOf("day");
                                                const newDate = dateOnly
                                                    .hour(dayjs(date).hour())
                                                    .minute(dayjs(date).minute())
                                                    .toDate();
                                                setReturnTime(newDate);
                                            }
                                        } else {
                                            // Si on supprime l'heure, on garde juste la date
                                            if (returnTime) {
                                                const dateOnly = dayjs(returnTime).startOf("day").toDate();
                                                setReturnTime(dateOnly);
                                            }
                                        }
                                    }}
                                    mode="time"
                                />
                                {returnTime && !dayjs(returnTime).isSame(dayjs(returnTime).startOf("day")) && (
                                    <Text style={styles.returnInfo}>
                                        Heure de retour définie : {dayjs(returnTime).format("HH:mm")}
                                    </Text>
                                )}
                                {returnTime && dayjs(returnTime).isSame(dayjs(returnTime).startOf("day")) && (
                                    <Text style={styles.returnInfo}>
                                        Heure de retour à définir
                                    </Text>
                                )}
                            </View>
                        )}
                    </View>
                );

            case 4:
                return (
                    <View style={styles.stepContent}>
                        <Text style={styles.stepTitle}>Détails & Validation</Text>
                        <Text style={styles.stepDescription}>
                            Ajoutez des informations complémentaires
                        </Text>
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
                        <View style={styles.inputGroup}>
                            <Text style={styles.inputLabel}>Montant (CHF)</Text>
                            <View style={styles.textInputContainer}>
                                <Ionicons name="cash-outline" size={18} color={palette.modalButton} />
                                <TextInput
                                    style={styles.textInput}
                                    value={amount}
                                    onChangeText={setAmount}
                                    placeholder="0.00"
                                    placeholderTextColor={palette.modalText}
                                    keyboardType="decimal-pad"
                                />
                            </View>
                        </View>
                        <View style={styles.wheelchairSection}>
                            <Text style={styles.wheelchairLabel}>Options chaise roulante</Text>
                            <View style={styles.checkboxGroup}>
                                <TouchableOpacity
                                    style={styles.checkboxRow}
                                    onPress={() => {
                                        setWheelchairClientHas(!wheelchairClientHas);
                                        if (!wheelchairClientHas) {
                                            setWheelchairNeed(false);
                                        }
                                    }}
                                >
                                    <View style={styles.checkbox}>
                                        {wheelchairClientHas && (
                                            <Ionicons name="checkmark" size={16} color={palette.modalButton} />
                                        )}
                                    </View>
                                    <Text style={styles.checkboxLabel}>
                                        ♿ Le client est en chaise roulante
                                    </Text>
                                </TouchableOpacity>
                                <TouchableOpacity
                                    style={styles.checkboxRow}
                                    onPress={() => {
                                        setWheelchairNeed(!wheelchairNeed);
                                        if (!wheelchairNeed) {
                                            setWheelchairClientHas(false);
                                        }
                                    }}
                                >
                                    <View style={styles.checkbox}>
                                        {wheelchairNeed && (
                                            <Ionicons name="checkmark" size={16} color={palette.modalButton} />
                                        )}
                                    </View>
                                    <Text style={styles.checkboxLabel}>
                                        🏥 Prendre une chaise roulante
                                    </Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                        <View style={styles.summaryContainer}>
                            <Text style={styles.summaryTitle}>Récapitulatif</Text>
                            <View style={styles.summaryItem}>
                                <Text style={styles.summaryLabel}>Client:</Text>
                                <Text style={styles.summaryValue}>
                                    {client?.name || customerName || "—"}
                                </Text>
                            </View>
                            <View style={styles.summaryItem}>
                                <Text style={styles.summaryLabel}>Départ:</Text>
                                <Text style={styles.summaryValue}>{pickupAddress || "—"}</Text>
                            </View>
                            <View style={styles.summaryItem}>
                                <Text style={styles.summaryLabel}>Arrivée:</Text>
                                <Text style={styles.summaryValue}>{dropoffAddress || "—"}</Text>
                            </View>
                            <View style={styles.summaryItem}>
                                <Text style={styles.summaryLabel}>Date:</Text>
                                <Text style={styles.summaryValue}>
                                    {scheduledTime ? dayjs(scheduledTime).format("DD/MM/YYYY HH:mm") : "—"}
                                </Text>
                            </View>
                        </View>
                    </View>
                );

            default:
                return null;
        }
    };

    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <View style={styles.modalOverlay}>
                <View style={styles.modalCard}>
                    <View style={styles.modalHeader}>
                        <View>
                            <Text style={styles.modalTitle}>Nouvelle course</Text>
                            <Text style={styles.modalSubtitle}>Étape {currentStep}/4</Text>
                        </View>
                        <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                            <Ionicons name="close" size={24} color={palette.modalText} />
                        </TouchableOpacity>
                    </View>

                    {renderStepIndicator()}

                    <ScrollView
                        style={styles.modalScroll}
                        contentContainerStyle={styles.modalContent}
                        showsVerticalScrollIndicator={false}
                    >
                        {renderStepContent()}
                    </ScrollView>

                    <View style={styles.modalActions}>
                        {currentStep > 1 && (
                            <TouchableOpacity
                                style={styles.modalCancel}
                                onPress={handlePrevious}
                                disabled={loading}
                            >
                                <Text style={styles.modalCancelText}>Précédent</Text>
                            </TouchableOpacity>
                        )}
                        <View style={{ flex: 1 }} />
                        {currentStep < 4 ? (
                            <TouchableOpacity
                                style={[
                                    styles.modalNext,
                                    !canGoNext() && styles.modalNextDisabled,
                                ]}
                                onPress={handleNext}
                                disabled={!canGoNext()}
                            >
                                <Text style={styles.modalNextText}>Suivant</Text>
                                <Ionicons name="chevron-forward" size={16} color="#FFFFFF" />
                            </TouchableOpacity>
                        ) : (
                            <TouchableOpacity
                                style={[
                                    styles.modalSave,
                                    (!canGoNext() || loading) && styles.modalSaveDisabled,
                                ]}
                                onPress={handleCreate}
                                disabled={!canGoNext() || loading}
                            >
                                {loading ? (
                                    <ActivityIndicator color="#FFFFFF" size="small" />
                                ) : (
                                    <>
                                        <Text style={styles.modalSaveText}>Créer la course</Text>
                                        <Ionicons name="checkmark" size={16} color="#FFFFFF" />
                                    </>
                                )}
                            </TouchableOpacity>
                        )}
                    </View>
                </View>
            </View>
        </Modal>
    );
};

// Ajout des imports manquants
import { TextInput } from "react-native";

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
        shadowColor: "rgba(15,54,43,0.15)",
        shadowOffset: { width: 0, height: 12 },
        shadowOpacity: 1,
        shadowRadius: 24,
        elevation: 8,
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
    stepIndicator: {
        flexDirection: "row",
        alignItems: "center",
        paddingHorizontal: 24,
        paddingVertical: 20,
        borderBottomWidth: 1,
        borderBottomColor: palette.divider,
    },
    stepCircle: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: palette.stepInactive,
        alignItems: "center",
        justifyContent: "center",
    },
    stepCircleActive: {
        backgroundColor: palette.stepActive,
    },
    stepNumber: {
        color: "#FFFFFF",
        fontSize: 14,
        fontWeight: "700",
    },
    stepNumberActive: {
        color: "#FFFFFF",
    },
    stepLine: {
        flex: 1,
        height: 2,
        backgroundColor: palette.stepInactive,
        marginHorizontal: 8,
    },
    stepLineActive: {
        backgroundColor: palette.stepActive,
    },
    modalScroll: {
        flex: 1,
    },
    modalContent: {
        padding: 24,
    },
    stepContent: {
        gap: 20,
    },
    stepTitle: {
        color: palette.modalTitle,
        fontSize: 18,
        fontWeight: "700",
    },
    stepDescription: {
        color: palette.modalText,
        fontSize: 14,
        marginTop: 2,
    },
    inputGroup: {
        gap: 8,
    },
    inputLabel: {
        color: palette.modalTitle,
        fontSize: 14,
        fontWeight: "600",
    },
    textInputContainer: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: palette.modalBackground,
        borderRadius: 14,
        borderWidth: 1.5,
        borderColor: palette.modalBorder,
        paddingHorizontal: 14,
        paddingVertical: 12,
        gap: 10,
    },
    textInputContainerDisabled: {
        backgroundColor: "rgba(15,54,43,0.04)",
        borderColor: "rgba(15,54,43,0.08)",
    },
    textInput: {
        flex: 1,
        color: palette.modalTitle,
        fontSize: 15,
        padding: 0,
    },
    textInputDisabled: {
        color: palette.modalText,
        opacity: 0.7,
    },
    checkboxContainer: {
        marginTop: 0,
    },
    checkbox: {
        width: 22,
        height: 22,
        borderRadius: 6,
        borderWidth: 2,
        borderColor: palette.modalBorder,
        backgroundColor: palette.modalBackground,
        alignItems: "center",
        justifyContent: "center",
    },
    checkboxLabel: {
        color: palette.modalTitle,
        fontSize: 15,
    },
    returnCheckboxContainer: {
        flexDirection: "row",
        alignItems: "center",
        gap: 10,
        paddingVertical: 8,
        marginTop: 0,
    },
    returnCheckboxLabel: {
        color: palette.modalTitle,
        fontSize: 15,
    },
    wheelchairSection: {
        marginTop: 16,
        gap: 12,
    },
    wheelchairLabel: {
        color: palette.modalTitle,
        fontSize: 14,
        fontWeight: "600",
    },
    checkboxGroup: {
        gap: 12,
    },
    checkboxRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 12,
    },
    returnSection: {
        marginTop: 16,
        padding: 16,
        backgroundColor: palette.sectionBg,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: palette.sectionBorder,
    },
    returnInfo: {
        fontSize: 13,
        color: palette.modalText,
        fontStyle: "italic",
        marginTop: 8,
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
    summaryContainer: {
        backgroundColor: palette.sectionBg,
        borderRadius: 18,
        padding: 18,
        borderWidth: 1.5,
        borderColor: palette.sectionBorder,
        gap: 12,
        marginTop: 8,
    },
    summaryTitle: {
        color: palette.modalTitle,
        fontSize: 15,
        fontWeight: "700",
        marginBottom: 4,
    },
    summaryItem: {
        flexDirection: "row",
        justifyContent: "space-between",
        paddingVertical: 6,
    },
    summaryLabel: {
        color: palette.modalText,
        fontSize: 14,
        fontWeight: "600",
    },
    summaryValue: {
        color: palette.modalTitle,
        fontSize: 14,
        flex: 1,
        textAlign: "right",
    },
    modalActions: {
        flexDirection: "row",
        alignItems: "center",
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
    modalNext: {
        flexDirection: "row",
        alignItems: "center",
        gap: 6,
        backgroundColor: palette.modalButton,
        paddingHorizontal: 24,
        paddingVertical: 12,
        borderRadius: 14,
    },
    modalNextDisabled: {
        backgroundColor: "rgba(10,127,89,0.4)",
    },
    modalNextText: {
        color: palette.modalButtonText,
        fontSize: 15,
        fontWeight: "700",
    },
    modalSave: {
        flexDirection: "row",
        alignItems: "center",
        gap: 6,
        backgroundColor: palette.modalButton,
        paddingHorizontal: 24,
        paddingVertical: 12,
        borderRadius: 14,
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

