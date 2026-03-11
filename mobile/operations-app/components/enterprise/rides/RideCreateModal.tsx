import React, { useState, useEffect, useCallback } from "react";
import {
    Modal,
    View,
    Text,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    StyleSheet,
    TextInput,
    KeyboardAvoidingView,
    Platform,
    Pressable,
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
import { RecurrenceSelector } from "./RecurrenceSelector";
import { createShadow } from "@/styles/shadowStyles";
import { getLogger } from "@/utils/logger";

const log = getLogger("RideCreate");

const MEDICAL_FACILITY_TYPES = new Set([
    "hospital", "dentist", "health", "physiotherapist", "pharmacy",
]);
const DOCTOR_TYPES = new Set(["doctor"]);

const BRAND = "#00796B";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const BG = "#f4f7fc";
const CARD = "#FFFFFF";
const DANGER = "#dc3545";

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

    // Client (obligatoire — alignement web)
    const [client, setClient] = useState<ClientOption | null>(null);

    // Addresses
    const [pickupAddress, setPickupAddress] = useState("");
    const [pickupSuggestion, setPickupSuggestion] = useState<AddressSuggestion | undefined>();
    const [dropoffAddress, setDropoffAddress] = useState("");
    const [dropoffSuggestion, setDropoffSuggestion] = useState<AddressSuggestion | undefined>();

    // Pre-fill pickup from client domicile
    useEffect(() => {
        if (client?.domicile_address) {
            let fullAddress = client.domicile_address;
            if (client.domicile_zip || client.domicile_city) {
                const parts = [client.domicile_address];
                if (client.domicile_zip) parts.push(client.domicile_zip);
                if (client.domicile_city) parts.push(client.domicile_city);
                fullAddress = parts.join(", ");
            }
            setPickupAddress(fullAddress);
            if (client.domicile_lat != null && client.domicile_lon != null) {
                setPickupSuggestion({
                    label: fullAddress,
                    address: fullAddress,
                    lat: client.domicile_lat,
                    lon: client.domicile_lon,
                });
            }
        } else if (!client) {
            setPickupAddress("");
            setPickupSuggestion(undefined);
        }
    }, [client]);

    // Schedule
    const [scheduledTime, setScheduledTime] = useState<Date | null>(null);
    const [isReturn, setIsReturn] = useState(false);
    const [returnTime, setReturnTime] = useState<Date | null>(null);
    const [returnDateManuallySet, setReturnDateManuallySet] = useState(false);

    // Recurrence
    const [isRecurring, setIsRecurring] = useState(false);
    const [recurrenceType, setRecurrenceType] = useState<"daily" | "weekly" | "custom">("weekly");
    const [recurrenceDays, setRecurrenceDays] = useState<number[]>([]);
    const [occurrences, setOccurrences] = useState(4);
    const [recurrenceEndDate, setRecurrenceEndDate] = useState("");

    // Amount
    const [amount, setAmount] = useState("");

    // Medical info
    const [medicalExpanded, setMedicalExpanded] = useState(false);
    const [medicalFacility, setMedicalFacility] = useState("");
    const [hospitalService, setHospitalService] = useState("");
    const [doctorName, setDoctorName] = useState("");
    const [notesMedical, setNotesMedical] = useState("");
    const [pickupAccessNotes, setPickupAccessNotes] = useState("");
    const [dropoffAccessNotes, setDropoffAccessNotes] = useState("");

    // Wheelchair
    const [wheelchairClientHas, setWheelchairClientHas] = useState(false);
    const [wheelchairNeed, setWheelchairNeed] = useState(false);

    // Notes & priority
    const [notes, setNotes] = useState("");
    const [priority, setPriority] = useState<"LOW" | "NORMAL" | "HIGH">("NORMAL");

    useEffect(() => {
        if (!visible) {
            setClient(null);
            setPickupAddress("");
            setPickupSuggestion(undefined);
            setDropoffAddress("");
            setDropoffSuggestion(undefined);
            setScheduledTime(null);
            setIsReturn(false);
            setReturnTime(null);
            setReturnDateManuallySet(false);
            setNotes("");
            setPriority("NORMAL");
            setAmount("");
            setWheelchairClientHas(false);
            setWheelchairNeed(false);
            setMedicalExpanded(false);
            setMedicalFacility("");
            setHospitalService("");
            setDoctorName("");
            setNotesMedical("");
            setPickupAccessNotes("");
            setDropoffAccessNotes("");
            setIsRecurring(false);
            setRecurrenceDays([]);
            setOccurrences(4);
            setRecurrenceEndDate("");
        }
    }, [visible]);

    // Synchroniser la date de retour avec la date de l'aller
    // sauf si l'utilisateur a manuellement modifie la date de retour
    useEffect(() => {
        if (!isReturn || !scheduledTime || returnDateManuallySet) return;
        const outboundDate = dayjs(scheduledTime).startOf("day");
        const currentReturnDate = returnTime ? dayjs(returnTime).startOf("day") : null;
        if (!currentReturnDate || !outboundDate.isSame(currentReturnDate, "day")) {
            setReturnTime((prev) => {
                if (!prev) return outboundDate.toDate();
                return outboundDate.hour(dayjs(prev).hour()).minute(dayjs(prev).minute()).toDate();
            });
        }
    }, [isReturn, scheduledTime, returnDateManuallySet]);

    const handleSwapAddresses = useCallback(() => {
        const tmpAddr = pickupAddress;
        const tmpSug = pickupSuggestion;
        setPickupAddress(dropoffAddress);
        setPickupSuggestion(dropoffSuggestion);
        setDropoffAddress(tmpAddr);
        setDropoffSuggestion(tmpSug);
    }, [pickupAddress, pickupSuggestion, dropoffAddress, dropoffSuggestion]);

    const handleDropoffChange = useCallback((addr: string, sug?: AddressSuggestion) => {
        setDropoffAddress(addr);
        setDropoffSuggestion(sug);
        if (!sug) return;

        const types = sug.types ?? [];
        const isMedicalFacility = types.some((t) => MEDICAL_FACILITY_TYPES.has(t)) || sug.category === "hospital";
        const isDoctorOffice = types.some((t) => DOCTOR_TYPES.has(t));

        if (isMedicalFacility && sug.name) {
            setMedicalFacility(sug.name);
            setMedicalExpanded(true);
        } else if (isDoctorOffice && sug.name) {
            setDoctorName(sug.name);
            setMedicalExpanded(true);
        }
    }, []);

    const applyPreset = useCallback((preset: "now30" | "now1h" | "tomorrow9") => {
        let d: Date;
        switch (preset) {
            case "now30":
                d = dayjs().add(30, "minute").toDate();
                break;
            case "now1h":
                d = dayjs().add(1, "hour").toDate();
                break;
            case "tomorrow9":
                d = dayjs().add(1, "day").hour(9).minute(0).second(0).toDate();
                break;
        }
        setScheduledTime(d);
    }, []);

    const canSubmit = client !== null
        && pickupAddress.trim().length > 0
        && dropoffAddress.trim().length > 0
        && scheduledTime !== null;

    const handleCreate = async () => {
        if (!canSubmit) return;

        const payload: RideCreatePayload = {
            client_id: client!.id,
            pickup_address: pickupAddress,
            dropoff_address: dropoffAddress,
            pickup_lat: pickupSuggestion?.lat,
            pickup_lon: pickupSuggestion?.lon,
            dropoff_lat: dropoffSuggestion?.lat,
            dropoff_lon: dropoffSuggestion?.lon,
            scheduled_time: scheduledTime ? dayjs(scheduledTime).format("YYYY-MM-DDTHH:mm:ss") : undefined,
            notes: notes || undefined,
            priority,
            amount: amount ? parseFloat(amount) : undefined,
            is_return: isReturn,
            return_time: returnTime ? (() => {
                const rd = dayjs(returnTime);
                if (rd.hour() === 0 && rd.minute() === 0) return rd.format("YYYY-MM-DD");
                return rd.format("YYYY-MM-DDTHH:mm:ss");
            })() : undefined,
            wheelchair_client_has: wheelchairClientHas || undefined,
            wheelchair_need: wheelchairNeed || undefined,
            medical_facility: medicalFacility || undefined,
            hospital_service: hospitalService || undefined,
            doctor_name: doctorName || undefined,
            notes_medical: notesMedical || undefined,
            pickup_access_notes: pickupAccessNotes || undefined,
            dropoff_access_notes: dropoffAccessNotes || undefined,
            ...(isRecurring ? {
                is_recurring: true,
                recurrence_type: recurrenceType,
                recurrence_days: recurrenceType === "custom" && recurrenceDays.length > 0 ? recurrenceDays : undefined,
                recurrence_end_date: recurrenceEndDate || undefined,
                occurrences: occurrences > 0 ? occurrences : undefined,
            } : {}),
        };

        try {
            log.info("payload before submit", { payload });
            const created = await create(payload);
            log.success("ride created", { created });
            if (payload.scheduled_time) {
                setSelectedDate(dayjs(payload.scheduled_time).format("YYYY-MM-DD"));
            }
            await new Promise((r) => setTimeout(r, 500));
            if (onSuccess) await onSuccess();
            onClose();
        } catch (error: any) {
            log.error("ride create failed", { error, response: error?.response?.data });
        }
    };

    return (
        <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
            <KeyboardAvoidingView
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                style={s.root}
            >
                <Pressable style={s.overlay} onPress={onClose} />
                <View style={s.sheet}>
                    <View style={s.handle} />

                    {/* Header */}
                    <View style={s.header}>
                        <View style={s.headerIconWrap}>
                            <Ionicons name="add-circle-outline" size={20} color={BRAND} />
                        </View>
                        <View style={{ flex: 1 }}>
                            <Text style={s.headerTitle}>Créer une réservation</Text>
                            <Text style={s.headerSub}>Renseignez le trajet, puis ajoutez les détails.</Text>
                        </View>
                        <TouchableOpacity onPress={onClose} style={s.closeBtn}>
                            <Ionicons name="close" size={22} color={TEXT_SEC} />
                        </TouchableOpacity>
                    </View>

                    <ScrollView
                        style={s.scroll}
                        contentContainerStyle={s.scrollContent}
                        showsVerticalScrollIndicator={false}
                        keyboardShouldPersistTaps="handled"
                        nestedScrollEnabled
                    >
                    {/* Client — obligatoire (alignement web) */}
                    <View style={s.fieldGroup}>
                        <Text style={s.label}>Client <Text style={s.required}>*</Text></Text>
                        <ClientSelector
                            label=""
                            value={client}
                            onChange={setClient}
                            onNewClient={() => onOpenClientCreate?.()}
                        />
                        {!client && (
                            <Text style={[s.labelSm, { marginTop: 6, color: TEXT_MUTED }]}>
                                Veuillez sélectionner un client existant pour créer une réservation.
                            </Text>
                        )}
                    </View>

                    {/* Addresses */}
                    <View style={s.fieldGroup}>
                        <Text style={s.label}>Lieu de prise en charge <Text style={s.required}>*</Text></Text>
                        <AddressSelector
                            label=""
                            value={pickupAddress}
                            onChange={(addr, sug) => { setPickupAddress(addr); setPickupSuggestion(sug); }}
                            icon="location-outline"
                        />
                    </View>

                    <TouchableOpacity style={s.swapBtn} onPress={handleSwapAddresses} activeOpacity={0.7}>
                        <Ionicons name="swap-vertical" size={18} color={BRAND} />
                    </TouchableOpacity>

                    <View style={s.fieldGroup}>
                        <Text style={s.label}>Lieu de destination <Text style={s.required}>*</Text></Text>
                        <AddressSelector
                            label=""
                            value={dropoffAddress}
                            onChange={handleDropoffChange}
                            icon="flag-outline"
                        />
                    </View>

                    {/* Date & Time */}
                    <View style={s.fieldGroup}>
                        <Text style={s.label}>Date & heure de départ <Text style={s.required}>*</Text></Text>
                        <TimeDatePicker
                            label=""
                            value={scheduledTime}
                            onChange={setScheduledTime}
                            mode="datetime"
                        />
                        <View style={s.presetRow}>
                            <TouchableOpacity style={s.presetChip} onPress={() => applyPreset("now30")}>
                                <Text style={s.presetText}>+30 min</Text>
                            </TouchableOpacity>
                            <TouchableOpacity style={s.presetChip} onPress={() => applyPreset("now1h")}>
                                <Text style={s.presetText}>+1h</Text>
                            </TouchableOpacity>
                            <TouchableOpacity style={s.presetChip} onPress={() => applyPreset("tomorrow9")}>
                                <Text style={s.presetText}>Demain 9h</Text>
                            </TouchableOpacity>
                        </View>
                    </View>

                    {/* Chips: AR / Récurrence */}
                    <View style={s.chipsRow}>
                        <TouchableOpacity
                            style={[s.toggleChip, isReturn && s.toggleChipActive]}
                            onPress={() => {
                                const next = !isReturn;
                                setIsReturn(next);
                                setReturnDateManuallySet(false);
                                if (next && scheduledTime) {
                                    setReturnTime(dayjs(scheduledTime).startOf("day").toDate());
                                } else if (!next) {
                                    setReturnTime(null);
                                }
                            }}
                        >
                            <Ionicons name="repeat-outline" size={14} color={isReturn ? BRAND : TEXT_SEC} />
                            <Text style={[s.toggleChipText, isReturn && s.toggleChipTextActive]}>Trajet AR</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[s.toggleChip, isRecurring && s.toggleChipActive]}
                            onPress={() => {
                                setIsRecurring(!isRecurring);
                                if (isRecurring) { setRecurrenceDays([]); setOccurrences(4); setRecurrenceEndDate(""); }
                            }}
                        >
                            <Ionicons name="calendar-outline" size={14} color={isRecurring ? BRAND : TEXT_SEC} />
                            <Text style={[s.toggleChipText, isRecurring && s.toggleChipTextActive]}>Récurrente</Text>
                        </TouchableOpacity>
                    </View>

                    {/* Return time */}
                    {isReturn && (
                        <View style={s.subsection}>
                            <TimeDatePicker label="Date de retour" value={returnTime} onChange={(d) => {
                                setReturnDateManuallySet(true);
                                if (d && returnTime) {
                                    setReturnTime(dayjs(d).hour(dayjs(returnTime).hour()).minute(dayjs(returnTime).minute()).toDate());
                                } else setReturnTime(d);
                            }} mode="date" />
                            <TimeDatePicker label="Heure de retour (opt.)" value={returnTime || (scheduledTime ? dayjs(scheduledTime).startOf("day").toDate() : null)} onChange={(d) => {
                                if (d) {
                                    const base = returnTime || scheduledTime || new Date();
                                    setReturnTime(dayjs(base).startOf("day").hour(dayjs(d).hour()).minute(dayjs(d).minute()).toDate());
                                } else if (returnTime) {
                                    setReturnTime(dayjs(returnTime).startOf("day").toDate());
                                }
                            }} mode="time" />
                        </View>
                    )}

                    {/* Recurrence */}
                    {isRecurring && (
                        <View style={s.subsection}>
                            <RecurrenceSelector
                                enabled={isRecurring}
                                onEnabledChange={setIsRecurring}
                                recurrenceType={recurrenceType}
                                onRecurrenceTypeChange={setRecurrenceType}
                                recurrenceDays={recurrenceDays}
                                onRecurrenceDaysChange={setRecurrenceDays}
                                occurrences={occurrences}
                                onOccurrencesChange={setOccurrences}
                                endDate={recurrenceEndDate}
                                onEndDateChange={setRecurrenceEndDate}
                            />
                        </View>
                    )}

                    {/* Amount */}
                    <View style={s.fieldGroup}>
                        <Text style={s.label}>Montant (optionnel)</Text>
                        <View style={s.inputRow}>
                            <Ionicons name="cash-outline" size={16} color={TEXT_MUTED} />
                            <TextInput
                                style={s.inputText}
                                value={amount}
                                onChangeText={setAmount}
                                placeholder="Ex: 45.00"
                                placeholderTextColor={TEXT_MUTED}
                                keyboardType="decimal-pad"
                            />
                        </View>
                    </View>

                    {/* Medical Info (collapsible) */}
                    <TouchableOpacity
                        style={s.sectionToggle}
                        onPress={() => setMedicalExpanded(!medicalExpanded)}
                        activeOpacity={0.7}
                    >
                        <View style={s.sectionToggleLeft}>
                            <Ionicons name="medkit-outline" size={16} color={BRAND} />
                            <Text style={s.sectionToggleText}>Informations médicales</Text>
                        </View>
                        <Ionicons name={medicalExpanded ? "chevron-up" : "chevron-down"} size={16} color={TEXT_SEC} />
                    </TouchableOpacity>

                    {medicalExpanded && (
                        <View style={s.medicalSection}>
                            <View style={s.fieldGroup}>
                                <Text style={s.labelSm}>Établissement</Text>
                                <View style={s.inputRow}>
                                    <TextInput
                                        style={s.inputText}
                                        value={medicalFacility}
                                        onChangeText={setMedicalFacility}
                                        placeholder="HUG, Clinique La Colline, Grangettes…"
                                        placeholderTextColor={TEXT_MUTED}
                                    />
                                </View>
                            </View>
                            <View style={s.fieldGroup}>
                                <Text style={s.labelSm}>Service / Bât.</Text>
                                <View style={s.inputRow}>
                                    <TextInput
                                        style={s.inputText}
                                        value={hospitalService}
                                        onChangeText={setHospitalService}
                                        placeholder="Ex: CHIR, Urgences, Cardiologie…"
                                        placeholderTextColor={TEXT_MUTED}
                                    />
                                </View>
                            </View>
                            <View style={s.fieldGroup}>
                                <Text style={s.labelSm}>Médecin</Text>
                                <View style={s.inputRow}>
                                    <TextInput
                                        style={s.inputText}
                                        value={doctorName}
                                        onChangeText={setDoctorName}
                                        placeholder="Ex: Dr Dupont"
                                        placeholderTextColor={TEXT_MUTED}
                                    />
                                </View>
                            </View>
                            <View style={s.fieldGroup}>
                                <Text style={s.labelSm}>Notes</Text>
                                <View style={[s.inputRow, { minHeight: 60, alignItems: "flex-start" }]}>
                                    <TextInput
                                        style={[s.inputText, { textAlignVertical: "top", paddingTop: 8 }]}
                                        value={notesMedical}
                                        onChangeText={setNotesMedical}
                                        placeholder="Instructions, bâtiment, étage…"
                                        placeholderTextColor={TEXT_MUTED}
                                        multiline
                                    />
                                </View>
                            </View>
                            <View style={s.fieldGroup}>
                                <Text style={s.labelSm}>Accès pickup</Text>
                                <View style={s.inputRow}>
                                    <TextInput
                                        style={s.inputText}
                                        value={pickupAccessNotes}
                                        onChangeText={setPickupAccessNotes}
                                        placeholder="Ex: entrée arrière, sonner à…"
                                        placeholderTextColor={TEXT_MUTED}
                                    />
                                </View>
                            </View>
                            <View style={s.fieldGroup}>
                                <Text style={s.labelSm}>Accès destination</Text>
                                <View style={s.inputRow}>
                                    <TextInput
                                        style={s.inputText}
                                        value={dropoffAccessNotes}
                                        onChangeText={setDropoffAccessNotes}
                                        placeholder="Ex: entrée B, étage 2, service…"
                                        placeholderTextColor={TEXT_MUTED}
                                    />
                                </View>
                            </View>

                            {/* Wheelchair */}
                            <View style={s.fieldGroup}>
                                <Text style={s.labelSm}>Chaise roulante</Text>
                                <View style={s.chipsRow}>
                                    <TouchableOpacity
                                        style={[s.toggleChip, wheelchairClientHas && s.toggleChipActive]}
                                        onPress={() => { setWheelchairClientHas(!wheelchairClientHas); if (!wheelchairClientHas) setWheelchairNeed(false); }}
                                    >
                                        <Text style={[s.toggleChipText, wheelchairClientHas && s.toggleChipTextActive]}>En chaise</Text>
                                    </TouchableOpacity>
                                    <TouchableOpacity
                                        style={[s.toggleChip, wheelchairNeed && s.toggleChipActive]}
                                        onPress={() => { setWheelchairNeed(!wheelchairNeed); if (!wheelchairNeed) setWheelchairClientHas(false); }}
                                    >
                                        <Text style={[s.toggleChipText, wheelchairNeed && s.toggleChipTextActive]}>Fournir chaise</Text>
                                    </TouchableOpacity>
                                </View>
                            </View>
                        </View>
                    )}

                    {/* Notes internes */}
                    <View style={s.fieldGroup}>
                        <Text style={s.label}>Notes internes</Text>
                        <NotesEditor label="" value={notes} onChange={setNotes} placeholder="Ajouter des notes..." />
                    </View>

                    <View style={{ height: 20 }} />
                </ScrollView>

                {/* Footer */}
                <View style={s.footer}>
                    <TouchableOpacity style={s.footerCancel} onPress={onClose} disabled={loading}>
                        <Text style={s.footerCancelText}>Annuler</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={[s.footerSubmit, !canSubmit && s.footerSubmitDisabled]}
                        onPress={handleCreate}
                        disabled={!canSubmit || loading}
                        activeOpacity={0.85}
                    >
                        {loading ? (
                            <ActivityIndicator color="#FFF" size="small" />
                        ) : (
                            <>
                                <Ionicons name="checkmark" size={16} color="#FFF" />
                                <Text style={s.footerSubmitText}>Créer la réservation</Text>
                            </>
                        )}
                    </TouchableOpacity>
                </View>
                </View>
            </KeyboardAvoidingView>
        </Modal>
    );
};

const s = StyleSheet.create({
    root: {
        flex: 1,
        justifyContent: "flex-end" as const,
    },
    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.35)",
    },
    sheet: {
        backgroundColor: CARD,
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        height: "90%",
        overflow: "hidden",
        ...createShadow({ shadowColor: "#000", shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.12, shadowRadius: 24, elevation: 12 }),
    },
    handle: {
        width: 36,
        height: 4,
        borderRadius: 2,
        backgroundColor: "#D1D5DB",
        alignSelf: "center",
        marginTop: 10,
        marginBottom: 6,
    },
    header: {
        flexDirection: "row",
        alignItems: "center",
        gap: 10,
        paddingHorizontal: 20,
        paddingVertical: 14,
        borderBottomWidth: 1,
        borderBottomColor: BORDER,
    },
    headerIconWrap: {
        width: 36,
        height: 36,
        borderRadius: 10,
        backgroundColor: "rgba(0,121,107,0.08)",
        alignItems: "center",
        justifyContent: "center",
    },
    headerTitle: {
        fontSize: 16,
        fontWeight: "700",
        color: TEXT,
    },
    headerSub: {
        fontSize: 12,
        color: TEXT_SEC,
        marginTop: 2,
    },
    closeBtn: {
        padding: 4,
    },
    scroll: {
        flex: 1,
    },
    scrollContent: {
        paddingHorizontal: 20,
        paddingTop: 16,
        paddingBottom: 20,
    },
    fieldGroup: {
        marginBottom: 14,
    },
    label: {
        fontSize: 13,
        fontWeight: "600",
        color: TEXT,
        marginBottom: 6,
    },
    labelSm: {
        fontSize: 12,
        fontWeight: "600",
        color: TEXT_SEC,
        marginBottom: 5,
    },
    required: {
        color: DANGER,
    },
    inputRow: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: BG,
        borderRadius: 10,
        borderWidth: 1,
        borderColor: BORDER,
        paddingHorizontal: 12,
        paddingVertical: 10,
        gap: 8,
    },
    inputText: {
        flex: 1,
        color: TEXT,
        fontSize: 14,
        padding: 0,
    },

    swapBtn: {
        alignSelf: "center",
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: "rgba(0,121,107,0.08)",
        alignItems: "center",
        justifyContent: "center",
        marginVertical: -4,
        marginBottom: 10,
    },

    presetRow: {
        flexDirection: "row",
        gap: 8,
        marginTop: 8,
    },
    presetChip: {
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 8,
        backgroundColor: BG,
        borderWidth: 1,
        borderColor: BORDER,
    },
    presetText: {
        fontSize: 12,
        fontWeight: "600",
        color: BRAND,
    },

    chipsRow: {
        flexDirection: "row",
        gap: 8,
        marginBottom: 14,
    },
    toggleChip: {
        flexDirection: "row",
        alignItems: "center",
        gap: 5,
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 8,
        backgroundColor: BG,
        borderWidth: 1,
        borderColor: BORDER,
    },
    toggleChipActive: {
        borderColor: BRAND,
        backgroundColor: "rgba(0,121,107,0.06)",
    },
    toggleChipText: {
        fontSize: 12,
        fontWeight: "600",
        color: TEXT_SEC,
    },
    toggleChipTextActive: {
        color: BRAND,
    },

    subsection: {
        backgroundColor: "rgba(0,121,107,0.03)",
        borderRadius: 12,
        borderWidth: 1,
        borderColor: "rgba(0,121,107,0.1)",
        padding: 14,
        marginBottom: 14,
        gap: 10,
    },

    sectionToggle: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        paddingVertical: 12,
        paddingHorizontal: 14,
        borderRadius: 12,
        backgroundColor: BG,
        borderWidth: 1,
        borderColor: BORDER,
        marginBottom: 10,
    },
    sectionToggleLeft: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
    },
    sectionToggleText: {
        fontSize: 13,
        fontWeight: "600",
        color: TEXT,
    },

    medicalSection: {
        backgroundColor: "rgba(0,121,107,0.02)",
        borderRadius: 12,
        borderWidth: 1,
        borderColor: "rgba(0,121,107,0.08)",
        padding: 14,
        marginBottom: 14,
        gap: 4,
    },

    footer: {
        flexDirection: "row",
        gap: 10,
        paddingHorizontal: 20,
        paddingVertical: 14,
        borderTopWidth: 1,
        borderTopColor: BORDER,
    },
    footerCancel: {
        flex: 1,
        alignItems: "center",
        paddingVertical: 13,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: BORDER,
        backgroundColor: CARD,
    },
    footerCancelText: {
        color: TEXT_SEC,
        fontWeight: "600",
        fontSize: 14,
    },
    footerSubmit: {
        flex: 2,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        paddingVertical: 13,
        borderRadius: 12,
        backgroundColor: BRAND,
        ...createShadow({ shadowColor: BRAND, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6, elevation: 3 }),
    },
    footerSubmitDisabled: {
        opacity: 0.4,
    },
    footerSubmitText: {
        color: "#FFFFFF",
        fontWeight: "700",
        fontSize: 14,
    },
});
