import React, { useState, useEffect, useCallback } from "react";
import {
    Modal,
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    StyleSheet,
    KeyboardAvoidingView,
    Platform,
    Pressable,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import { RideSummary, RideEditPayload, AddressSuggestion } from "@/types/enterpriseDispatch";
import { useRideEdit } from "@/hooks/useRideEdit";
import { AddressSelector } from "./AddressSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { NotesEditor } from "./NotesEditor";
import { createShadow } from "@/styles/shadowStyles";
import { getLogger } from "@/utils/logger";

const log = getLogger("RideEdit");

const BRAND = "#00796B";
const TXT = "#1E293B";
const TXT_SEC = "#64748B";
const TXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const BG = "#f4f7fc";
const CARD = "#FFFFFF";
const DANGER = "#dc3545";

const STATUS_MAP: Record<string, { label: string; color: string }> = {
    pending:          { label: "En attente",      color: TXT_MUTED },
    assigned:         { label: "Assignée",        color: "#2563EB" },
    en_route:         { label: "En route",        color: "#7C3AED" },
    in_progress:      { label: "En cours",        color: BRAND },
    completed:        { label: "Terminée",        color: "#16A34A" },
    return_completed: { label: "Retour terminé",  color: "#16A34A" },
    canceled:         { label: "Annulée",         color: DANGER },
    cancelled:        { label: "Annulée",         color: DANGER },
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

    // Addresses
    const [pickupAddress, setPickupAddress] = useState("");
    const [pickupSuggestion, setPickupSuggestion] = useState<AddressSuggestion | undefined>();
    const [dropoffAddress, setDropoffAddress] = useState("");
    const [dropoffSuggestion, setDropoffSuggestion] = useState<AddressSuggestion | undefined>();

    // Schedule
    const [scheduledTime, setScheduledTime] = useState<Date | null>(null);

    // Amount
    const [amount, setAmount] = useState("");

    // Medical
    const [medicalExpanded, setMedicalExpanded] = useState(false);
    const [medicalFacility, setMedicalFacility] = useState("");
    const [hospitalService, setHospitalService] = useState("");
    const [doctorName, setDoctorName] = useState("");
    const [notesMedical, setNotesMedical] = useState("");
    const [pickupAccessNotes, setPickupAccessNotes] = useState("");
    const [dropoffAccessNotes, setDropoffAccessNotes] = useState("");
    const [wheelchairClientHas, setWheelchairClientHas] = useState(false);
    const [wheelchairNeed, setWheelchairNeed] = useState(false);

    // Notes & priority
    const [notes, setNotes] = useState("");
    const [priority, setPriority] = useState<"LOW" | "NORMAL" | "HIGH">("NORMAL");

    useEffect(() => {
        if (visible && ride) {
            loadRideDetail(ride.id);
        } else if (!visible) {
            clear();
            setMedicalExpanded(false);
        }
    }, [visible, ride, loadRideDetail, clear]);

    useEffect(() => {
        const summary = rideDetail?.summary || (ride && !loadingDetail ? ride : null);
        if (!summary) return;

        setPickupAddress(summary.route?.pickup_address || "");
        setDropoffAddress(summary.route?.dropoff_address || "");
        setScheduledTime(
            summary.time?.pickup_at ? dayjs(summary.time.pickup_at).toDate() : null
        );
        setNotes(rideDetail?.notes?.[0] || "");
        setPriority(summary.client?.priority || "NORMAL");

        const d = rideDetail?.summary as any;
        setAmount(d?.amount ? String(d.amount) : "");
        setMedicalFacility(d?.medical_facility || "");
        setHospitalService(d?.hospital_service || "");
        setDoctorName(d?.doctor_name || "");
        setNotesMedical(d?.notes_medical || "");
        setPickupAccessNotes(d?.pickup_access_notes || "");
        setDropoffAccessNotes(d?.dropoff_access_notes || "");
        setWheelchairClientHas(!!d?.wheelchair_client_has);
        setWheelchairNeed(!!d?.wheelchair_need);

        if (d?.medical_facility || d?.hospital_service || d?.doctor_name || d?.notes_medical || d?.pickup_access_notes || d?.dropoff_access_notes) {
            setMedicalExpanded(true);
        }
    }, [rideDetail, ride, loadingDetail]);

    const handleSwapAddresses = useCallback(() => {
        const tmpAddr = pickupAddress;
        const tmpSug = pickupSuggestion;
        setPickupAddress(dropoffAddress);
        setPickupSuggestion(dropoffSuggestion);
        setDropoffAddress(tmpAddr);
        setDropoffSuggestion(tmpSug);
    }, [pickupAddress, pickupSuggestion, dropoffAddress, dropoffSuggestion]);

    const applyPreset = useCallback((preset: "now30" | "now1h" | "tomorrow9") => {
        switch (preset) {
            case "now30":
                setScheduledTime(dayjs().add(30, "minute").toDate()); break;
            case "now1h":
                setScheduledTime(dayjs().add(1, "hour").toDate()); break;
            case "tomorrow9":
                setScheduledTime(dayjs().add(1, "day").hour(9).minute(0).second(0).toDate()); break;
        }
    }, []);

    const handleSave = useCallback(async () => {
        if (!ride) return;

        const payload: RideEditPayload = {
            pickup_address: pickupAddress,
            dropoff_address: dropoffAddress,
            pickup_lat: pickupSuggestion?.lat,
            pickup_lon: pickupSuggestion?.lon,
            dropoff_lat: dropoffSuggestion?.lat,
            dropoff_lon: dropoffSuggestion?.lon,
            scheduled_time: scheduledTime
                ? dayjs(scheduledTime).format("YYYY-MM-DDTHH:mm:ss")
                : undefined,
            notes: notes || undefined,
            priority,
            amount: amount ? parseFloat(amount) : undefined,
            medical_facility: medicalFacility.trim() || undefined,
            hospital_service: hospitalService.trim() || undefined,
            doctor_name: doctorName.trim() || undefined,
            notes_medical: notesMedical.trim() || undefined,
            pickup_access_notes: pickupAccessNotes.trim() || undefined,
            dropoff_access_notes: dropoffAccessNotes.trim() || undefined,
            wheelchair_client_has: wheelchairClientHas || undefined,
            wheelchair_need: wheelchairNeed || undefined,
        };

        try {
            log.info("payload", { payload });
            await update(ride.id, payload);
            onClose();
        } catch {
            // handled in hook
        }
    }, [ride, pickupAddress, pickupSuggestion, dropoffAddress, dropoffSuggestion, scheduledTime, notes, priority, amount, medicalFacility, hospitalService, doctorName, notesMedical, pickupAccessNotes, dropoffAccessNotes, wheelchairClientHas, wheelchairNeed, update, onClose]);

    const hasChanges = useCallback(() => {
        const summary = rideDetail?.summary || ride;
        if (!summary) return false;
        const currentTimeStr = scheduledTime ? dayjs(scheduledTime).format("YYYY-MM-DDTHH:mm:ss") : null;
        const originalTimeStr = summary.time?.pickup_at ? dayjs(summary.time.pickup_at).format("YYYY-MM-DDTHH:mm:ss") : null;
        return (
            pickupAddress !== (summary.route?.pickup_address || "") ||
            dropoffAddress !== (summary.route?.dropoff_address || "") ||
            currentTimeStr !== originalTimeStr ||
            notes !== (rideDetail?.notes?.[0] || "") ||
            priority !== (summary.client?.priority || "NORMAL") ||
            amount !== ((rideDetail?.summary as any)?.amount ? String((rideDetail?.summary as any).amount) : "") ||
            medicalFacility !== ((rideDetail?.summary as any)?.medical_facility || "") ||
            hospitalService !== ((rideDetail?.summary as any)?.hospital_service || "") ||
            doctorName !== ((rideDetail?.summary as any)?.doctor_name || "") ||
            notesMedical !== ((rideDetail?.summary as any)?.notes_medical || "") ||
            pickupAccessNotes !== ((rideDetail?.summary as any)?.pickup_access_notes || "") ||
            dropoffAccessNotes !== ((rideDetail?.summary as any)?.dropoff_access_notes || "") ||
            wheelchairClientHas !== !!(rideDetail?.summary as any)?.wheelchair_client_has ||
            wheelchairNeed !== !!(rideDetail?.summary as any)?.wheelchair_need
        );
    }, [rideDetail, ride, pickupAddress, dropoffAddress, scheduledTime, notes, priority, amount, medicalFacility, hospitalService, doctorName, notesMedical, pickupAccessNotes, dropoffAccessNotes, wheelchairClientHas, wheelchairNeed]);

    if (!ride) return null;

    const clientName = ride.client?.name || "—";
    const statusKey = ride.status?.toLowerCase() || "pending";
    const statusInfo = STATUS_MAP[statusKey] || { label: statusKey, color: TXT_MUTED };

    return (
        <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
            <KeyboardAvoidingView
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                style={st.root}
            >
                <Pressable style={st.overlay} onPress={onClose} />
                <View style={st.sheet}>
                    <View style={st.handle} />

                    {/* Header */}
                    <View style={st.header}>
                        <View style={st.headerIconWrap}>
                            <Ionicons name="create-outline" size={20} color={BRAND} />
                        </View>
                        <View style={{ flex: 1 }}>
                            <Text style={st.headerTitle}>Modifier la course</Text>
                            <Text style={st.headerSub}>#{ride.id.slice(-6)} — {clientName}</Text>
                        </View>
                        <View style={[st.statusBadge, { backgroundColor: `${statusInfo.color}14` }]}>
                            <View style={[st.statusDot, { backgroundColor: statusInfo.color }]} />
                            <Text style={[st.statusLabel, { color: statusInfo.color }]}>{statusInfo.label}</Text>
                        </View>
                        <TouchableOpacity onPress={onClose} style={st.closeBtn}>
                            <Ionicons name="close" size={22} color={TXT_SEC} />
                        </TouchableOpacity>
                    </View>

                    {loadingDetail ? (
                        <View style={st.loadingWrap}>
                            <ActivityIndicator color={BRAND} size="large" />
                            <Text style={st.loadingText}>Chargement des détails...</Text>
                        </View>
                    ) : (
                        <ScrollView
                            style={st.scroll}
                            contentContainerStyle={st.scrollContent}
                            showsVerticalScrollIndicator={false}
                            keyboardShouldPersistTaps="handled"
                            nestedScrollEnabled
                        >
                            {!rideDetail?.summary && ride && (
                                <View style={st.warningBanner}>
                                    <Ionicons name="warning-outline" size={16} color="#D97706" />
                                    <Text style={st.warningText}>
                                        Détails partiels. Vous pouvez modifier avec les informations disponibles.
                                    </Text>
                                </View>
                            )}

                            {/* Context card */}
                            <View style={st.contextCard}>
                                <View style={st.contextRow}>
                                    <Ionicons name="person-outline" size={14} color={TXT_SEC} />
                                    <Text style={st.contextLabel}>Client</Text>
                                    <Text style={st.contextValue} numberOfLines={1}>{clientName}</Text>
                                </View>
                                {ride.driver?.name && (
                                    <View style={st.contextRow}>
                                        <Ionicons name="car-outline" size={14} color={TXT_SEC} />
                                        <Text style={st.contextLabel}>Chauffeur</Text>
                                        <Text style={st.contextValue} numberOfLines={1}>{ride.driver.name}</Text>
                                    </View>
                                )}
                            </View>

                            {/* Pickup */}
                            <View style={st.fieldGroup}>
                                <Text style={st.label}>Lieu de prise en charge <Text style={st.req}>*</Text></Text>
                                <AddressSelector
                                    label=""
                                    value={pickupAddress}
                                    onChange={(addr, sug) => { setPickupAddress(addr); setPickupSuggestion(sug); }}
                                    icon="location-outline"
                                />
                            </View>

                            {/* Swap */}
                            <TouchableOpacity style={st.swapBtn} onPress={handleSwapAddresses} activeOpacity={0.7}>
                                <Ionicons name="swap-vertical" size={18} color={BRAND} />
                            </TouchableOpacity>

                            {/* Dropoff */}
                            <View style={st.fieldGroup}>
                                <Text style={st.label}>Lieu de destination <Text style={st.req}>*</Text></Text>
                                <AddressSelector
                                    label=""
                                    value={dropoffAddress}
                                    onChange={(addr, sug) => { setDropoffAddress(addr); setDropoffSuggestion(sug); }}
                                    icon="flag-outline"
                                />
                            </View>

                            {/* Date & Time */}
                            <View style={st.fieldGroup}>
                                <Text style={st.label}>Date & heure de départ <Text style={st.req}>*</Text></Text>
                                <TimeDatePicker label="" value={scheduledTime} onChange={setScheduledTime} mode="datetime" />
                                <View style={st.presetRow}>
                                    <TouchableOpacity style={st.presetChip} onPress={() => applyPreset("now30")}>
                                        <Text style={st.presetText}>+30 min</Text>
                                    </TouchableOpacity>
                                    <TouchableOpacity style={st.presetChip} onPress={() => applyPreset("now1h")}>
                                        <Text style={st.presetText}>+1h</Text>
                                    </TouchableOpacity>
                                    <TouchableOpacity style={st.presetChip} onPress={() => applyPreset("tomorrow9")}>
                                        <Text style={st.presetText}>Demain 9h</Text>
                                    </TouchableOpacity>
                                </View>
                            </View>

                            {/* Amount */}
                            <View style={st.fieldGroup}>
                                <Text style={st.label}>Montant (optionnel)</Text>
                                <View style={st.inputRow}>
                                    <Ionicons name="cash-outline" size={16} color={TXT_MUTED} />
                                    <TextInput
                                        style={st.inputText}
                                        value={amount}
                                        onChangeText={setAmount}
                                        placeholder="Ex: 45.00"
                                        placeholderTextColor={TXT_MUTED}
                                        keyboardType="decimal-pad"
                                    />
                                </View>
                            </View>

                            {/* Medical info (collapsible) */}
                            <TouchableOpacity
                                style={st.sectionToggle}
                                onPress={() => setMedicalExpanded(!medicalExpanded)}
                                activeOpacity={0.7}
                            >
                                <View style={st.sectionToggleLeft}>
                                    <Ionicons name="medkit-outline" size={16} color={BRAND} />
                                    <Text style={st.sectionToggleText}>Informations médicales</Text>
                                </View>
                                <Ionicons name={medicalExpanded ? "chevron-up" : "chevron-down"} size={16} color={TXT_SEC} />
                            </TouchableOpacity>

                            {medicalExpanded && (
                                <View style={st.medicalSection}>
                                    <View style={st.fieldGroup}>
                                        <Text style={st.labelSm}>Établissement</Text>
                                        <View style={st.inputRow}>
                                            <TextInput style={st.inputText} value={medicalFacility} onChangeText={setMedicalFacility} placeholder="HUG, Clinique La Colline…" placeholderTextColor={TXT_MUTED} />
                                        </View>
                                    </View>
                                    <View style={st.fieldGroup}>
                                        <Text style={st.labelSm}>Service / Bât.</Text>
                                        <View style={st.inputRow}>
                                            <TextInput style={st.inputText} value={hospitalService} onChangeText={setHospitalService} placeholder="Ex: CHIR, Urgences…" placeholderTextColor={TXT_MUTED} />
                                        </View>
                                    </View>
                                    <View style={st.fieldGroup}>
                                        <Text style={st.labelSm}>Médecin</Text>
                                        <View style={st.inputRow}>
                                            <TextInput style={st.inputText} value={doctorName} onChangeText={setDoctorName} placeholder="Ex: Dr Dupont" placeholderTextColor={TXT_MUTED} />
                                        </View>
                                    </View>
                                    <View style={st.fieldGroup}>
                                        <Text style={st.labelSm}>Notes</Text>
                                        <View style={[st.inputRow, { minHeight: 60, alignItems: "flex-start" }]}>
                                            <TextInput style={[st.inputText, { textAlignVertical: "top", paddingTop: 8 }]} value={notesMedical} onChangeText={setNotesMedical} placeholder="Instructions, bâtiment, étage…" placeholderTextColor={TXT_MUTED} multiline />
                                        </View>
                                    </View>
                                    <View style={st.fieldGroup}>
                                        <Text style={st.labelSm}>Accès pickup</Text>
                                        <View style={st.inputRow}>
                                            <TextInput style={st.inputText} value={pickupAccessNotes} onChangeText={setPickupAccessNotes} placeholder="Ex: entrée arrière, sonner à…" placeholderTextColor={TXT_MUTED} />
                                        </View>
                                    </View>
                                    <View style={st.fieldGroup}>
                                        <Text style={st.labelSm}>Accès destination</Text>
                                        <View style={st.inputRow}>
                                            <TextInput style={st.inputText} value={dropoffAccessNotes} onChangeText={setDropoffAccessNotes} placeholder="Ex: entrée B, étage 2…" placeholderTextColor={TXT_MUTED} />
                                        </View>
                                    </View>
                                    <View style={st.fieldGroup}>
                                        <Text style={st.labelSm}>Chaise roulante</Text>
                                        <View style={st.chipsRow}>
                                            <TouchableOpacity
                                                style={[st.toggleChip, wheelchairClientHas && st.toggleChipActive]}
                                                onPress={() => { setWheelchairClientHas(!wheelchairClientHas); if (!wheelchairClientHas) setWheelchairNeed(false); }}
                                            >
                                                <Text style={[st.toggleChipText, wheelchairClientHas && st.toggleChipTextActive]}>En chaise</Text>
                                            </TouchableOpacity>
                                            <TouchableOpacity
                                                style={[st.toggleChip, wheelchairNeed && st.toggleChipActive]}
                                                onPress={() => { setWheelchairNeed(!wheelchairNeed); if (!wheelchairNeed) setWheelchairClientHas(false); }}
                                            >
                                                <Text style={[st.toggleChipText, wheelchairNeed && st.toggleChipTextActive]}>Fournir chaise</Text>
                                            </TouchableOpacity>
                                        </View>
                                    </View>
                                </View>
                            )}

                            {/* Priority */}
                            <View style={st.fieldGroup}>
                                <Text style={st.label}>Priorité</Text>
                                <View style={st.priorityRow}>
                                    {(["LOW", "NORMAL", "HIGH"] as const).map((p) => {
                                        const active = priority === p;
                                        const lbl = p === "LOW" ? "Basse" : p === "NORMAL" ? "Normale" : "Haute";
                                        const c = p === "HIGH" ? DANGER : p === "LOW" ? TXT_MUTED : BRAND;
                                        return (
                                            <TouchableOpacity
                                                key={p}
                                                style={[st.priorityBtn, active && { backgroundColor: c, borderColor: c }]}
                                                onPress={() => setPriority(p)}
                                            >
                                                <Text style={[st.priorityText, active && { color: "#FFF" }]}>{lbl}</Text>
                                            </TouchableOpacity>
                                        );
                                    })}
                                </View>
                            </View>

                            {/* Notes internes */}
                            <View style={st.fieldGroup}>
                                <Text style={st.label}>Notes internes</Text>
                                <NotesEditor label="" value={notes} onChange={setNotes} placeholder="Ajouter des notes..." />
                            </View>

                            <View style={{ height: 20 }} />
                        </ScrollView>
                    )}

                    {/* Footer */}
                    <View style={st.footer}>
                        <TouchableOpacity style={st.footerCancel} onPress={onClose} disabled={loading}>
                            <Text style={st.footerCancelText}>Annuler</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[st.footerSubmit, (!hasChanges() || loading) && st.footerSubmitDisabled]}
                            onPress={handleSave}
                            disabled={!hasChanges() || loading}
                            activeOpacity={0.85}
                        >
                            {loading ? (
                                <ActivityIndicator color="#FFF" size="small" />
                            ) : (
                                <>
                                    <Ionicons name="checkmark" size={16} color="#FFF" />
                                    <Text style={st.footerSubmitText}>Enregistrer</Text>
                                </>
                            )}
                        </TouchableOpacity>
                    </View>
                </View>
            </KeyboardAvoidingView>
        </Modal>
    );
};

const sheetShadow = createShadow({
    shadowColor: "#000",
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.12,
    shadowRadius: 24,
    elevation: 12,
});

const st = StyleSheet.create({
    root: { flex: 1, justifyContent: "flex-end" as const },
    overlay: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(0,0,0,0.35)" },
    sheet: {
        backgroundColor: CARD,
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        height: "90%",
        overflow: "hidden",
        ...sheetShadow,
    },
    handle: { width: 36, height: 4, borderRadius: 2, backgroundColor: "#D1D5DB", alignSelf: "center", marginTop: 10, marginBottom: 6 },

    header: { flexDirection: "row", alignItems: "center", gap: 10, paddingHorizontal: 20, paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: BORDER },
    headerIconWrap: { width: 36, height: 36, borderRadius: 10, backgroundColor: "rgba(0,121,107,0.08)", alignItems: "center", justifyContent: "center" },
    headerTitle: { fontSize: 16, fontWeight: "700", color: TXT },
    headerSub: { fontSize: 12, color: TXT_SEC, marginTop: 2 },
    statusBadge: { flexDirection: "row", alignItems: "center", gap: 5, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8 },
    statusDot: { width: 6, height: 6, borderRadius: 3 },
    statusLabel: { fontSize: 11, fontWeight: "600" },
    closeBtn: { padding: 4 },

    loadingWrap: { flex: 1, alignItems: "center", justifyContent: "center", gap: 12 },
    loadingText: { fontSize: 14, color: TXT_SEC },

    scroll: { flex: 1 },
    scrollContent: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 20 },

    warningBanner: { flexDirection: "row", alignItems: "center", gap: 8, backgroundColor: "rgba(245,158,11,0.08)", padding: 10, borderRadius: 10, marginBottom: 12, borderWidth: 1, borderColor: "rgba(245,158,11,0.2)" },
    warningText: { flex: 1, color: "#92400E", fontSize: 12, lineHeight: 16 },

    contextCard: { backgroundColor: BG, borderRadius: 12, borderWidth: 1, borderColor: BORDER, padding: 12, marginBottom: 14, gap: 6 },
    contextRow: { flexDirection: "row", alignItems: "center", gap: 8 },
    contextLabel: { fontSize: 12, color: TXT_SEC, fontWeight: "600", width: 70 },
    contextValue: { flex: 1, fontSize: 12, color: TXT, fontWeight: "500" },

    fieldGroup: { marginBottom: 14 },
    label: { fontSize: 13, fontWeight: "600", color: TXT, marginBottom: 6 },
    labelSm: { fontSize: 12, fontWeight: "600", color: TXT_SEC, marginBottom: 5 },
    req: { color: DANGER },
    inputRow: { flexDirection: "row", alignItems: "center", backgroundColor: BG, borderRadius: 10, borderWidth: 1, borderColor: BORDER, paddingHorizontal: 12, paddingVertical: 10, gap: 8 },
    inputText: { flex: 1, color: TXT, fontSize: 14, padding: 0 },

    swapBtn: { alignSelf: "center", width: 32, height: 32, borderRadius: 16, backgroundColor: "rgba(0,121,107,0.08)", alignItems: "center", justifyContent: "center", marginVertical: -4, marginBottom: 10 },

    presetRow: { flexDirection: "row", gap: 8, marginTop: 8 },
    presetChip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8, backgroundColor: BG, borderWidth: 1, borderColor: BORDER },
    presetText: { fontSize: 12, fontWeight: "600", color: BRAND },

    chipsRow: { flexDirection: "row", gap: 8, marginBottom: 14 },
    toggleChip: { flexDirection: "row", alignItems: "center", gap: 5, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, backgroundColor: BG, borderWidth: 1, borderColor: BORDER },
    toggleChipActive: { borderColor: BRAND, backgroundColor: "rgba(0,121,107,0.06)" },
    toggleChipText: { fontSize: 12, fontWeight: "600", color: TXT_SEC },
    toggleChipTextActive: { color: BRAND },

    subsection: { backgroundColor: "rgba(0,121,107,0.03)", borderRadius: 12, borderWidth: 1, borderColor: "rgba(0,121,107,0.1)", padding: 14, marginBottom: 14, gap: 10 },

    sectionToggle: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingVertical: 12, paddingHorizontal: 14, borderRadius: 12, backgroundColor: BG, borderWidth: 1, borderColor: BORDER, marginBottom: 10 },
    sectionToggleLeft: { flexDirection: "row", alignItems: "center", gap: 8 },
    sectionToggleText: { fontSize: 13, fontWeight: "600", color: TXT },

    medicalSection: { backgroundColor: "rgba(0,121,107,0.02)", borderRadius: 12, borderWidth: 1, borderColor: "rgba(0,121,107,0.08)", padding: 14, marginBottom: 14, gap: 4 },

    priorityRow: { flexDirection: "row", gap: 8 },
    priorityBtn: { flex: 1, paddingVertical: 10, borderRadius: 10, backgroundColor: BG, borderWidth: 1.5, borderColor: BORDER, alignItems: "center" },
    priorityText: { fontSize: 13, fontWeight: "600", color: TXT_SEC },

    footer: { flexDirection: "row", gap: 10, paddingHorizontal: 20, paddingVertical: 14, borderTopWidth: 1, borderTopColor: BORDER },
    footerCancel: { flex: 1, alignItems: "center", paddingVertical: 13, borderRadius: 12, borderWidth: 1, borderColor: BORDER, backgroundColor: CARD },
    footerCancelText: { color: TXT_SEC, fontWeight: "600", fontSize: 14 },
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
    footerSubmitDisabled: { opacity: 0.4 },
    footerSubmitText: { color: "#FFFFFF", fontWeight: "700", fontSize: 14 },
});
