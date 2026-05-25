import { useEffect, useMemo, useRef, useState } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, Modal } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";
import { isFeatureEnabled } from "../../../../core/featureFlags/registry";
import { useCompanyRideDetailsQuery } from "../../hooks";
import {
  normalizeScheduledTimeIso,
  rideMissionMedicalTripFields,
  useRideEdit,
  useRideFormState,
} from "../../useRideForms";
import { AddressSelector } from "./AddressSelector";
import { ClientSelector } from "./ClientSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

const ROW_RADIUS = 12;
const FIELD_ICON_SIZE = 18;
const COMPACT_CONTROL_RADIUS = 11;
const COMPACT_CHIP_HEIGHT = 32;
const COMPACT_MULTILINE_MEDIUM_HEIGHT = 72;
const COMPACT_MULTILINE_MEDIUM_INPUT_HEIGHT = 56;

type RideEditModalProps = {
  visible: boolean;
  missionId: number | null;
  /** Date journée dispatch (requise pour charger le détail mission et préremplir les champs médicaux). */
  detailDate: string;
  /** True : course invitée (pas de sélecteur client, édition de cette seule course). */
  isGuestMission?: boolean;
  initial: {
    clientId?: number | null;
    /** Libellé affiché dans l’entête (ex. `client_name` de la liste dispatch). */
    clientLabel?: string | null;
    pickup?: string | null;
    dropoff?: string | null;
    scheduledAt?: string | null;
    notes?: string | null;
  } | null;
  onClose: () => void;
  onSaved?: () => void;
};

const BACK_BOX = {
  width: 40,
  height: 40,
  borderRadius: 12,
  backgroundColor: "transparent",
  alignItems: "center" as const,
  justifyContent: "center" as const,
};

const OUTLINE_SECONDARY = {
  borderColor: "rgba(0, 121, 107, 0.28)",
  backgroundColor: "#fff",
} as const;

const s = StyleSheet.create({
  form: { gap: 12 },
  sectionBlock: { gap: 6 },
  sectionLabel: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "600" as const,
    color: E.TEXT,
    marginBottom: 2,
  },
  sectionHelper: {
    fontSize: FONT_SIZE.px12,
    color: E.TEXT_MUTED,
    lineHeight: 17,
  },
  sectionDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(148, 163, 184, 0.24)",
    marginVertical: 0,
  },
  card: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    backgroundColor: "#FAFBFA",
    paddingHorizontal: 10,
    paddingVertical: 10,
    gap: 8,
  },
  pickupDropoffRow: {
    flexDirection: "row" as const,
    alignItems: "flex-start" as const,
    columnGap: 4,
  },
  addressFieldsColumn: {
    width: "82%",
    minWidth: 0,
    gap: 10,
  },
  swapColumn: {
    width: "18%",
    minWidth: 52,
    maxWidth: 64,
    alignItems: "center" as const,
    justifyContent: "flex-start" as const,
    paddingTop: 50,
    marginLeft: 0,
  },
  swapBtn: {
    width: 32,
    height: 32,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.28)",
    backgroundColor: "#FFFFFF",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  headerRow: {
    flexDirection: "row" as const,
    alignItems: "flex-start" as const,
    gap: 12,
    marginBottom: 2,
    paddingBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(148, 163, 184, 0.28)",
  },
  headerCenter: { flex: 1, gap: 4 },
  headerTitle: {
    fontSize: FONT_SIZE.px18,
    fontWeight: "700" as const,
    color: E.TEXT,
    letterSpacing: 0.15,
  },
  headerClientLine: {
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    fontWeight: "600" as const,
    color: E.TEXT,
  },
  footerRow: {
    gap: 8,
    paddingTop: 8,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(148, 163, 184, 0.28)",
  },
  footerSummary: {
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    color: E.TEXT_MUTED,
    fontWeight: "600" as const,
  },
  footerHint: {
    fontSize: FONT_SIZE.px12,
    color: E.TEXT_MUTED,
    textAlign: "left" as const,
  },
  footerButtons: {
    flexDirection: "row" as const,
    gap: 8,
  },
  tertiaryCard: {
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    backgroundColor: "#FAFBFA",
    paddingHorizontal: 10,
    paddingVertical: 10,
    gap: 8,
  },
  wheelchairRow: {
    flexDirection: "row" as const,
    gap: 8,
    flexWrap: "wrap" as const,
  },
  chip: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: COMPACT_CONTROL_RADIUS,
    borderWidth: StyleSheet.hairlineWidth,
    minHeight: COMPACT_CHIP_HEIGHT,
  },
  chipOn: {
    backgroundColor: "rgba(0, 121, 107, 0.14)",
    borderColor: E.BRAND,
  },
  chipOff: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(0, 121, 107, 0.28)",
  },
  chipLabelOn: { color: E.BRAND, fontWeight: "700" as const, fontSize: FONT_SIZE.px13, lineHeight: 16 },
  chipLabelOff: {
    color: E.TEXT_SEC,
    fontWeight: "600" as const,
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
  },
  footerBtn: {
    flex: 1,
    minHeight: 48,
    borderRadius: 12,
  },
  compactAddressContainer: { gap: 8 },
  compactAddressShell: {
    minHeight: 32,
    paddingHorizontal: 4,
    borderRadius: 12,
    borderColor: "rgba(145, 165, 157, 0.38)",
  },
  compactAddressInput: {
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    paddingVertical: 2,
  },
});

export function RideEditModal({
  visible,
  missionId,
  detailDate,
  initial,
  isGuestMission = false,
  onClose,
  onSaved,
}: RideEditModalProps) {
  const editRide = useRideEdit();
  const form = useRideFormState();
  const [error, setError] = useState<string | null>(null);
  const medicalHydratedMissionRef = useRef<number | null>(null);
  const structuredPayloadEnabled = isFeatureEnabled("company_mobile_structured_ride_payload_enabled");
  const rideDetailQuery = useCompanyRideDetailsQuery({
    date: detailDate,
    rideId: visible && missionId != null ? missionId : null,
  });
  const {
    setClientId,
    setPickup,
    setDropoff,
    setScheduledAt,
    setInternalNotes,
    setEstablishment,
    setHospitalService,
    setDoctorName,
    setNotesMedical,
    setPickupAccessNotes,
    setDropoffAccessNotes,
    setWheelchairClient,
    setWheelchairProvide,
    setRecurrence,
    scheduledAt,
  } = form;

  useEffect(() => {
    if (!visible || !initial) return;
    if (!isGuestMission) {
      setClientId(initial.clientId ?? null);
    } else {
      setClientId(null);
    }
    setPickup(initial.pickup ?? "");
    setDropoff(initial.dropoff ?? "");
    setScheduledAt(initial.scheduledAt ?? scheduledAt);
    setInternalNotes(initial.notes ?? "");
    setRecurrence("none");
  }, [
    initial,
    isGuestMission,
    scheduledAt,
    setClientId,
    setDropoff,
    setInternalNotes,
    setPickup,
    setRecurrence,
    setScheduledAt,
    visible,
  ]);

  useEffect(() => {
    if (!visible || missionId == null) return;
    medicalHydratedMissionRef.current = null;
    setEstablishment("");
    setHospitalService("");
    setDoctorName("");
    setNotesMedical("");
    setPickupAccessNotes("");
    setDropoffAccessNotes("");
    setWheelchairClient(false);
    setWheelchairProvide(false);
  }, [
    missionId,
    setDoctorName,
    setDropoffAccessNotes,
    setEstablishment,
    setHospitalService,
    setNotesMedical,
    setPickupAccessNotes,
    setWheelchairClient,
    setWheelchairProvide,
    visible,
  ]);

  useEffect(() => {
    if (!visible) {
      medicalHydratedMissionRef.current = null;
      return;
    }
    if (missionId == null) return;
    const row = rideDetailQuery.data;
    if (!row || typeof row !== "object") return;
    if (medicalHydratedMissionRef.current === missionId) return;
    medicalHydratedMissionRef.current = missionId;
    const m = rideMissionMedicalTripFields(row as Record<string, unknown>);
    setNotesMedical(m.notesMedical);
    setPickupAccessNotes(m.pickupAccessNotes);
    setDropoffAccessNotes(m.dropoffAccessNotes);
    setEstablishment(m.establishment);
    setHospitalService(m.hospitalService);
    setDoctorName(m.doctorName);
    setWheelchairClient(m.wheelchairClient);
    setWheelchairProvide(m.wheelchairProvide);
  }, [
    missionId,
    rideDetailQuery.data,
    setDoctorName,
    setDropoffAccessNotes,
    setEstablishment,
    setHospitalService,
    setNotesMedical,
    setPickupAccessNotes,
    setWheelchairClient,
    setWheelchairProvide,
    visible,
  ]);

  const canSubmit = useMemo(() => {
    if (!missionId) return false;
    if (!form.pickup.trim() || !form.dropoff.trim()) return false;
    if (!isGuestMission && !form.clientId) return false;
    return true;
  }, [form.pickup, form.dropoff, form.clientId, isGuestMission, missionId]);

  const submit = async () => {
    if (!missionId) return;
    if (!form.pickup.trim() || !form.dropoff.trim()) {
      setError("La prise en charge et la destination sont requises.");
      return;
    }
    if (!isGuestMission && !form.clientId) {
      setError("Sélectionnez un client.");
      return;
    }
    try {
      const pickupPayload =
        structuredPayloadEnabled && form.pickupAddress
          ? {
              label: form.pickupAddress.label,
              place_id: form.pickupAddress.placeId,
              lat: form.pickupAddress.latitude,
              lon: form.pickupAddress.longitude,
            }
          : form.pickup.trim();
      const dropoffPayload =
        structuredPayloadEnabled && form.dropoffAddress
          ? {
              label: form.dropoffAddress.label,
              place_id: form.dropoffAddress.placeId,
              lat: form.dropoffAddress.latitude,
              lon: form.dropoffAddress.longitude,
            }
          : form.dropoff.trim();
      await editRide.mutateAsync({
        missionId,
        payload: {
          client_id: form.clientId,
          pickup_address: pickupPayload,
          dropoff_address: dropoffPayload,
          pickup_lat: form.pickupAddress?.latitude ?? null,
          pickup_lon: form.pickupAddress?.longitude ?? null,
          dropoff_lat: form.dropoffAddress?.latitude ?? null,
          dropoff_lon: form.dropoffAddress?.longitude ?? null,
          scheduled_time: normalizeScheduledTimeIso(form.scheduledAt),
          recurrence: form.recurrence === "none" ? null : form.recurrence,
          notes_medical: form.notesMedical.trim() || null,
          pickup_access_notes: form.pickupAccessNotes.trim() || null,
          dropoff_access_notes: form.dropoffAccessNotes.trim() || null,
          wheelchair_client_has: Boolean(form.wheelchairClient),
          wheelchair_need: Boolean(form.wheelchairProvide),
          notes: form.internalNotes.trim() || null,
        },
      });
      setError(null);
      onSaved?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Edition ride impossible.");
    }
  };

  const clientHeaderLabel = initial?.clientLabel?.trim() ?? "";

  const header = () => (
    <View style={s.headerRow}>
      <Pressable
        onPress={onClose}
        style={BACK_BOX}
        accessibilityRole="button"
        accessibilityLabel="Retour"
      >
        <Ionicons name="chevron-back" size={22} color={E.BRAND} />
      </Pressable>
      <View style={s.headerCenter}>
        <AppText style={s.headerTitle}>Modifier la mission</AppText>
        {clientHeaderLabel ? <AppText style={s.headerClientLine}>{clientHeaderLabel}</AppText> : null}
      </View>
    </View>
  );

  const footer = (
    <View style={s.footerRow}>
      <AppText style={s.footerSummary}>
        {form.pickup.trim() && form.dropoff.trim()
          ? "Vérifiez les adresses et l'heure avant enregistrement."
          : "Complétez les champs requis avant d'enregistrer."}
      </AppText>
      {!canSubmit && !editRide.isPending ? (
        <AppText style={s.footerHint}>
          {isGuestMission
            ? "Renseignez la prise en charge et la destination."
            : "Sélectionnez un client ainsi que les deux lieux du trajet."}
        </AppText>
      ) : null}
      <View style={s.footerButtons}>
        <AppButton
          title="Fermer"
          variant="secondary"
          onPress={onClose}
          style={{ ...s.footerBtn, ...OUTLINE_SECONDARY }}
        />
        <AppButton
          title={editRide.isPending ? "Enregistrement…" : "Enregistrer"}
          variant="primary"
          onPress={() => void submit()}
          disabled={!canSubmit || editRide.isPending}
          loading={editRide.isPending}
          style={s.footerBtn}
          leftIcon={
            <Ionicons
              name="checkmark-circle-outline"
              size={20}
              color={!canSubmit || editRide.isPending ? "rgba(255,255,255,0.85)" : "#fff"}
            />
          }
        />
      </View>
      {error ? <AppText variant="error">{error}</AppText> : null}
    </View>
  );

  return (
    <Modal
      visible={visible}
      title=""
      onClose={onClose}
      presentation="bottomSheet"
      renderHeader={header}
      footer={footer}
      sheetBodyMaxHeightRatio={0.74}
    >
      <View style={s.form}>
        {!isGuestMission ? (
          <View style={s.sectionBlock}>
            <AppText style={s.sectionLabel}>Client</AppText>
            <ClientSelector value={form.clientId} onChange={form.setClientId} />
          </View>
        ) : null}
        <View style={s.card}>
          <View style={s.pickupDropoffRow}>
            <View style={s.addressFieldsColumn}>
              <View style={s.sectionBlock}>
                <AppText style={s.sectionLabel}>Prise en charge</AppText>
                <AddressSelector
                  label=""
                  value={form.pickup}
                  onChange={form.setPickup}
                  onSelectAddress={form.selectPickupAddress}
                  placeholder="Adresse de prise en charge"
                  leftSlot={<Ionicons name="navigate-outline" size={16} color={E.TEXT_SEC} />}
                  containerStyle={s.compactAddressContainer}
                  shellStyle={s.compactAddressShell}
                  inputStyle={s.compactAddressInput}
                />
              </View>
              <View style={s.sectionBlock}>
                <AppText style={s.sectionLabel}>Destination</AppText>
                <AddressSelector
                  label=""
                  value={form.dropoff}
                  onChange={form.setDropoff}
                  onSelectAddress={form.selectDropoffAddress}
                  placeholder="Adresse de destination"
                  leftSlot={<Ionicons name="location-outline" size={16} color={E.TEXT_SEC} />}
                  containerStyle={s.compactAddressContainer}
                  shellStyle={s.compactAddressShell}
                  inputStyle={s.compactAddressInput}
                />
              </View>
            </View>
            <View style={s.swapColumn}>
              <Pressable
                onPress={form.swapAddresses}
                style={s.swapBtn}
                accessibilityRole="button"
                accessibilityLabel="Inverser prise en charge et destination"
              >
                <Ionicons name="swap-vertical-outline" size={16} color={E.BRAND} />
              </Pressable>
            </View>
          </View>
        </View>
        <View style={s.card}>
          <TimeDatePicker value={form.scheduledAt} onChange={form.setScheduledAt} />
        </View>
        <View style={s.tertiaryCard}>
          <AppInput
            value={form.establishment}
            onChangeText={form.setEstablishment}
            placeholder="Établissement (optionnel)"
            leftSlot={<Ionicons name="business-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
            shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
          />
          <AppInput
            value={form.hospitalService}
            onChangeText={form.setHospitalService}
            placeholder="Service hospitalier (optionnel)"
            leftSlot={<Ionicons name="medkit-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
            shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
          />
          <AppInput
            value={form.doctorName}
            onChangeText={form.setDoctorName}
            placeholder="Médecin référent (optionnel)"
            leftSlot={<Ionicons name="person-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
            shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
          />
          <AppInput
            value={form.notesMedical}
            onChangeText={form.setNotesMedical}
            placeholder="Instructions particulières, bâtiment, étage…"
            multiline
            textAlignVertical="top"
            shellStyle={{
              borderRadius: ROW_RADIUS,
              minHeight: COMPACT_MULTILINE_MEDIUM_HEIGHT,
              alignItems: "flex-start",
              backgroundColor: "#FFFFFF",
            }}
            style={{ minHeight: COMPACT_MULTILINE_MEDIUM_INPUT_HEIGHT }}
          />
          <View style={s.sectionBlock}>
            <AppText style={s.sectionLabel}>Accès pickup</AppText>
            <AppInput
              value={form.pickupAccessNotes}
              onChangeText={form.setPickupAccessNotes}
              placeholder="Ex: entrée arrière, sonner à…, appeler avant…"
              leftSlot={<Ionicons name="navigate-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
              shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
            />
          </View>
          <View style={s.sectionBlock}>
            <AppText style={s.sectionLabel}>Accès destination</AppText>
            <AppInput
              value={form.dropoffAccessNotes}
              onChangeText={form.setDropoffAccessNotes}
              placeholder="Ex: entrée B, étage 2, service…, appeler secrétariat…"
              leftSlot={<Ionicons name="location-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
              shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
            />
          </View>
          <AppText style={s.sectionLabel}>Chaise roulante</AppText>
          <View style={s.wheelchairRow}>
            <Pressable
              onPress={() => form.setWheelchairClient(!form.wheelchairClient)}
              style={[s.chip, form.wheelchairClient ? s.chipOn : s.chipOff]}
              accessibilityRole="button"
              accessibilityState={{ selected: form.wheelchairClient }}
            >
              <AppText style={form.wheelchairClient ? s.chipLabelOn : s.chipLabelOff}>
                {"♿"} En chaise
              </AppText>
            </Pressable>
            <Pressable
              onPress={() => form.setWheelchairProvide(!form.wheelchairProvide)}
              style={[s.chip, form.wheelchairProvide ? s.chipOn : s.chipOff]}
              accessibilityRole="button"
              accessibilityState={{ selected: form.wheelchairProvide }}
            >
              <AppText style={form.wheelchairProvide ? s.chipLabelOn : s.chipLabelOff}>
                {"🏥"} Fournir chaise
              </AppText>
            </Pressable>
          </View>
        </View>
        <View style={s.sectionDivider} />
        <View style={s.sectionBlock}>
          <AppText style={s.sectionLabel}>Remarques</AppText>
          <AppInput
            value={form.internalNotes}
            onChangeText={form.setInternalNotes}
            placeholder="Remarques (optionnel)"
            multiline
            textAlignVertical="top"
            shellStyle={{ borderRadius: 14, minHeight: 100, alignItems: "flex-start" }}
            style={{ minHeight: 80 }}
          />
        </View>
      </View>
    </Modal>
  );
}
