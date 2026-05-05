import { useEffect, useState } from "react";
import { View } from "react-native";
import { AppButton, Modal, useResponsiveTokens } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { isFeatureEnabled } from "../../../../core/featureFlags/registry";
import { normalizeScheduledTimeIso, useRideEdit, useRideFormState } from "../../useRideForms";
import { AddressSelector } from "./AddressSelector";
import { ClientSelector } from "./ClientSelector";
import { TimeDatePicker } from "./TimeDatePicker";

type RideEditModalProps = {
  visible: boolean;
  missionId: number | null;
  /** True : course invitée (pas de sélecteur client, édition de cette seule course). */
  isGuestMission?: boolean;
  initial: {
    clientId?: number | null;
    pickup?: string | null;
    dropoff?: string | null;
    scheduledAt?: string | null;
    notes?: string | null;
  } | null;
  onClose: () => void;
  onSaved?: () => void;
};

export function RideEditModal({
  visible,
  missionId,
  initial,
  isGuestMission = false,
  onClose,
  onSaved,
}: RideEditModalProps) {
  const t = useResponsiveTokens();
  const editRide = useRideEdit();
  const form = useRideFormState();
  const [error, setError] = useState<string | null>(null);
  const structuredPayloadEnabled = isFeatureEnabled("company_mobile_structured_ride_payload_enabled");
  const {
    setClientId,
    setPickup,
    setDropoff,
    setScheduledAt,
    setNotes,
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
    setNotes(initial.notes ?? "");
    setRecurrence("none");
  }, [
    initial,
    isGuestMission,
    scheduledAt,
    setClientId,
    setDropoff,
    setNotes,
    setPickup,
    setRecurrence,
    setScheduledAt,
    visible,
  ]);

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
          notes: form.notes.trim() || null,
        },
      });
      setError(null);
      onSaved?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Edition ride impossible.");
    }
  };

  return (
    <Modal visible={visible} title="Modifier la mission" onClose={onClose}>
      <View style={{ gap: t.sectionGap }}>
        {isGuestMission ? (
          <View style={{ paddingVertical: 2 }}>
            <AppText variant="bodyMuted" style={{ lineHeight: 18 }}>
              Course invitée (sans fiche client). Vous modifiez uniquement cette course, pas de série
              récurrente ici.
            </AppText>
          </View>
        ) : (
          <ClientSelector value={form.clientId} onChange={form.setClientId} />
        )}
        <AddressSelector
          label="Prise en charge"
          value={form.pickup}
          onChange={form.setPickup}
          onSelectAddress={form.selectPickupAddress}
          placeholder="Adresse de prise en charge"
        />
        <AddressSelector
          label="Destination"
          value={form.dropoff}
          onChange={form.setDropoff}
          onSelectAddress={form.selectDropoffAddress}
          placeholder="Adresse de destination"
        />
        <TimeDatePicker value={form.scheduledAt} onChange={form.setScheduledAt} />
        <AppInput
          value={form.notes}
          onChangeText={form.setNotes}
          placeholder="Remarques (optionnel)"
          multiline
          textAlignVertical="top"
          shellStyle={{ borderRadius: 14, minHeight: 100, alignItems: "flex-start" }}
          style={{ minHeight: 80 }}
        />
        <AppButton
          title={editRide.isPending ? "Enregistrement…" : "Enregistrer"}
          variant="primary"
          onPress={() => void submit()}
          disabled={!missionId}
          style={{ minHeight: 54, borderRadius: 14 }}
        />
        {error ? <AppText variant="error">{error}</AppText> : null}
      </View>
    </Modal>
  );
}
