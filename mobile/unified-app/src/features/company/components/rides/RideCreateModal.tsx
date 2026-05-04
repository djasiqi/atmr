import { useState } from "react";
import { TextInput, View } from "react-native";
import { AppButton, Modal } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { isFeatureEnabled } from "../../../../core/featureFlags/registry";
import { useRideCreate, useRideFormState } from "../../useRideForms";
import { AddressSelector } from "./AddressSelector";
import { ClientSelector } from "./ClientSelector";
import { RecurrenceSelector } from "./RecurrenceSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { ClientCreateModal } from "./ClientCreateModal";

type RideCreateModalProps = {
  visible: boolean;
  onClose: () => void;
  onCreated?: () => void;
};

export function RideCreateModal({ visible, onClose, onCreated }: RideCreateModalProps) {
  const createRide = useRideCreate();
  const form = useRideFormState();
  const [error, setError] = useState<string | null>(null);
  const [createClientVisible, setCreateClientVisible] = useState(false);
  const structuredPayloadEnabled = isFeatureEnabled("company_mobile_structured_ride_payload_enabled");

  const submit = async () => {
    if (!form.clientId || !form.pickup.trim() || !form.dropoff.trim()) {
      setError("Client, pickup et dropoff sont requis.");
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
      await createRide.mutateAsync({
        client_id: form.clientId,
        pickup_address: pickupPayload,
        dropoff_address: dropoffPayload,
        pickup_lat: form.pickupAddress?.latitude ?? null,
        pickup_lon: form.pickupAddress?.longitude ?? null,
        dropoff_lat: form.dropoffAddress?.latitude ?? null,
        dropoff_lon: form.dropoffAddress?.longitude ?? null,
        pickup_at: form.scheduledAt,
        recurrence: form.recurrence === "none" ? null : form.recurrence,
        notes: form.notes.trim() || null,
      });
      form.reset();
      setError(null);
      onCreated?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Creation ride impossible.");
    }
  };

  return (
    <>
      <Modal visible={visible} title="Creer une mission" onClose={onClose}>
        <View style={{ gap: 8 }}>
          <ClientSelector value={form.clientId} onChange={form.setClientId} />
          <AppButton title="Nouveau client" variant="secondary" onPress={() => setCreateClientVisible(true)} />
          <AddressSelector
            label="Pickup"
            value={form.pickup}
            onChange={form.setPickup}
            onSelectAddress={form.selectPickupAddress}
          />
          <AddressSelector
            label="Dropoff"
            value={form.dropoff}
            onChange={form.setDropoff}
            onSelectAddress={form.selectDropoffAddress}
          />
          <TimeDatePicker value={form.scheduledAt} onChange={form.setScheduledAt} />
          <RecurrenceSelector value={form.recurrence} onChange={form.setRecurrence} />
          <TextInput
            value={form.notes}
            onChangeText={form.setNotes}
            placeholder="Notes"
            multiline
            style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10, minHeight: 80 }}
          />
          <AppButton
            title={createRide.isPending ? "Creation..." : "Creer la mission"}
            variant="primary"
            onPress={() => void submit()}
          />
          {error ? <AppText variant="error">{error}</AppText> : null}
        </View>
      </Modal>
      <ClientCreateModal
        visible={createClientVisible}
        onClose={() => setCreateClientVisible(false)}
        onCreated={() => setError(null)}
      />
    </>
  );
}
