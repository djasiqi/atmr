import { useState } from "react";
import { Text, TextInput, View } from "react-native";
import { Button, Modal } from "../../../components/ui";

type CancelJustificationModalProps = {
  visible: boolean;
  pending?: boolean;
  onCancel: () => void;
  onConfirm: (reason: string) => void;
};

export function CancelJustificationModal({
  visible,
  pending = false,
  onCancel,
  onConfirm,
}: CancelJustificationModalProps) {
  const [reason, setReason] = useState("");

  return (
    <Modal visible={visible} title="Annuler la mission" onClose={onCancel}>
      <View style={{ gap: 8 }}>
        <Text>Precisez une justification d&apos;annulation.</Text>
        <TextInput
          value={reason}
          onChangeText={setReason}
          multiline
          placeholder="Raison d'annulation"
          style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10, minHeight: 90 }}
        />
        <Button label={pending ? "Annulation..." : "Confirmer l'annulation"} onPress={() => onConfirm(reason)} />
      </View>
    </Modal>
  );
}
