import { useState } from "react";
import { TextInput, View } from "react-native";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";

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
        <AppText variant="body">Precisez une justification d&apos;annulation.</AppText>
        <TextInput
          value={reason}
          onChangeText={setReason}
          multiline
          placeholder="Raison d'annulation"
          style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10, minHeight: 90 }}
        />
        <AppButton
          title={pending ? "Annulation..." : "Confirmer l'annulation"}
          variant="secondary"
          onPress={() => onConfirm(reason)}
        />
      </View>
    </Modal>
  );
}
