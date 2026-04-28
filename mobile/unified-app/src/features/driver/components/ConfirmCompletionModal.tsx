import { Text, View } from "react-native";
import { Button, Modal } from "../../../components/ui";

type ConfirmCompletionModalProps = {
  visible: boolean;
  missionId: number | null;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

export function ConfirmCompletionModal(props: ConfirmCompletionModalProps) {
  return (
    <Modal visible={props.visible} title="Confirmer la fin de mission" onClose={props.onCancel}>
      <View style={{ gap: 8 }}>
        <Text>
          Confirmer le passage de la mission #{props.missionId ?? "n/a"} au statut COMPLETED ?
        </Text>
        <View style={{ flexDirection: "row", gap: 8 }}>
          <Button label="Annuler" onPress={props.onCancel} disabled={props.pending} />
          <Button
            label={props.pending ? "Validation..." : "Confirmer"}
            variant="primary"
            onPress={props.onConfirm}
            disabled={props.pending}
          />
        </View>
      </View>
    </Modal>
  );
}
