import { View } from "react-native";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";

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
        <AppText variant="body">
          Confirmer le passage de la mission #{props.missionId ?? "n/a"} au statut COMPLETED ?
        </AppText>
        <View style={{ flexDirection: "row", gap: 8 }}>
          <AppButton title="Annuler" variant="secondary" onPress={props.onCancel} disabled={props.pending} />
          <AppButton
            title={props.pending ? "Validation..." : "Confirmer"}
            variant="primary"
            onPress={props.onConfirm}
            disabled={props.pending}
          />
        </View>
      </View>
    </Modal>
  );
}
