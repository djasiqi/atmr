import { Pressable } from "react-native";
import { AppButton, AppSpinner, Modal, brandPrimary, brandText } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";

type TransferOption = { id: number; label: string };

type TransferRideModalProps = {
  visible: boolean;
  pending?: boolean;
  options: TransferOption[];
  selectedPartnerId: number | null;
  error: string | null;
  onSelect: (id: number) => void;
  onConfirm: () => void;
  onClose: () => void;
};

export function TransferRideModal({
  visible,
  pending = false,
  options,
  selectedPartnerId,
  error,
  onSelect,
  onConfirm,
  onClose,
}: TransferRideModalProps) {
  return (
    <Modal visible={visible} title="Transferer la course" onClose={onClose}>
      {pending ? <AppSpinner size="small" /> : null}
      {!pending && options.length === 0 ? (
        <AppText variant="bodyMuted">Aucun partenaire disponible.</AppText>
      ) : null}
      {options.map((company) => (
        <Pressable
          key={company.id}
          onPress={() => onSelect(company.id)}
          style={{
            borderWidth: 1,
            borderColor: selectedPartnerId === company.id ? "#0A8F7A" : "#ddd",
            borderRadius: 8,
            padding: 10,
          }}
        >
          <AppText
            variant="body"
            style={{ color: selectedPartnerId === company.id ? brandPrimary : brandText }}
          >
            {company.label}
          </AppText>
        </Pressable>
      ))}
      <AppButton
        title={pending ? "Transfert..." : "Confirmer le transfert"}
        variant="primary"
        onPress={onConfirm}
        disabled={pending || selectedPartnerId == null}
      />
      {error ? <AppText variant="error">{error}</AppText> : null}
    </Modal>
  );
}
