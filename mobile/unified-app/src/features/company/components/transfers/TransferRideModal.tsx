import { Pressable, Text } from "react-native";
import { Button, Loader, Modal } from "../../../../components/ui";

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
      {pending ? <Loader /> : null}
      {!pending && options.length === 0 ? (
        <Text style={{ color: "#666" }}>Aucun partenaire disponible.</Text>
      ) : null}
      {options.map((company) => (
        <Pressable
          key={company.id}
          onPress={() => onSelect(company.id)}
          style={{
            borderWidth: 1,
            borderColor: selectedPartnerId === company.id ? "#0a7ea4" : "#ddd",
            borderRadius: 8,
            padding: 10,
          }}
        >
          <Text style={{ color: selectedPartnerId === company.id ? "#0a7ea4" : "#333" }}>
            {company.label}
          </Text>
        </Pressable>
      ))}
      <Button
        label={pending ? "Transfert..." : "Confirmer le transfert"}
        variant="primary"
        onPress={onConfirm}
        disabled={pending || selectedPartnerId == null}
      />
      {error ? <Text style={{ color: "#B00020" }}>{error}</Text> : null}
    </Modal>
  );
}
