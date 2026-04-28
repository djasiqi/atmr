import { Button, Modal } from "../../../components/ui";

type AttachmentSheetProps = {
  visible: boolean;
  onClose: () => void;
  onPickImage?: () => void;
  onPickPdf?: () => void;
};

export function AttachmentSheet({ visible, onClose, onPickImage, onPickPdf }: AttachmentSheetProps) {
  return (
    <Modal visible={visible} title="Ajouter une piece jointe" onClose={onClose}>
      <Button label="Choisir une image" onPress={onPickImage} />
      <Button label="Choisir un PDF" onPress={onPickPdf} />
      <Button label="Fermer" variant="secondary" onPress={onClose} />
    </Modal>
  );
}
