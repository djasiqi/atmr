import { AppButton, Modal } from "../../../design/responsive";

type AttachmentSheetProps = {
  visible: boolean;
  onClose: () => void;
  onPickImage?: () => void;
  onPickPdf?: () => void;
};

export function AttachmentSheet({ visible, onClose, onPickImage, onPickPdf }: AttachmentSheetProps) {
  return (
    <Modal visible={visible} title="Ajouter une piece jointe" onClose={onClose}>
      <AppButton title="Choisir une image" variant="secondary" onPress={onPickImage} />
      <AppButton title="Choisir un PDF" variant="secondary" onPress={onPickPdf} />
      <AppButton title="Fermer" variant="secondary" onPress={onClose} />
    </Modal>
  );
}
