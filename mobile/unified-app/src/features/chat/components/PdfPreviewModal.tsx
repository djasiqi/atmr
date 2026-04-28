import { Text } from "react-native";
import { Button, Modal } from "../../../components/ui";

type PdfPreviewModalProps = {
  visible: boolean;
  filename: string | null;
  pdfUrl: string | null;
  onOpenPdf?: () => void;
  onClose: () => void;
};

export function PdfPreviewModal({
  visible,
  filename,
  pdfUrl,
  onOpenPdf,
  onClose,
}: PdfPreviewModalProps) {
  return (
    <Modal visible={visible} title="Apercu document PDF" onClose={onClose}>
      <Text style={{ color: "#666" }}>{filename ?? "Document sans nom"}</Text>
      <Text numberOfLines={2} style={{ color: "#999", fontSize: 12 }}>
        {pdfUrl ?? "URL indisponible"}
      </Text>
      <Button label="Ouvrir PDF" onPress={onOpenPdf} disabled={!pdfUrl} />
      <Button label="Fermer" variant="secondary" onPress={onClose} />
    </Modal>
  );
}
