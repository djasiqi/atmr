import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";

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
      <AppText variant="bodyMuted">{filename ?? "Document sans nom"}</AppText>
      <AppText variant="caption" numberOfLines={2} style={{ marginTop: 4 }}>
        {pdfUrl ?? "URL indisponible"}
      </AppText>
      <AppButton title="Ouvrir PDF" variant="secondary" onPress={onOpenPdf} disabled={!pdfUrl} />
      <AppButton title="Fermer" variant="secondary" onPress={onClose} />
    </Modal>
  );
}
