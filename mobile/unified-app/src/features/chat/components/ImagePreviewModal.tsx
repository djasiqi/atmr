import { Image } from "react-native";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { resolveMediaUrl } from "../../../core/api/mediaUrl";

type ImagePreviewModalProps = {
  visible: boolean;
  imageUrl: string | null;
  onClose: () => void;
};

export function ImagePreviewModal({ visible, imageUrl, onClose }: ImagePreviewModalProps) {
  const resolved = imageUrl ? resolveMediaUrl(imageUrl) : null;
  return (
    <Modal visible={visible} title="Apercu image" onClose={onClose}>
      {resolved || imageUrl ? (
        <Image
          source={{ uri: (resolved ?? imageUrl) as string }}
          style={{ width: "100%", aspectRatio: 1, borderRadius: 8, backgroundColor: "#f2f2f2" }}
        />
      ) : (
        <AppText variant="bodyMuted">Aucune image selectionnee.</AppText>
      )}
      <AppButton title="Fermer" variant="secondary" onPress={onClose} />
    </Modal>
  );
}
