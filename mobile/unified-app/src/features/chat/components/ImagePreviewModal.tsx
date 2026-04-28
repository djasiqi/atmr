import { Image, Text } from "react-native";
import { Button, Modal } from "../../../components/ui";
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
        <Text style={{ color: "#666" }}>Aucune image selectionnee.</Text>
      )}
      <Button label="Fermer" onPress={onClose} />
    </Modal>
  );
}
