import { Modal as NativeModal, Pressable, Text, View } from "react-native";
import { PropsWithChildren } from "react";

type ModalProps = PropsWithChildren<{
  visible: boolean;
  title: string;
  onClose: () => void;
}>;

export function Modal({ visible, title, onClose, children }: ModalProps) {
  return (
    <NativeModal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View
        style={{
          flex: 1,
          backgroundColor: "rgba(0,0,0,0.35)",
          justifyContent: "center",
          padding: 20,
        }}
      >
        <View style={{ backgroundColor: "#fff", borderRadius: 12, padding: 16, gap: 12 }}>
          <Text style={{ fontWeight: "700", fontSize: 16 }}>{title}</Text>
          {children}
          <Pressable onPress={onClose}>
            <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Fermer</Text>
          </Pressable>
        </View>
      </View>
    </NativeModal>
  );
}

