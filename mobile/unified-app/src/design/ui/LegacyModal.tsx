import type { PropsWithChildren } from "react";
import { Modal as NativeModal, Pressable, View } from "react-native";
import { brandPrimary } from "../responsive/colors";
import { AppText } from "./AppText";

export type LegacyModalProps = PropsWithChildren<{
  visible: boolean;
  title: string;
  onClose: () => void;
}>;

/** Modal centrée simple (titres + actions legacy). */
export function Modal({ visible, title, onClose, children }: LegacyModalProps) {
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
          <AppText variant="sectionTitle">{title}</AppText>
          {children}
          <Pressable onPress={onClose}>
            <AppText variant="body" style={{ color: brandPrimary, fontWeight: "600" }}>
              Fermer
            </AppText>
          </Pressable>
        </View>
      </View>
    </NativeModal>
  );
}
