import { Modal, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { M } from "../messagingTheme";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

export type InboxMenuItem = {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
};

type Props = {
  visible: boolean;
  horizontalPadding: number;
  items: InboxMenuItem[];
  onClose: () => void;
};

export function MessagesInboxMenu({ visible, horizontalPadding, items, onClose }: Props) {
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <Pressable style={styles.backdrop} onPress={onClose}>
        <View style={[styles.sheet, { marginRight: horizontalPadding }]}>
          {items.map((item) => (
            <Pressable
              key={item.label}
              style={styles.item}
              onPress={() => {
                onClose();
                item.onPress();
              }}
            >
              <Ionicons name={item.icon} size={20} color={M.TEXT} />
              <AppText variant="body" style={styles.label}>
                {item.label}
              </AppText>
            </Pressable>
          ))}
        </View>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.25)",
    justifyContent: "flex-start",
    alignItems: "flex-end",
    paddingTop: 56,
  },
  sheet: {
    backgroundColor: M.CARD,
    borderRadius: 12,
    paddingVertical: 8,
    minWidth: 240,
    elevation: 6,
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 12,
  },
  item: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  label: { fontSize: FONT_SIZE.px15, color: M.TEXT },
});
