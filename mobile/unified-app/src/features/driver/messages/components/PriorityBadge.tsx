import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { MessagePriority } from "../types";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

const COLORS: Record<MessagePriority, string> = {
  normal: "#047857",
  important: "#b45309",
  urgent: "#b91c1c",
};

type Props = { priority: MessagePriority };

export function PriorityBadge({ priority }: Props) {
  if (priority === "normal") return null;
  return (
    <View style={[styles.dot, { backgroundColor: COLORS[priority] }]} accessibilityLabel={priority}>
      <AppText variant="caption" style={styles.label}>
        {priority === "urgent" ? "URGENT" : "IMPORTANT"}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  dot: {
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
    alignSelf: "flex-start",
  },
  label: { color: "#fff", fontSize: FONT_SIZE.px10, fontWeight: "700" },
});
