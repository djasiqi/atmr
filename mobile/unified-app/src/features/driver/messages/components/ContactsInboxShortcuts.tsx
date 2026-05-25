import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
type Props = {
  onNewColleague: () => void;
  onOpenSupport: () => void;
};

/** Raccourcis visibles dans l’onglet CONTACTS (maquette + actions métier). */
export function ContactsInboxShortcuts({ onNewColleague, onOpenSupport }: Props) {
  return (
    <View style={styles.wrap}>
      <Pressable style={styles.chip} onPress={onNewColleague} accessibilityRole="button">
        <Ionicons name="person-add-outline" size={18} color={M.BRAND} />
        <AppText variant="caption" style={styles.chipText}>
          Nouveau contact
        </AppText>
      </Pressable>
      <Pressable style={styles.chip} onPress={onOpenSupport} accessibilityRole="button">
        <Ionicons name="headset-outline" size={18} color={M.BRAND} />
        <AppText variant="caption" style={styles.chipText}>
          Support LIRIE
        </AppText>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: M.CARD,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: M.BORDER,
  },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: M.BRAND_SOFT,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#99F6E4",
  },
  chipText: {
    color: M.BRAND,    fontWeight: "600",
    fontSize: FONT_SIZE.px13,
  },
});
