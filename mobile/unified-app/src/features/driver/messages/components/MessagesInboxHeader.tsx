import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
type Props = {
  urgentFilterActive: boolean;
  searchOpen: boolean;
  onToggleFilter: () => void;
  onToggleSearch: () => void;
  onOpenMenu: () => void;
};

/** En-tête aligné maquette : titre + filtre / recherche / menu (icônes uniquement). */
export function MessagesInboxHeader({
  urgentFilterActive,
  searchOpen,
  onToggleFilter,
  onToggleSearch,
  onOpenMenu,
}: Props) {
  return (
    <View style={styles.wrap}>
      <AppText variant="sectionTitle" style={styles.title}>
        Messages
      </AppText>
      <View style={styles.actions}>
        <Pressable
          style={[styles.iconBtn, urgentFilterActive && styles.iconBtnActive]}
          onPress={onToggleFilter}
          accessibilityLabel="Filtrer"
        >
          <Ionicons
            name="funnel-outline"
            size={22}
            color={urgentFilterActive ? M.BRAND : M.TEXT}
          />
        </Pressable>
        <Pressable
          style={[styles.iconBtn, searchOpen && styles.iconBtnActive]}
          onPress={onToggleSearch}
          accessibilityLabel="Rechercher"
        >
          <Ionicons name="search-outline" size={22} color={M.TEXT} />
        </Pressable>
        <Pressable style={styles.iconBtn} onPress={onOpenMenu} accessibilityLabel="Plus d'options">
          <Ionicons name="ellipsis-vertical" size={22} color={M.TEXT} />
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingBottom: 4,
  },
  title: {
    fontSize: FONT_SIZE.px28,
    lineHeight: Math.round(FONT_SIZE.px28 * 1.2),
    fontWeight: "700",
    color: M.TEXT,
  },
  actions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 2,
  },
  iconBtn: {
    padding: 8,
    borderRadius: 20,
  },
  iconBtnActive: {
    backgroundColor: M.BRAND_SOFT,
  },
});