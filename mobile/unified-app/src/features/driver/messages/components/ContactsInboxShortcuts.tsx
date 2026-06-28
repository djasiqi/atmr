import { Pressable, ScrollView, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  onOpenTeam: () => void;
  onOpenDispatch: () => void;
  onNewColleague: () => void;
  onOpenSupport: () => void;
};

type ShortcutItem = {
  key: string;
  label: string;
  shortLabel: string;
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  primary?: boolean;
};

/** Raccourcis canaux — une ligne horizontale, icônes + libellé court. */
export function ContactsInboxShortcuts({
  onOpenTeam,
  onOpenDispatch,
  onNewColleague,
  onOpenSupport,
}: Props) {
  const items: ShortcutItem[] = [
    {
      key: "team",
      label: "Ouvrir le canal équipe",
      shortLabel: "Équipe",
      icon: "people-outline",
      onPress: onOpenTeam,
      primary: true,
    },
    {
      key: "dispatch",
      label: "Ouvrir le dispatch",
      shortLabel: "Dispatch",
      icon: "business-outline",
      onPress: onOpenDispatch,
    },
    {
      key: "contact",
      label: "Nouveau contact",
      shortLabel: "Contact",
      icon: "person-add-outline",
      onPress: onNewColleague,
    },
    {
      key: "support",
      label: "Support LIRIE",
      shortLabel: "Support",
      icon: "headset-outline",
      onPress: onOpenSupport,
    },
  ];

  return (
    <View style={styles.wrap}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.row}
        keyboardShouldPersistTaps="handled"
      >
        {items.map((item) => (
          <Pressable
            key={item.key}
            onPress={item.onPress}
            accessibilityRole="button"
            accessibilityLabel={item.label}
            style={({ pressed }) => [styles.action, pressed && styles.actionPressed]}
          >
            <View
              style={[
                styles.iconCircle,
                item.primary ? styles.iconCirclePrimary : null,
              ]}
            >
              <Ionicons
                name={item.icon}
                size={22}
                color={item.primary ? "#FFFFFF" : M.BRAND}
              />
            </View>
            <AppText variant="caption" style={styles.label} numberOfLines={1}>
              {item.shortLabel}
            </AppText>
          </Pressable>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    backgroundColor: M.CARD,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: M.BORDER,
    paddingVertical: 10,
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 4,
    paddingHorizontal: 12,
  },
  action: {
    width: 76,
    alignItems: "center",
    gap: 6,
    paddingVertical: 2,
  },
  actionPressed: {
    opacity: 0.88,
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: M.BRAND_SOFT,
    borderWidth: 1,
    borderColor: "rgba(10, 143, 122, 0.18)",
  },
  iconCirclePrimary: {
    backgroundColor: M.BRAND,
    borderColor: M.BRAND_DARK,
  },
  label: {
    color: M.TEXT_SEC,
    fontSize: FONT_SIZE.px11,
    fontWeight: "600",
    textAlign: "center",
    maxWidth: 72,
  },
});
