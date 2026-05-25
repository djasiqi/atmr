import { Pressable, ScrollView, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardQuickAction } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  actions: DashboardQuickAction[];
  onPressAction: (key: DashboardQuickAction["key"]) => void;
  activeKeys?: DashboardQuickAction["key"][];
  compact?: boolean;
};

const ACTION_ICONS: Record<DashboardQuickAction["key"], { icon: keyof typeof Ionicons.glyphMap; tint: string }> = {
  create: { icon: "add", tint: "#00796B" },
  assign: { icon: "person-add-outline", tint: "#3B82F6" },
  urgent: { icon: "warning-outline", tint: "#EF4444" },
  search: { icon: "search-outline", tint: "#64748B" },
};

/** Toolbar flottante compacte — outils dispatch. */
export function QuickActionsBar({ actions, onPressAction, activeKeys = [], compact = true }: Props) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={[s.row, compact && s.rowCompact]}
      accessibilityLabel="Actions rapides"
    >
      {actions.map((action) => {
        const meta = ACTION_ICONS[action.key];
        const active = activeKeys.includes(action.key);
        return (
          <Pressable
            key={action.key}
            onPress={() => onPressAction(action.key)}
            style={({ pressed }) => [
              s.tool,
              active && s.toolActive,
              pressed && s.pressed,
            ]}
            accessibilityRole="button"
            accessibilityLabel={action.label}
            accessibilityState={{ selected: active }}
          >
            <View style={[s.iconWell, { backgroundColor: `${meta.tint}18` }]}>
              <Ionicons name={meta.icon} size={16} color={meta.tint} />
            </View>
            <AppText variant="caption" style={s.label} numberOfLines={1}>
              {action.label}
            </AppText>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const s = StyleSheet.create({
  row: { gap: 8, paddingVertical: 2, alignItems: "center" },
  rowCompact: { gap: 6 },
  tool: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.72)",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.28)",
    maxHeight: 40,
  },
  toolActive: {
    borderColor: D.brand,
    backgroundColor: "rgba(0, 121, 107, 0.12)",
  },
  pressed: { opacity: 0.88 },
  iconWell: {
    width: 26,
    height: 26,
    borderRadius: 13,
    alignItems: "center",
    justifyContent: "center",
  },
  label: {
    color: D.text,
    fontWeight: "700",
    fontSize: FONT_SIZE.px11,
    maxWidth: 72,
  },
});
