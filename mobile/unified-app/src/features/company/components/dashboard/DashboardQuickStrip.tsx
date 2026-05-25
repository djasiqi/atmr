import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardQuickAction } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { M } from "./dashboardMobileTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

const ACTION_ACCENT: Partial<Record<DashboardQuickAction["key"], { bg: string; fg: string }>> = {
  create: { bg: D.brandSoft, fg: D.brandDark },
  assign: { bg: "rgba(59, 130, 246, 0.12)", fg: D.inProgress },
  urgent: { bg: D.dangerSoft, fg: D.danger },
  search: { bg: "rgba(100, 116, 139, 0.1)", fg: D.textSecondary },
};

type Props = {
  actions: DashboardQuickAction[];
  onPressAction: (key: DashboardQuickAction["key"]) => void;
};

export function DashboardQuickStrip({ actions, onPressAction }: Props) {
  return (
    <View style={s.row} accessibilityLabel="Actions rapides">
      {actions.map((action) => {
        const accent = ACTION_ACCENT[action.key] ?? { bg: D.brandSoft, fg: D.brandDark };
        return (
          <Pressable
            key={action.key}
            onPress={() => onPressAction(action.key)}
            style={({ pressed }) => [s.cell, pressed && s.pressed]}
            accessibilityRole="button"
            accessibilityLabel={action.label}
          >
            <View style={[s.iconWell, { backgroundColor: accent.bg }]}>
              <Ionicons name={action.icon} size={17} color={accent.fg} />
            </View>
            <AppText variant="caption" style={s.label} numberOfLines={1}>
              {action.label.split(" ")[0]}
            </AppText>
          </Pressable>
        );
      })}
    </View>
  );
}

const s = StyleSheet.create({
  row: {
    flexDirection: "row",
    paddingHorizontal: M.padH - 2,
    paddingBottom: 10,
    gap: 4,
  },
  cell: {
    flex: 1,
    alignItems: "center",
    gap: 4,
    paddingVertical: 2,
    minWidth: 0,
  },
  pressed: { opacity: 0.85, transform: [{ scale: 0.97 }] },
  iconWell: {
    width: M.iconMd,
    height: M.iconMd,
    borderRadius: 11,
    alignItems: "center",
    justifyContent: "center",
  },
  label: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    color: D.textSecondary,
    textAlign: "center",
  },
});
