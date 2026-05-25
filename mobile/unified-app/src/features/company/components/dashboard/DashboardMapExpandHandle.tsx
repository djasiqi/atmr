import { Pressable, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import { FLEET_MAP_COLORS } from "../maps/mapStatusTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  expanded: boolean;
  onPress: () => void;
};

/** Poignée en bas de la carte pour passer en mode immersif / revenir au split. */
export function DashboardMapExpandHandle({ expanded, onPress }: Props) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [s.wrap, fleetGlassPanel(), pressed && s.pressed]}
      accessibilityRole="button"
      accessibilityLabel={expanded ? "Réduire la carte" : "Agrandir la carte en plein écran"}
    >
      <Ionicons
        name={expanded ? "chevron-down-outline" : "chevron-up-outline"}
        size={16}
        color={FLEET_MAP_COLORS.text}
      />
      <AppText variant="caption" style={s.label}>
        {expanded ? "Réduire" : "Plein écran"}
      </AppText>
    </Pressable>
  );
}

const s = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "center",
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    marginTop: -18,
    zIndex: 12,
  },
  pressed: { opacity: 0.9 },
  label: {
    color: FLEET_MAP_COLORS.text,
    fontWeight: "700",
    fontSize: FONT_SIZE.px12,
  },
});
