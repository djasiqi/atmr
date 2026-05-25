import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { FLEET_MAP_COLORS } from "./mapStatusTheme";
import { fleetGlassPanel } from "./fleetMapUiTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  filteredOut?: boolean;
};

export function MapEmptyState({ filteredOut }: Props) {
  return (
    <View style={s.wrap} pointerEvents="none" accessibilityLabel="Aucun chauffeur visible">
      <View style={[fleetGlassPanel(), s.card]}>
        <Ionicons name="car-outline" size={20} color={FLEET_MAP_COLORS.textMuted} />
        <AppText variant="caption" style={s.text}>
          {filteredOut
            ? "Aucun chauffeur ne correspond aux filtres"
            : "Aucune position chauffeur pour le moment"}
        </AppText>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
    zIndex: 11,
  },
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 18,
    maxWidth: "85%",
  },
  text: {
    color: FLEET_MAP_COLORS.textMuted,
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
    flexShrink: 1,
  },
});
