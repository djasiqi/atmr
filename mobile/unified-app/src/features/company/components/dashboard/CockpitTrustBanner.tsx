import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  message: string;
};

/** Signal discret de confiance opérationnelle (mode dégradé, surcharge, etc.). */
export function CockpitTrustBanner({ message }: Props) {
  return (
    <View style={[s.wrap, fleetGlassPanel()]} accessibilityRole="text">
      <Ionicons name="information-circle-outline" size={14} color="#64748B" />
      <AppText variant="caption" style={s.text} numberOfLines={1}>
        {message}
      </AppText>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    alignSelf: "center",
    maxWidth: "92%",
  },
  text: {
    color: "#475569",
    fontWeight: "600",
    fontSize: FONT_SIZE.px11,
    flexShrink: 1,
  },
});
