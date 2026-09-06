import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import type { CockpitConnectivityCopy } from "./resolveCockpitBanner";

type Props = {
  copy: CockpitConnectivityCopy;
};

/** Ligne 3 du chrome cockpit — sous les stats, jamais en superposition. */
export function CockpitConnectivityBanner({ copy }: Props) {
  return (
    <View
      style={[s.wrap, fleetGlassPanel()]}
      accessibilityRole="text"
      accessibilityLabel={`${copy.title}. ${copy.body}`}
    >
      <AppText variant="label" style={s.title}>
        {copy.title}
      </AppText>
      <AppText variant="caption" style={s.body}>
        {copy.body}
      </AppText>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    alignSelf: "stretch",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 14,
    gap: 2,
  },
  title: {
    color: "#64748B",
    fontWeight: "700",
    fontSize: FONT_SIZE.px11,
    textAlign: "center",
  },
  body: {
    color: "#64748B",
    fontWeight: "500",
    fontSize: FONT_SIZE.px11,
    lineHeight: 15,
    textAlign: "center",
  },
});
