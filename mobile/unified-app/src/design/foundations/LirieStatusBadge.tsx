import { StyleSheet, Text, View } from "react-native";
import { lirieColor, lirieRadius, lirieStatusLabel, type LirieStatus } from "./tokens";

type Props = {
  status: LirieStatus;
};

/** Le libellé métier reste entier. La pastille grandit avec la police. */
export function LirieStatusBadge({ status }: Props) {
  return <LirieLabelBadge label={lirieStatusLabel[status]} />;
}

type BadgeTone = "neutral" | "warning" | "danger";

const TONE: Record<BadgeTone, { fill: string; ink: string; border: string }> = {
  neutral: { fill: lirieColor.statusFill, ink: lirieColor.ink, border: lirieColor.border },
  warning: { fill: lirieColor.warningFill, ink: lirieColor.warningInk, border: lirieColor.warningFill },
  danger: { fill: lirieColor.dangerFill, ink: lirieColor.danger, border: lirieColor.dangerFill },
};

/** Pastille pour un libellé métier libre, sans troncature. Le vert de marque n'est pas un statut. */
export function LirieLabelBadge({
  label,
  tone = "neutral",
  live = false,
}: {
  label: string;
  tone?: BadgeTone;
  live?: boolean;
}) {
  const palette = TONE[tone];
  return (
    <View
      accessibilityRole="text"
      accessibilityLabel={label}
      style={[styles.badge, { backgroundColor: palette.fill, borderColor: palette.border }]}
    >
      <Text
        maxFontSizeMultiplier={2}
        accessibilityLiveRegion={live ? "polite" : undefined}
        style={[styles.label, { color: palette.ink }]}
      >
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignSelf: "flex-start",
    borderWidth: 1,
    borderRadius: lirieRadius.badge,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  label: {
    fontSize: 14,
    lineHeight: 18,
    fontWeight: "700",
  },
});
