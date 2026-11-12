import { Pressable, type PressableProps, StyleSheet, Text } from "react-native";
import { useReduceMotion } from "../navigation/useReduceMotion";
import { lirieColor, lirieControl, lirieRadius } from "./tokens";

export type LirieButtonVariant = "primary" | "secondary" | "quiet" | "danger";

type Props = Omit<PressableProps, "children"> & {
  label: string;
  variant?: LirieButtonVariant;
  busy?: boolean;
};

export function LirieButton({ label, variant = "primary", disabled, busy, ...rest }: Props) {
  const reduceMotion = useReduceMotion();
  const inactive = disabled === true || busy === true;
  const palette = inactive ? disabledPalette : PALETTE[variant];
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: inactive, busy: busy === true }}
      disabled={inactive}
      {...rest}
      style={({ pressed }) => [
        styles.base,
        {
          backgroundColor: palette.fill,
          borderColor: palette.border,
          opacity: pressed && !inactive ? 0.88 : 1,
          transform: pressed && !inactive && !reduceMotion ? [{ scale: 0.98 }] : [],
        },
      ]}
    >
      <Text
        maxFontSizeMultiplier={2}
        style={[styles.label, { color: palette.ink }]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

const PALETTE: Record<LirieButtonVariant, { fill: string; ink: string; border: string }> = {
  primary: { fill: lirieColor.green, ink: lirieColor.onBrand, border: lirieColor.green },
  secondary: { fill: lirieColor.white, ink: lirieColor.ink, border: lirieColor.border },
  quiet: { fill: "transparent", ink: lirieColor.green, border: "transparent" },
  danger: { fill: lirieColor.dangerFill, ink: lirieColor.danger, border: lirieColor.danger },
};

const disabledPalette = {
  fill: lirieColor.disabledFill,
  ink: lirieColor.disabledInk,
  border: lirieColor.disabledFill,
};

const styles = StyleSheet.create({
  base: {
    minHeight: lirieControl.minHeight,
    borderRadius: lirieRadius.control,
    borderWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  label: {
    fontSize: 16,
    lineHeight: 22,
    fontWeight: "700",
    textAlign: "center",
  },
});
