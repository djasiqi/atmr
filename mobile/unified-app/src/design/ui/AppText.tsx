import type { ReactNode } from "react";
import { Text, type TextProps } from "react-native";
import {
  fontCapForScaleRole,
  type AppTextScaleRole,
} from "../responsive/fontScaleCaps";
import { useAppViewport } from "../responsive/useAppViewport";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { type AppTextVariant, getAppTextStyle } from "./typography";

export type AppTextProps = TextProps & {
  variant: AppTextVariant;
  children?: ReactNode;
  /**
   * Rôle d’échelle : `content` (défaut, cap 2.0) ou `chrome` (cap 1.3).
   * Un `maxFontSizeMultiplier` explicite prime toujours (exception documentée).
   */
  scaleRole?: AppTextScaleRole;
};

export function AppText({
  variant,
  style,
  children,
  scaleRole = "content",
  maxFontSizeMultiplier,
  ...rest
}: AppTextProps) {
  const tokens = useResponsiveTokens();
  const viewport = useAppViewport();
  const base = getAppTextStyle(variant, tokens, viewport);
  const resolvedMultiplier =
    maxFontSizeMultiplier !== undefined
      ? maxFontSizeMultiplier
      : fontCapForScaleRole(scaleRole);

  return (
    <Text maxFontSizeMultiplier={resolvedMultiplier} style={[base, style]} {...rest}>
      {children}
    </Text>
  );
}
