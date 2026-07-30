import type { ReactNode } from "react";
import {
  Platform,
  StyleSheet,
  Text,
  type StyleProp,
  type TextProps,
  type TextStyle,
} from "react-native";
import {
  clampScale,
  fontCapForScaleRole,
  type AppTextScaleRole,
} from "../responsive/fontScaleCaps";
import { useAccessibilityScale } from "../responsive/useAccessibilityScale";
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

/**
 * Sur Android, `fontSize` suit `fontScale` mais `lineHeight` fixe ne suit pas —
 * d’où le clipping vertical avec une grande police système.
 * On multiplie donc le lineHeight par le même facteur effectif que le texte.
 */
export function scaleLineHeightForFontScale(
  style: StyleProp<TextStyle>,
  fontScale: number,
  maxFontSizeMultiplier: number
): TextStyle | null {
  if (Platform.OS !== "android") return null;
  const flat = StyleSheet.flatten(style);
  const lineHeight = flat?.lineHeight;
  if (typeof lineHeight !== "number" || !Number.isFinite(lineHeight) || lineHeight <= 0) {
    return null;
  }
  const scale = clampScale(fontScale, maxFontSizeMultiplier);
  if (scale <= 1) return null;
  return { lineHeight: Math.round(lineHeight * scale) };
}

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
  const { fontScale } = useAccessibilityScale();
  const base = getAppTextStyle(variant, tokens, viewport);
  const resolvedMultiplier =
    maxFontSizeMultiplier !== undefined
      ? maxFontSizeMultiplier
      : fontCapForScaleRole(scaleRole);
  const combinedStyle: StyleProp<TextStyle> = [base, style];
  const androidLineHeight = scaleLineHeightForFontScale(
    combinedStyle,
    fontScale,
    resolvedMultiplier
  );

  return (
    <Text
      maxFontSizeMultiplier={resolvedMultiplier}
      style={[combinedStyle, androidLineHeight]}
      {...rest}
    >
      {children}
    </Text>
  );
}
