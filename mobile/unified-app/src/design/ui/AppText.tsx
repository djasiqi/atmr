import type { ReactNode } from "react";
import { Text, type TextProps } from "react-native";
import { useAppViewport } from "../responsive/useAppViewport";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { type AppTextVariant, getAppTextStyle } from "./typography";

export type AppTextProps = TextProps & {
  variant: AppTextVariant;
  children?: ReactNode;
};

const MAX_FONT_MULTIPLIER = 1.35;

export function AppText({ variant, style, children, ...rest }: AppTextProps) {
  const tokens = useResponsiveTokens();
  const viewport = useAppViewport();
  const base = getAppTextStyle(variant, tokens, viewport);

  return (
    <Text maxFontSizeMultiplier={MAX_FONT_MULTIPLIER} style={[base, style]} {...rest}>
      {children}
    </Text>
  );
}
