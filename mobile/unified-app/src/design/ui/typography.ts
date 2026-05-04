import type { TextStyle } from "react-native";
import { brandText, brandTextMuted } from "../responsive/brand";
import type { AppViewport } from "../responsive/useAppViewport";
import type { ResponsiveTokens } from "../responsive/useResponsiveTokens";

export type AppTextVariant =
  | "screenTitle"
  | "sectionTitle"
  | "body"
  | "bodyMuted"
  | "caption"
  | "label"
  | "error";

/** Couleur texte d’erreur (hors palette brand). */
export const appTextErrorColor = "#B91C1C";

type ViewportPick = Pick<AppViewport, "isTiny" | "isCompact" | "isTablet">;

/**
 * Styles typographiques purs pour `AppText`, dérivés des tokens et du viewport.
 */
export function getAppTextStyle(
  variant: AppTextVariant,
  tokens: ResponsiveTokens,
  viewport: ViewportPick
): TextStyle {
  const { isTiny, isCompact, isTablet } = viewport;

  const screenTitleSize = isTablet ? 26 : isTiny ? 20 : isCompact ? 21 : 22;
  const sectionTitleSize = isTablet ? 20 : isTiny ? 16 : isCompact ? 17 : 18;
  const bodySize = isTablet ? 17 : isTiny ? 14 : isCompact ? 15 : 16;
  const captionSize = isTablet ? 14 : isTiny ? 11 : isCompact ? 12 : 13;
  const labelSize = isTablet ? 14 : isTiny ? 12 : isCompact ? 12.5 : 13;

  switch (variant) {
    case "screenTitle":
      return {
        fontSize: screenTitleSize,
        lineHeight: Math.round(screenTitleSize * tokens.headingLineHeightRatio),
        fontWeight: "700",
        color: brandText,
      };
    case "sectionTitle":
      return {
        fontSize: sectionTitleSize,
        lineHeight: Math.round(sectionTitleSize * tokens.headingLineHeightRatio),
        fontWeight: "600",
        color: brandText,
      };
    case "body":
      return {
        fontSize: bodySize,
        lineHeight: Math.round(bodySize * tokens.bodyLineHeightRatio),
        fontWeight: "400",
        color: brandText,
      };
    case "bodyMuted":
      return {
        fontSize: bodySize,
        lineHeight: Math.round(bodySize * tokens.bodyLineHeightRatio),
        fontWeight: "400",
        color: brandTextMuted,
      };
    case "caption":
      return {
        fontSize: captionSize,
        lineHeight: Math.round(captionSize * tokens.bodyLineHeightRatio),
        fontWeight: "400",
        color: brandTextMuted,
      };
    case "label":
      return {
        fontSize: labelSize,
        lineHeight: Math.round(labelSize * tokens.bodyLineHeightRatio),
        fontWeight: "600",
        color: brandText,
      };
    case "error":
      return {
        fontSize: captionSize,
        lineHeight: Math.round(captionSize * tokens.bodyLineHeightRatio),
        fontWeight: "500",
        color: appTextErrorColor,
      };
    default: {
      const _exhaustive: never = variant;
      return _exhaustive;
    }
  }
}
