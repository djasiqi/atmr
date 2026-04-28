/**
 * Ombres cross-plateforme (même principe que operations-app).
 */
import { Platform, ViewStyle } from "react-native";

export interface ShadowConfig {
  shadowColor: string;
  shadowOffset: { width: number; height: number };
  shadowOpacity: number;
  shadowRadius: number;
  elevation?: number;
}

export function createShadow(config: ShadowConfig): ViewStyle {
  const { shadowColor, shadowOffset, shadowOpacity, shadowRadius, elevation } = config;

  if (Platform.OS === "web") {
    const { width, height } = shadowOffset;
    const alpha = Math.round(shadowOpacity * 255);
    const colorWithAlpha = shadowColor.startsWith("#")
      ? `${shadowColor}${alpha.toString(16).padStart(2, "0")}`
      : shadowColor;
    return { boxShadow: `${width}px ${height}px ${shadowRadius}px ${colorWithAlpha}` } as ViewStyle;
  }

  const style: ViewStyle = {
    shadowColor,
    shadowOffset,
    shadowOpacity,
    shadowRadius,
  };
  if (Platform.OS === "android" && elevation !== undefined) {
    style.elevation = elevation;
  }
  return style;
}
