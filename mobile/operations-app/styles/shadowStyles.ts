/**
 * Utility for cross-platform shadow styles
 * 
 * React Native Web now recommends using boxShadow instead of shadow* props.
 * This utility provides a cross-platform solution that:
 * - Uses boxShadow on web (modern approach)
 * - Uses shadow* props on iOS/Android (required for native)
 */

import { Platform, ViewStyle } from "react-native";

export interface ShadowConfig {
  shadowColor: string;
  shadowOffset: { width: number; height: number };
  shadowOpacity: number;
  shadowRadius: number;
  elevation?: number; // For Android
}

/**
 * Creates cross-platform shadow styles
 * 
 * @param config Shadow configuration
 * @returns Platform-appropriate shadow styles
 * 
 * @example
 * const styles = StyleSheet.create({
 *   card: {
 *     ...createShadow({
 *       shadowColor: "#000",
 *       shadowOffset: { width: 0, height: 2 },
 *       shadowOpacity: 0.1,
 *       shadowRadius: 8,
 *       elevation: 3,
 *     }),
 *   },
 * });
 */
export function createShadow(config: ShadowConfig): ViewStyle {
  const { shadowColor, shadowOffset, shadowOpacity, shadowRadius, elevation } = config;

  if (Platform.OS === "web") {
    // Web: Use boxShadow (recommended by React Native Web)
    const { width, height } = shadowOffset;
    const alpha = Math.round(shadowOpacity * 255);
    const colorWithAlpha = shadowColor.startsWith("#")
      ? `${shadowColor}${alpha.toString(16).padStart(2, "0")}`
      : shadowColor;

    return {
      boxShadow: `${width}px ${height}px ${shadowRadius}px ${colorWithAlpha}`,
    } as ViewStyle;
  }

  // iOS/Android: Use shadow* props
  const style: ViewStyle = {
    shadowColor,
    shadowOffset,
    shadowOpacity,
    shadowRadius,
  };

  // Android: Add elevation for shadow support
  if (Platform.OS === "android" && elevation !== undefined) {
    style.elevation = elevation;
  }

  return style;
}

/**
 * Predefined shadow presets for common use cases
 */
export const shadowPresets = {
  /** Small shadow for subtle depth */
  small: createShadow({
    shadowColor: "rgba(15,54,43,0.06)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius: 4,
    elevation: 2,
  }),

  /** Medium shadow for cards */
  medium: createShadow({
    shadowColor: "rgba(15,54,43,0.08)",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 12,
    elevation: 4,
  }),

  /** Large shadow for modals and elevated content */
  large: createShadow({
    shadowColor: "rgba(15,54,43,0.15)",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 1,
    shadowRadius: 24,
    elevation: 8,
  }),

  /** Accent shadow with primary color */
  accent: createShadow({
    shadowColor: "rgba(10,127,89,0.3)",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 12,
    elevation: 4,
  }),
};
