import { Platform, type ViewStyle } from "react-native";
import { brandPrimary } from "../responsive/colors";

export type ShadowLevel = "sm" | "md" | "lg";

/**
 * Ombres homogènes cartes / surfaces — évite la duplication de Platform.select dans les écrans.
 */
export function getShadowStyle(level: ShadowLevel): ViewStyle {
  switch (level) {
    case "sm":
      return Platform.select({
        web: { boxShadow: "0 1px 6px rgba(15, 23, 42, 0.06)" },
        ios: {
          shadowColor: "#0f172a",
          shadowOffset: { width: 0, height: 1 },
          shadowOpacity: 0.06,
          shadowRadius: 6,
        },
        android: { elevation: 2 },
        default: {},
      }) as ViewStyle;
    case "md":
      return Platform.select({
        web: { boxShadow: "0 4px 14px rgba(22, 58, 52, 0.12)" },
        ios: {
          shadowColor: "#163A34",
          shadowOpacity: 0.12,
          shadowRadius: 14,
          shadowOffset: { width: 0, height: 6 },
        },
        android: { elevation: 4 },
        default: {},
      }) as ViewStyle;
    case "lg":
      return Platform.select({
        web: { boxShadow: "0 12px 40px rgba(22, 58, 52, 0.14)" },
        ios: {
          shadowColor: "#163A34",
          shadowOpacity: 0.14,
          shadowRadius: 22,
          shadowOffset: { width: 0, height: 10 },
        },
        android: { elevation: 6 },
        default: {},
      }) as ViewStyle;
    default: {
      const _exhaustive: never = level;
      return _exhaustive;
    }
  }
}

/** Ombre bouton primaire (léger glow marque). */
export function getPrimaryButtonShadowStyle(): ViewStyle {
  return Platform.select({
    web: { boxShadow: "0 2px 14px rgba(0, 121, 107, 0.28)" },
    ios: {
      shadowColor: brandPrimary,
      shadowOpacity: 0.28,
      shadowRadius: 8,
      shadowOffset: { width: 0, height: 3 },
    },
    android: { elevation: 3 },
    default: {},
  }) as ViewStyle;
}
