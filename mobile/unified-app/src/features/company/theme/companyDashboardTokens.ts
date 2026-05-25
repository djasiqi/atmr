import { Platform, type ViewStyle } from "react-native";
import { createShadow } from "../../../styles/shadowStyles";
import { E } from "./enterpriseOpsTheme";

/** Tokens premium dashboard entreprise (mobile cockpit). */
export const D = {
  pageBg: "#F4F7F9",
  cardBg: "#FFFFFF",
  text: "#0F172A",
  textSecondary: "#64748B",
  textMuted: "#94A3B8",
  border: "rgba(148, 163, 184, 0.28)",
  brand: "#00796B",
  brandDark: "#00796B",
  brandSoft: "rgba(0, 121, 107, 0.12)",
  inProgress: "#3B82F6",
  upcoming: "#F59E0B",
  danger: "#EF4444",
  dangerSoft: "rgba(239, 68, 68, 0.08)",
  dangerBorder: "rgba(239, 68, 68, 0.35)",
  glassBg: "rgba(255, 255, 255, 0.82)",
  glassBorder: "rgba(255, 255, 255, 0.55)",
  mapHeroMaxHeight: 252,
  radiusCard: 24,
  radiusPill: 999,
  radiusChip: 14,
} as const;

export const dashboardCardShadow = createShadow({
  shadowColor: "#0F172A",
  shadowOffset: { width: 0, height: 4 },
  shadowOpacity: 0.06,
  shadowRadius: 14,
  elevation: 3,
});

export const dashboardCardShadowWeb: ViewStyle =
  Platform.OS === "web"
    ? ({ boxShadow: "0 4px 18px rgba(15, 23, 42, 0.07)" } as ViewStyle)
    : dashboardCardShadow;

export const dashboardSurface = {
  backgroundColor: D.cardBg,
  borderRadius: D.radiusCard,
  borderWidth: 1,
  borderColor: D.border,
  ...dashboardCardShadowWeb,
} as const;

/** Compat legacy — réexport palette entreprise existante. */
export const dashboardLegacy = E;
