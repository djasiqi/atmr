import { E } from "../../company/theme/enterpriseOpsTheme";

/** Tokens visuels dashboard chauffeur — alignés maquette ops driver. */
export const D = {
  pageBg: "#F5F7F6",
  cardBg: E.CARD,
  cardBorder: "rgba(145, 165, 157, 0.38)",
  /** Contour carte mission active — très discret. */
  cardBorderSoft: "rgba(100, 116, 139, 0.06)",
  cardRadius: 18,
  controlRadius: 16,
  brand: E.BRAND,
  brandDark: "#006D5B",
  brandCta: "#0A6A61",
  text: E.TEXT,
  textSub: E.TEXT_SEC,
  textMuted: E.TEXT_MUTED,
  routeText: "#5C6B7A",
  routeLabel: E.TEXT_MUTED,
  available: "#22C55E",
  assignedBadgeBg: "rgba(34, 197, 94, 0.11)",
  assignedBadgeBorder: "rgba(34, 197, 94, 0.24)",
  assignedBadgeText: "#15803D",
  flag: "#3B82F6",
  metricDivider: "rgba(145, 165, 157, 0.35)",
  stepLine: "#E2E8F0",
} as const;

export const dashboardCardShadow = {
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.045,
  shadowRadius: 10,
  elevation: 2,
} as const;

/** Ombre très légère — carte mission active sans bordure visible. */
export const missionActiveCardShadow = {
  shadowColor: "#0F172A",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.035,
  shadowRadius: 14,
  elevation: 1,
} as const;

export const dashboardSoftShadow = {
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.035,
  shadowRadius: 6,
  elevation: 1,
} as const;
