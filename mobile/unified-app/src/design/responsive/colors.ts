/**
 * Palette sémantique et marque LIRIE — source unique pour imports directs.
 * `brand.ts` réexporte ce module pour compatibilité chemins historiques.
 */

/** Marque */
export const brandPrimary = "#00796B";
/**
 * Fond bulle chat « envoyé » (MessageBubble) / surfaces pleines alignées (ex. pilule barre onglets entreprise).
 * Équivalent Flowbite / Tailwind `teal-600`.
 */
export const chatOutgoingBubbleFill = "#00796B";
export const brandPrimaryDisabled = "#84B7AE";
export const brandSurfacePage = "#EAF3F1";
export const brandSurfaceSoft = "#F7FBFA";
export const brandText = "#163A34";
export const brandTextMuted = "#5F7369";

/** Surfaces additionnelles */
export const surfaceCard = "#FFFFFF";
export const surfaceModal = "#FFFCFC";
export const surfaceMuted = "#F2F4F7";

/** Bordures */
export const borderDefault = "rgba(145, 165, 157, 0.38)";
export const borderMuted = "rgba(228, 231, 236, 0.95)";
export const borderStrong = "rgba(145, 165, 157, 0.42)";

/** États sémantiques (alignés usages login / client home) */
export const semanticDanger = {
  fg: "#7A0012",
  fgStrong: "#B42318",
  bg: "#FFF1F1",
  border: "#D92D20",
  ctaBg: "#B00020",
  ctaFg: "#FFFFFF",
} as const;

export const semanticWarning = {
  fg: "#6A5320",
  bg: "#FFF7E6",
  border: "#E0B86C",
  ctaBg: "#00796B",
  ctaFg: "#FFFFFF",
} as const;

export const semanticSuccess = {
  fg: "#166534",
  bg: "#ecfdf5",
  border: "#bbf7d0",
} as const;

export const semanticInfo = {
  fg: "#1e40af",
  bg: "#eff6ff",
  border: "#bfdbfe",
} as const;

/** Pastille transport « en route » (cyan discret, aligné web client) */
export const semanticRoute = {
  fg: "#0e7490",
  bg: "#ecfeff",
  border: "#a5f3fc",
} as const;

/** Couleur erreur texte champs (hors palette danger carte) */
export const semanticErrorText = "#B91C1C";
