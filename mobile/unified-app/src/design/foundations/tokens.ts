/**
 * Jetons D2 — Signature LIRIE.
 * Distincts de `colors.ts` et `lirieShape.ts`, que les écrans métier
 * continuent d'utiliser jusqu'à la migration D3.
 * La marque n'est pas une couleur de statut : le vert porte l'action,
 * les statuts ont leur propre encre.
 */

export const lirieBrand = {
  green: "#00796B",
  white: "#FFFFFF",
  canvas: "#F5F7F6",
  ink: "#263532",
} as const;

export const lirieColor = {
  ...lirieBrand,
  inkMuted: "#3E5650",
  border: "#D5E0DC",
  onBrand: "#FFFFFF",
  brandPressed: "#00685C",
  disabledFill: "#E4EEEC",
  disabledInk: "#3A524C",
  danger: "#8C1D18",
  dangerFill: "#FDECEA",
  warningInk: "#6A4B12",
  warningFill: "#FFF6E8",
  statusFill: "#E7F4F1",
} as const;

export const lirieSpace = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
} as const;

export const lirieRadius = {
  control: 12,
  card: 12,
  badge: 999,
} as const;

export const lirieControl = {
  minHeight: 48,
  operationalMinHeight: 44,
} as const;

/** Apparition de feuille uniquement. Les onglets ne s'animent pas. */
export const lirieMotion = {
  sheetMs: 180,
} as const;

export type LirieReading = "comfort" | "operational";

export type LirieStatus =
  | "draft"
  | "assigned"
  | "cancelled"
  | "completed"
  | "unassigned";

export const lirieStatusLabel: Record<LirieStatus, string> = {
  draft: "Brouillon",
  assigned: "Assignée",
  cancelled: "Annulée",
  completed: "Terminée",
  unassigned: "Non assigné",
};

function channel(value: number): number {
  const s = value / 255;
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
}

export function contrastRatio(foreground: string, background: string): number {
  const luminance = (hex: string) => {
    const raw = hex.replace("#", "");
    const r = channel(parseInt(raw.slice(0, 2), 16));
    const g = channel(parseInt(raw.slice(2, 4), 16));
    const b = channel(parseInt(raw.slice(4, 6), 16));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const a = luminance(foreground);
  const b = luminance(background);
  const lighter = Math.max(a, b);
  const darker = Math.min(a, b);
  return (lighter + 0.05) / (darker + 0.05);
}
