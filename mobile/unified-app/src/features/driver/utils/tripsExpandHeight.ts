/** Garantit l’absence de plafond d’expansion trips à 520 px. */
export function tripsExpandMaxHeightFromContent(contentHeight: number): number {
  return Math.max(contentHeight, 1);
}
