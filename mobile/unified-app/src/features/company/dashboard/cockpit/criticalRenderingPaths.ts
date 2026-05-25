/** Chemins de rendu critiques — priorité sous charge élevée. */
export const CRITICAL_RENDERING_PRIORITY = [
  "map_gestures",
  "selected_driver",
  "active_route",
  "critical_alerts",
  "secondary_overlays",
] as const;

export type CriticalRenderingPath = (typeof CRITICAL_RENDERING_PRIORITY)[number];

export function isCriticalRenderingPath(path: CriticalRenderingPath): boolean {
  const idx = CRITICAL_RENDERING_PRIORITY.indexOf(path);
  return idx >= 0 && idx <= 3;
}

export function shouldDeferRendering(
  path: CriticalRenderingPath,
  underLoad: boolean
): boolean {
  if (!underLoad) return false;
  return !isCriticalRenderingPath(path);
}
