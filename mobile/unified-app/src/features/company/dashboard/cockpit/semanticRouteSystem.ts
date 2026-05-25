import type { AttentionLevel } from "./fleetAttentionSystem";
import type { MapDensityPolicy } from "./mapDensityGovernor";

export type SemanticRouteKind = "active" | "informative" | "historical" | "incident";

export type SemanticRouteStyle = {
  kind: SemanticRouteKind;
  opacity: number;
  strokeWidth: number;
  visible: boolean;
  zIndex: number;
};

const ROUTE_STYLES: Record<SemanticRouteKind, Omit<SemanticRouteStyle, "kind" | "visible">> = {
  active: { opacity: 0.95, strokeWidth: 4.5, zIndex: 40 },
  informative: { opacity: 0.45, strokeWidth: 2.5, zIndex: 20 },
  historical: { opacity: 0.12, strokeWidth: 1.5, zIndex: 5 },
  incident: { opacity: 0.88, strokeWidth: 4, zIndex: 35 },
};

export type SemanticRouteInput = {
  isSelected: boolean;
  isCritical: boolean;
  isSearchResult: boolean;
  recentlyInteracted: boolean;
  attentionLevel: AttentionLevel;
  density: MapDensityPolicy;
  maxVisibleRoutes: number;
  routeIndex: number;
};

export function classifySemanticRoute(input: SemanticRouteInput): SemanticRouteStyle {
  let kind: SemanticRouteKind = "informative";

  if (input.isCritical || input.attentionLevel === "CRITICAL") {
    kind = "incident";
  } else if (input.isSelected || input.isSearchResult || input.recentlyInteracted) {
    kind = "active";
  } else if (input.density.hideSecondaryRoutes) {
    kind = "historical";
  }

  const base = ROUTE_STYLES[kind];
  const visible =
    kind === "active" ||
    kind === "incident" ||
    (kind === "informative" &&
      !input.density.hideSecondaryRoutes &&
      input.routeIndex < input.maxVisibleRoutes);

  return {
    kind,
    ...base,
    visible,
  };
}

export function capVisibleRouteCount(
  density: MapDensityPolicy,
  globalMode: boolean
): number {
  if (globalMode) return 0;
  if (density.aggregateMode) return 1;
  if (density.level === "extreme" || density.level === "high") return 1;
  return 2;
}
