import type { CompanyDriverLiveLocation } from "../../api/contracts";
import type { FleetOperationalStatus } from "./mapStatusTheme";
import type { DriverLocationPresence } from "./driverLocationPresence";

/** Palette marqueurs carte — parité web (DriverLiveMap + mapUtils). */
export const FLEET_WEB_STATUS_COLORS = {
  available: "#4ade80",
  assigned: "#f59e0b",
  busy: "#00796B",
  offline: "#91A3A0",
  emergency: "#ef4444",
  constrained: "#f97316",
  brandDark: "#00695C",
  staleMuted: "#94A3B8",
} as const;

const CONSTRAINED_DRIVER_STATUSES = new Set(["assigned_constrained", "available_constrained"]);

/** Interpolation RGB linéaire entre deux couleurs hex (parité web DriverLiveMap). */
export function blendHexColors(hexA: string, hexB: string, amount: number): string {
  const parse = (hex: string) => {
    const h = hex.replace("#", "");
    return {
      r: Number.parseInt(h.slice(0, 2), 16),
      g: Number.parseInt(h.slice(2, 4), 16),
      b: Number.parseInt(h.slice(4, 6), 16),
    };
  };
  const a = parse(hexA);
  const b = parse(hexB);
  const t = Math.max(0, Math.min(1, amount));
  const mix = (x: number, y: number) => Math.round(x * (1 - t) + y * t);
  const r = mix(a.r, b.r);
  const g = mix(a.g, b.g);
  const bl = mix(a.b, b.b);
  return `#${[r, g, bl].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

export function isFleetDriverConstrained(driver: CompanyDriverLiveLocation): boolean {
  const presence = String(driver.presence_status ?? "").toLowerCase();
  if (presence === "degraded_constrained") return true;
  const tracking = String(driver.tracking_display_status ?? "").toLowerCase();
  if (tracking === "degraded_constrained") return true;
  const status = String(driver.status ?? "").toLowerCase();
  return CONSTRAINED_DRIVER_STATUSES.has(status);
}

/** Couleur fill marqueur pour un statut enrichi (enrichissements locaux mappés sur palette web). */
export function resolveFleetMarkerFillColor(status: FleetOperationalStatus): string {
  switch (status) {
    case "available":
      return FLEET_WEB_STATUS_COLORS.available;
    case "assigned":
    case "break":
      return FLEET_WEB_STATUS_COLORS.assigned;
    case "busy":
      return FLEET_WEB_STATUS_COLORS.busy;
    case "delayed":
    case "incident":
    case "emergency":
      return FLEET_WEB_STATUS_COLORS.emergency;
    case "constrained":
      return FLEET_WEB_STATUS_COLORS.constrained;
    case "last_known":
    case "offline":
      return FLEET_WEB_STATUS_COLORS.offline;
    default:
      return FLEET_WEB_STATUS_COLORS.available;
  }
}

/**
 * Visuel marqueur — machine d’état présence GPS canonique.
 * live 100 % métier | recent atténué | stale gris | last_known fantôme.
 */
export function resolveMarkerVisual(
  status: FleetOperationalStatus,
  locationPresence: DriverLocationPresence | boolean
): { fill: string; opacity: number } {
  const base = resolveFleetMarkerFillColor(status);
  const presence: DriverLocationPresence =
    typeof locationPresence === "boolean"
      ? locationPresence
        ? "stale"
        : "live"
      : locationPresence;

  if (presence === "offline_unknown") {
    return { fill: FLEET_WEB_STATUS_COLORS.staleMuted, opacity: 0 };
  }
  if (presence === "last_known") {
    return {
      fill: FLEET_WEB_STATUS_COLORS.staleMuted,
      opacity: 0.45,
    };
  }
  if (presence === "stale") {
    return {
      fill: blendHexColors(base, FLEET_WEB_STATUS_COLORS.staleMuted, 0.55),
      opacity: 0.88,
    };
  }
  if (presence === "recent") {
    return {
      fill: base,
      opacity: 0.72,
    };
  }
  // live — couleur métier pleine (contrainte garde sa couleur orange)
  return { fill: base, opacity: 1 };
}
