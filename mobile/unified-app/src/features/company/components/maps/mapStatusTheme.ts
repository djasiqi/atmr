import type { Ionicons } from "@expo/vector-icons";

import { FLEET_WEB_STATUS_COLORS } from "./fleetMapStatusContract";

/** Statut opérationnel affiché sur la carte flotte (dispatch). */
export type FleetOperationalStatus =
  | "busy"
  | "assigned"
  | "available"
  | "break"
  | "constrained"
  | "last_known"
  | "delayed"
  | "incident"
  | "emergency"
  | "offline";

/** Variante visuelle (legacy — carte utilise cercle + initiales). */
export type FleetMarkerVariant = "vehicle" | "vehicle_alert" | "incident_triangle";

/** Palette carte flotte — alignée web via fleetMapStatusContract. */
export const FLEET_MAP_COLORS = {
  brand: FLEET_WEB_STATUS_COLORS.busy,
  busy: FLEET_WEB_STATUS_COLORS.busy,
  assigned: FLEET_WEB_STATUS_COLORS.assigned,
  available: FLEET_WEB_STATUS_COLORS.available,
  break: FLEET_WEB_STATUS_COLORS.assigned,
  delayed: FLEET_WEB_STATUS_COLORS.emergency,
  incident: FLEET_WEB_STATUS_COLORS.emergency,
  emergency: FLEET_WEB_STATUS_COLORS.emergency,
  constrained: FLEET_WEB_STATUS_COLORS.constrained,
  offline: FLEET_WEB_STATUS_COLORS.offline,
  /** Fallback légende uniquement — rendu cluster = couleur statut dominant. */
  cluster: FLEET_WEB_STATUS_COLORS.brandDark,
  route: "#3498DB",
  routeActive: FLEET_WEB_STATUS_COLORS.busy,
  routeMuted: "rgba(0, 121, 107, 0.55)",
  text: "#0F172A",
  textMuted: "#64748B",
  fabBg: "#FFFFFF",
  fabBorder: "rgba(148, 163, 184, 0.35)",
  sheetRadius: 28,
  fabRadius: 20,
} as const;

export const FLEET_STATUS_THEME: Record<
  FleetOperationalStatus,
  {
    fill: string;
    label: string;
    icon: keyof typeof Ionicons.glyphMap;
    markerVariant: FleetMarkerVariant;
    priority: number;
  }
> = {
  emergency: {
    fill: FLEET_MAP_COLORS.emergency,
    label: "Urgence",
    icon: "warning",
    markerVariant: "incident_triangle",
    priority: 105,
  },
  incident: {
    fill: FLEET_MAP_COLORS.incident,
    label: "Incident",
    icon: "warning",
    markerVariant: "incident_triangle",
    priority: 100,
  },
  delayed: {
    fill: FLEET_MAP_COLORS.delayed,
    label: "En retard",
    icon: "car",
    markerVariant: "vehicle_alert",
    priority: 90,
  },
  busy: {
    fill: FLEET_MAP_COLORS.busy,
    label: "En mission",
    icon: "car",
    markerVariant: "vehicle",
    priority: 55,
  },
  assigned: {
    fill: FLEET_MAP_COLORS.assigned,
    label: "Assigné",
    icon: "car",
    markerVariant: "vehicle",
    priority: 50,
  },
  break: {
    fill: FLEET_MAP_COLORS.break,
    label: "En pause",
    icon: "car",
    markerVariant: "vehicle",
    priority: 40,
  },
  constrained: {
    fill: FLEET_MAP_COLORS.constrained,
    label: "Batterie restreinte",
    icon: "battery-half",
    markerVariant: "vehicle",
    priority: 48,
  },
  last_known: {
    fill: FLEET_MAP_COLORS.offline,
    label: "Dernière position",
    icon: "location",
    markerVariant: "vehicle",
    priority: 12,
  },
  available: {
    fill: FLEET_MAP_COLORS.available,
    label: "Disponible",
    icon: "car",
    markerVariant: "vehicle",
    priority: 32,
  },
  offline: {
    fill: FLEET_MAP_COLORS.offline,
    label: "Hors ligne",
    icon: "car",
    markerVariant: "vehicle",
    priority: 10,
  },
};

export type FleetMapLegendItem = {
  status: FleetOperationalStatus | "cluster";
  label: string;
  color: string;
  variant?: FleetMarkerVariant;
};

export const FLEET_MAP_LEGEND_ITEMS: FleetMapLegendItem[] = [
  { status: "busy", label: "En mission", color: FLEET_MAP_COLORS.busy, variant: "vehicle" },
  { status: "assigned", label: "Assigné", color: FLEET_MAP_COLORS.assigned, variant: "vehicle" },
  { status: "available", label: "Disponible", color: FLEET_MAP_COLORS.available, variant: "vehicle" },
  { status: "constrained", label: "Batterie restreinte", color: FLEET_MAP_COLORS.constrained, variant: "vehicle" },
  { status: "break", label: "Pause", color: FLEET_MAP_COLORS.break, variant: "vehicle" },
  { status: "delayed", label: "Retard", color: FLEET_MAP_COLORS.delayed, variant: "vehicle_alert" },
  { status: "incident", label: "Incident", color: FLEET_MAP_COLORS.incident, variant: "incident_triangle" },
  { status: "emergency", label: "Urgence", color: FLEET_MAP_COLORS.emergency, variant: "incident_triangle" },
  { status: "cluster", label: "Plusieurs chauffeurs", color: FLEET_MAP_COLORS.cluster },
];
