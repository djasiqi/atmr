import type { Ionicons } from "@expo/vector-icons";



/** Statut opérationnel affiché sur la carte flotte (dispatch). */

export type FleetOperationalStatus =

  | "on_mission"

  | "available"

  | "break"

  | "delayed"

  | "incident"

  | "offline";



/** Variante visuelle alignée sur le guide « MAPS – propositions d’amélioration ». */

export type FleetMarkerVariant = "vehicle" | "vehicle_alert" | "incident_triangle";



/** Palette alignée sur le mockup (bleu mission, vert dispo, orange pause, rouge alerte). */

export const FLEET_MAP_COLORS = {

  brand: "#00796B",

  onMission: "#3B82F6",

  available: "#00796B",

  break: "#F59E0B",

  delayed: "#EF4444",

  incident: "#EF4444",

  offline: "#94A3B8",

  cluster: "#3498DB",

  route: "#3498DB",

  routeActive: "#00796B",

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

    pulse?: boolean;

  }

> = {

  incident: {

    fill: FLEET_MAP_COLORS.incident,

    label: "Incident",

    icon: "warning",

    markerVariant: "incident_triangle",

    priority: 100,

    pulse: true,

  },

  delayed: {

    fill: FLEET_MAP_COLORS.delayed,

    label: "En retard",

    icon: "car",

    markerVariant: "vehicle_alert",

    priority: 90,

    pulse: true,

  },

  on_mission: {

    fill: FLEET_MAP_COLORS.onMission,

    label: "En mission",

    icon: "car",

    markerVariant: "vehicle",

    priority: 55,

    pulse: true,

  },

  break: {

    fill: FLEET_MAP_COLORS.break,

    label: "En pause",

    icon: "car",

    markerVariant: "vehicle",

    priority: 40,

  },

  available: {

    fill: FLEET_MAP_COLORS.available,

    label: "Disponible",

    icon: "car",

    markerVariant: "vehicle",

    priority: 32,

    pulse: true,

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

  { status: "on_mission", label: "En mission", color: FLEET_MAP_COLORS.onMission, variant: "vehicle" },

  { status: "available", label: "Disponible", color: FLEET_MAP_COLORS.available, variant: "vehicle" },

  { status: "break", label: "Pause", color: FLEET_MAP_COLORS.break, variant: "vehicle" },

  { status: "delayed", label: "Retard", color: FLEET_MAP_COLORS.delayed, variant: "vehicle_alert" },

  { status: "incident", label: "Incident", color: FLEET_MAP_COLORS.incident, variant: "incident_triangle" },

  { status: "cluster", label: "Plusieurs chauffeurs", color: FLEET_MAP_COLORS.brand },

];


