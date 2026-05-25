import { Platform, type ViewStyle } from "react-native";

/** Android natif / WebView : pas de backdrop-filter ; fond opaque pour éviter la carte en transparence. */
export function usesSolidFleetGlass(): boolean {
  if (Platform.OS === "android") return true;
  if (Platform.OS === "web" && typeof navigator !== "undefined") {
    return /Android/i.test(navigator.userAgent);
  }
  return false;
}

const FLEET_GLASS_SOLID: Pick<ViewStyle, "backgroundColor" | "borderColor"> = {
  backgroundColor: "#FFFFFF",
  borderColor: "rgba(226, 232, 240, 0.95)",
};

/** Panneaux flottants carte (overlay, légende, FAB). */
export const FLEET_UI = {
  fabSize: 44,
  fabGap: 8,
  /** Position verticale de la colonne FAB (carte dashboard). */
  fabStackTop: 32,
  fabRight: 12,
  overlayMaxWidthRatio: 0.78,
  overlayBottom: 12,
  overlayLeft: 12,
  overlayRadius: 22,
  overlayPaddingH: 10,
  overlayPaddingV: 7,
  legendTop: 10,
  legendLeft: 10,
  glassBg: "rgba(255, 255, 255, 0.78)",
  glassBorder: "rgba(255, 255, 255, 0.62)",
  shadow: Platform.select({
    ios: {
      shadowColor: "#0f172a",
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.08,
      shadowRadius: 10,
    },
    android: { elevation: 4 },
    web: { boxShadow: "0 6px 20px rgba(15, 23, 42, 0.1)" } as ViewStyle,
    default: {},
  }),
  blur: Platform.OS === "web" ? ({ backdropFilter: "blur(14px)" } as ViewStyle) : {},
  /** Garde-fous animations cockpit (Reanimated). */
  animMaxScale: 1.06,
  animPulseMaxOpacity: 0.9,
  animMaxConcurrent: 2,
  /** Tempo carte mission-first (aligné fleetMapMissionPolicies). */
  mapCameraMs: 620,
  mapRouteEnterMs: 380,
  mapRouteExitMs: 420,
  mapPriorityDecayMs: 520,
} as const;

export const fleetGlassPanel = (extra?: ViewStyle): ViewStyle => ({
  ...(usesSolidFleetGlass()
    ? FLEET_GLASS_SOLID
    : {
        backgroundColor: FLEET_UI.glassBg,
        borderColor: FLEET_UI.glassBorder,
        ...FLEET_UI.blur,
      }),
  borderRadius: FLEET_UI.overlayRadius,
  borderWidth: 1,
  ...FLEET_UI.shadow,
  ...(Platform.OS === "android" ? { overflow: "hidden" as const } : {}),
  ...extra,
});
