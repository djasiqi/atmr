import { computeCompanyFloatingBottomPad } from "../../../../design/navigation/BaseFloatingBar";

/** Métriques cockpit — carte + zone opérations. */
export const FLEET_COCKPIT = {
  topBarSpacing: 12,
  topBarContentHeight: 56,
  controlsTopMin: 148,
  statusBarHeight: 40,
  quickActionsHeight: 70,
  delayedBannerHeight: 76,
  driverSheetCollapsed: 92,
  tabBarHeight: 64,
  fabAboveNav: 86,
  sideGutter: 16,
} as const;

export type FleetCockpitMode = "split" | "immersive";

export type FleetCockpitLayout = {
  mode: FleetCockpitMode;
  mapHeight: number;
  topBarTop: number;
  topBarHeight: number;
  controlsTop: number;
  statusBarBottom: number;
  quickActionsBottom: number;
  delayedBannerBottom: number;
  driverSheetBottom: number;
  fabBottom: number;
  opsPanelBottomPad: number;
  cameraInsets: { top: number; right: number; bottom: number; left: number };
  livePillBottom: number;
};

export type CameraInsetOptions = {
  driverSheetOpen?: boolean;
  driverSheetSnap?: "collapsed" | "medium" | "expanded";
  /** Panneau tableau « Prochaines courses » toujours visible en cockpit. */
  cockpitOpsPanel?: boolean;
  sheetExtraBottom?: number;
  safeRight?: number;
  safeLeft?: number;
};

export function getMapCameraInsets(): { top: number; right: number; bottom: number; left: number } {
  const horizontal = FLEET_COCKPIT.sideGutter * 5;
  const top =
    FLEET_COCKPIT.controlsTopMin +
    FLEET_COCKPIT.statusBarHeight -
    FLEET_COCKPIT.topBarSpacing +
    4;
  const bottom =
    FLEET_COCKPIT.driverSheetCollapsed +
    FLEET_COCKPIT.tabBarHeight +
    FLEET_COCKPIT.quickActionsHeight +
    FLEET_COCKPIT.delayedBannerHeight +
    18;

  return { top, right: horizontal, bottom, left: horizontal };
}

/** Insets caméra dynamiques — évite occlusion par overlays/sheet. */
export function computeDynamicCameraInsets(
  layout: Pick<FleetCockpitLayout, "topBarHeight" | "driverSheetBottom">,
  options: CameraInsetOptions = {}
): { top: number; right: number; bottom: number; left: number } {
  const snap = options.driverSheetSnap ?? "collapsed";
  const base = getMapCameraInsets();
  let bottom = base.bottom;
  if (options.driverSheetOpen) {
    const sheetHeights = { collapsed: 92, medium: 200, expanded: 340 };
    bottom = Math.max(bottom, layout.driverSheetBottom + sheetHeights[snap] + 24);
  }
  if (options.cockpitOpsPanel) {
    bottom = Math.max(bottom, layout.driverSheetBottom + FLEET_COCKPIT.driverSheetCollapsed + 48);
  }
  if (options.sheetExtraBottom) {
    bottom += options.sheetExtraBottom;
  }
  const safeRight = options.safeRight ?? 0;
  const safeLeft = options.safeLeft ?? 0;
  return {
    top: Math.max(base.top, layout.topBarHeight + 24),
    right: Math.max(base.right, safeRight + 72),
    bottom,
    left: Math.max(base.left, safeLeft + 16),
  };
}

export function computeFleetCockpitLayout(
  usableHeight: number,
  topInset: number,
  bottomInset: number,
  immersive = true
): FleetCockpitLayout {
  const tabBarChrome =
    FLEET_COCKPIT.tabBarHeight + computeCompanyFloatingBottomPad(bottomInset);
  const topBarTop = topInset + FLEET_COCKPIT.topBarSpacing;
  const topBarHeight =
    topInset + FLEET_COCKPIT.topBarSpacing + FLEET_COCKPIT.topBarContentHeight;
  const driverSheetBottom = tabBarChrome + 6;
  const opsPanelBottomPad = tabBarChrome + 8;

  void immersive;
  const statusBarBottom = driverSheetBottom + FLEET_COCKPIT.driverSheetCollapsed + 12;
  const quickActionsBottom = statusBarBottom - (FLEET_COCKPIT.quickActionsHeight + 14);
  const delayedBannerBottom = quickActionsBottom - (FLEET_COCKPIT.delayedBannerHeight + 10);

  return {
    mode: "immersive",
    mapHeight: usableHeight,
    topBarTop,
    topBarHeight,
    controlsTop: Math.max(FLEET_COCKPIT.controlsTopMin, topBarHeight + 28),
    statusBarBottom,
    quickActionsBottom,
    delayedBannerBottom,
    driverSheetBottom,
    fabBottom: tabBarChrome + 52,
    livePillBottom: tabBarChrome + 52,
    opsPanelBottomPad,
    cameraInsets: getMapCameraInsets(),
  };
}
