import { describe, expect, it } from "@jest/globals";
import { computeCompanyFloatingBottomPad } from "../../../../design/navigation/BaseFloatingBar";
import {
  computeDynamicCameraInsets,
  computeFleetCockpitLayout,
  FLEET_COCKPIT,
  getMapCameraInsets,
  resolveUpcomingRidesBottomOffset,
} from "./companyFleetCockpitLayout";

describe("companyFleetCockpitLayout", () => {
  it("applique les insets caméra recommandés en mode immersif", () => {
    const layout = computeFleetCockpitLayout(800, 48, 34, true);
    const base = getMapCameraInsets();
    expect(layout.cameraInsets.top).toBe(180);
    expect(layout.cameraInsets.bottom).toBe(320);
    expect(layout.cameraInsets).toEqual(base);
  });

  it("augmente le padding bas quand la sheet chauffeur est ouverte", () => {
    const layout = computeFleetCockpitLayout(800, 48, 34, true);
    const closed = computeDynamicCameraInsets(layout, { driverSheetOpen: false });
    const open = computeDynamicCameraInsets(layout, {
      driverSheetOpen: true,
      driverSheetSnap: "medium",
    });
    expect(open.bottom).toBeGreaterThan(closed.bottom);
  });

  it("place Prochaines courses au-dessus de la nav, sans plafond 14 px", () => {
    const bottomInset = 34;
    const layout = computeFleetCockpitLayout(800, 48, bottomInset, true);
    const expected =
      FLEET_COCKPIT.tabBarHeight + computeCompanyFloatingBottomPad(bottomInset) + 6;
    expect(layout.driverSheetBottom).toBe(expected);
    expect(resolveUpcomingRidesBottomOffset(layout.driverSheetBottom)).toBe(expected);
    expect(resolveUpcomingRidesBottomOffset(layout.driverSheetBottom)).toBeGreaterThan(14);
  });

  it("élargit les insets latéraux selon la safe area", () => {
    const layout = computeFleetCockpitLayout(800, 48, 34, true);
    const base = computeDynamicCameraInsets(layout, { driverSheetOpen: false });
    const withSafe = computeDynamicCameraInsets(layout, {
      driverSheetOpen: false,
      safeRight: 34,
      safeLeft: 0,
    });
    expect(withSafe.right).toBeGreaterThan(base.right);
  });
});
