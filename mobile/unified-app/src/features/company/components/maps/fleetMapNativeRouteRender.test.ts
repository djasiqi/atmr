import { resolveNativeMissionRouteStrokes } from "./fleetMapNativeRouteRender";
import type { FleetMissionRouteStyle } from "./fleetMapMissionVisual";

describe("fleetMapNativeRouteRender", () => {
  it("caps native stroke widths below web values", () => {
    const webStyle: FleetMissionRouteStyle = {
      color: "#EF4444",
      glowColor: "rgba(239, 68, 68, 0.28)",
      strokeWidth: 6.5,
      glowWidth: 15.5,
      opacity: 1,
      lineDashPattern: null,
      zIndex: 50,
    };

    const native = resolveNativeMissionRouteStrokes(webStyle);
    expect(native.mainStroke).toBeLessThanOrEqual(4);
    expect(native.glowStroke).toBeLessThanOrEqual(8);
    expect(native.glowStroke).toBeGreaterThan(native.mainStroke);
  });
});
