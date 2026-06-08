import { describe, expect, it } from "@jest/globals";

import { resolveFleetMarkerAnchor } from "./resolveFleetMarkerAnchor";

describe("resolveFleetMarkerAnchor", () => {
  it("centers PNG circle markers", () => {
    expect(
      resolveFleetMarkerAnchor({
        uri: "data:image/png;base64,abc",
        width: 56,
        height: 56,
      })
    ).toEqual({ x: 0.5, y: 0.5 });
  });

  it("anchors Lirie pin markers at bottom center", () => {
    expect(
      resolveFleetMarkerAnchor({
        uri: "asset:/marker.png",
        width: 56,
        height: 56,
        assetModule: 1,
      })
    ).toEqual({ x: 0.5, y: 1 });
  });
});
