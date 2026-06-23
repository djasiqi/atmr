import {
  buildDriversBoundsSignature,
  buildDriversStructuralSignature,
  shouldTriggerFleetStructuralAutoFit,
} from "./fleetMapFitPadding";

describe("fleetMapFitPadding", () => {
  it("buildDriversStructuralSignature reste stable quand seules lat/lng changent", () => {
    const ids = [{ driver_id: 2 }, { driver_id: 1 }];
    expect(buildDriversStructuralSignature(ids)).toBe("1|2");

    const withGps = [
      { driver_id: 1, latitude: 46.2, longitude: 6.14 },
      { driver_id: 2, latitude: 46.3, longitude: 6.15 },
    ];
    expect(buildDriversStructuralSignature(withGps)).toBe("1|2");
  });

  it("buildDriversBoundsSignature change quand le GPS change", () => {
    const a = [{ driver_id: 1, latitude: 46.2, longitude: 6.14 }];
    const b = [{ driver_id: 1, latitude: 46.21, longitude: 6.14 }];
    expect(buildDriversBoundsSignature(a)).not.toBe(buildDriversBoundsSignature(b));
  });

  it("shouldTriggerFleetStructuralAutoFit : join/leave déclenche un fit", () => {
    expect(
      shouldTriggerFleetStructuralAutoFit({
        previousSignature: "1|2",
        nextSignature: "1|2|3",
        isFirstFit: false,
      })
    ).toBe(true);
  });

  it("shouldTriggerFleetStructuralAutoFit : tick GPS ne déclenche pas (signature inchangée)", () => {
    expect(
      shouldTriggerFleetStructuralAutoFit({
        previousSignature: "1|2",
        nextSignature: "1|2",
        isFirstFit: false,
      })
    ).toBe(false);
  });

  it("shouldTriggerFleetStructuralAutoFit : premier fit sans recadrage", () => {
    expect(
      shouldTriggerFleetStructuralAutoFit({
        previousSignature: "",
        nextSignature: "1|2",
        isFirstFit: true,
      })
    ).toBe(false);
  });
});
