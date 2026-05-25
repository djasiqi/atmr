import {
  getNativeGoogleMapViewStyleProps,
  getNativeOperationalMapBehaviorProps,
} from "./nativeGoogleMapStyle";

describe("nativeGoogleMapStyle", () => {
  it("masque les POI et le chrome Google natif", () => {
    const behavior = getNativeOperationalMapBehaviorProps();
    expect(behavior.showsPointsOfInterest).toBe(false);
    expect(behavior.toolbarEnabled).toBe(false);
    expect(behavior.moveOnMarkerPress).toBe(false);
  });

  it("fusionne style Lirie et comportement opérationnel", () => {
    const props = getNativeGoogleMapViewStyleProps();
    expect(props.showsPointsOfInterest).toBe(false);
    expect(props.customMapStyle != null || props.googleMapId != null).toBe(true);
  });
});
