import { describe, expect, it } from "@jest/globals";
import {
  isPointVisibleInRegion,
  isUsableMapRegion,
  resolveColdStartCameraAction,
  type DriverMapRegion,
} from "./driverMapCameraPolicy";

const region: DriverMapRegion = {
  latitude: 46.2,
  longitude: 6.14,
  latitudeDelta: 0.08,
  longitudeDelta: 0.08,
};

const inside = { latitude: 46.21, longitude: 6.15 };
const outside = { latitude: 46.9, longitude: 7.4 };

describe("driverMapCameraPolicy", () => {
  it("rejette une région invalide", () => {
    expect(isUsableMapRegion({ ...region, latitudeDelta: 0 })).toBe(false);
    expect(isUsableMapRegion({ ...region, latitude: 200 })).toBe(false);
  });

  it("chauffeur déjà visible → aucun recentrage, consume", () => {
    expect(isPointVisibleInRegion(inside, region)).toBe(true);
    expect(
      resolveColdStartCameraAction({
        consumed: false,
        gnssPoint: inside,
        currentRegion: region,
        hadUsefulViewport: true,
      })
    ).toEqual({ action: "none", consume: true });
  });

  it("chauffeur hors viewport → une animation, consume", () => {
    expect(isPointVisibleInRegion(outside, region)).toBe(false);
    expect(
      resolveColdStartCameraAction({
        consumed: false,
        gnssPoint: outside,
        currentRegion: region,
        hadUsefulViewport: true,
      })
    ).toEqual({ action: "recenter", consume: true });
  });

  it("aucune viewport utile → recentrage autorisé", () => {
    expect(
      resolveColdStartCameraAction({
        consumed: false,
        gnssPoint: inside,
        currentRegion: null,
        hadUsefulViewport: false,
      })
    ).toEqual({ action: "recenter", consume: true });
  });

  it("déjà consommé → plus jamais de recentrage cold start", () => {
    expect(
      resolveColdStartCameraAction({
        consumed: true,
        gnssPoint: outside,
        currentRegion: region,
        hadUsefulViewport: true,
      })
    ).toEqual({ action: "none", consume: true });
  });

  it("pas encore de GNSS → on attend, sans consommer", () => {
    expect(
      resolveColdStartCameraAction({
        consumed: false,
        gnssPoint: null,
        currentRegion: region,
        hadUsefulViewport: true,
      })
    ).toEqual({ action: "none", consume: false });
  });
});
