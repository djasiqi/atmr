import { describe, expect, it } from "@jest/globals";
import { resolveDriverMapDisplayPosition } from "./driverMapDisplayPosition";

describe("resolveDriverMapDisplayPosition", () => {
  it("GNSS prioritaire, snapshot API ignoré comme source", () => {
    const view = resolveDriverMapDisplayPosition({ lat: 46.2, lng: 6.14 }, 46.5, 6.9);
    expect(view.source).toBe("gnss");
    expect(view.gnssCoord).toEqual({ latitude: 46.2, longitude: 6.14 });
    expect(view.coord).toEqual(view.gnssCoord);
  });

  it("API sert uniquement d’affichage, jamais de GNSS", () => {
    const view = resolveDriverMapDisplayPosition(null, 46.2044, 6.1432);
    expect(view.source).toBe("api");
    expect(view.gnssCoord).toBeNull();
    expect(view.coord).toEqual({ latitude: 46.2044, longitude: 6.1432 });
  });

  it("aucune coordonnée inventée", () => {
    expect(resolveDriverMapDisplayPosition(null, null, null)).toEqual({
      coord: null,
      source: null,
      gnssCoord: null,
    });
  });
});
