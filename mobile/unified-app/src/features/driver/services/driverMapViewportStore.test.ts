import { describe, expect, it, beforeEach } from "@jest/globals";
import {
  __resetDriverMapViewportStoreForTests,
  parseDriverMapViewport,
} from "./driverMapViewportStore";

const now = Date.parse("2026-09-06T12:00:00.000Z");

describe("parseDriverMapViewport", () => {
  beforeEach(() => {
    __resetDriverMapViewportStoreForTests();
  });

  it("accepte un viewport récent utilisable", () => {
    const parsed = parseDriverMapViewport(
      {
        center: { latitude: 46.2, longitude: 6.14 },
        latitudeDelta: 0.04,
        longitudeDelta: 0.04,
        saved_at: "2026-09-06T10:00:00.000Z",
      },
      now
    );
    expect(parsed?.center.latitude).toBe(46.2);
    expect(parsed?.latitudeDelta).toBe(0.04);
  });

  it("rejette un viewport trop ancien", () => {
    expect(
      parseDriverMapViewport(
        {
          center: { latitude: 46.2, longitude: 6.14 },
          latitudeDelta: 0.04,
          longitudeDelta: 0.04,
          saved_at: "2026-08-01T10:00:00.000Z",
        },
        now
      )
    ).toBeNull();
  });

  it("ne transforme jamais le viewport en position chauffeur", () => {
    const parsed = parseDriverMapViewport(
      {
        center: { latitude: 46.2, longitude: 6.14 },
        latitudeDelta: 0.04,
        longitudeDelta: 0.04,
        saved_at: "2026-09-06T10:00:00.000Z",
      },
      now
    );
    expect(parsed && "lat" in parsed).toBe(false);
    expect(parsed && "lng" in parsed).toBe(false);
  });
});
