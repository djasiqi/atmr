import { describe, expect, it } from "@jest/globals";
import type { CompanyDriverLiveLocation } from "../../api/contracts";
import {
  buildLiveCoverageRows,
  computeGpsCoverageCounts,
  formatGpsCoverageA11y,
  formatGpsCoverageRatio,
  formatGpsCoverageSummary,
  isActiveFleetDriver,
} from "./liveGpsCoverage";

function recordedAt(ageSeconds: number, nowMs: number): string {
  return new Date(nowMs - ageSeconds * 1000).toISOString();
}

function driver(
  partial: Partial<CompanyDriverLiveLocation> & { driver_id: number }
): CompanyDriverLiveLocation {
  return {
    latitude: Object.prototype.hasOwnProperty.call(partial, "latitude") ? partial.latitude : 46.2,
    longitude: Object.prototype.hasOwnProperty.call(partial, "longitude") ? partial.longitude : 6.14,
    ...partial,
  };
}

describe("liveGpsCoverage", () => {
  const now = Date.parse("2026-09-05T12:00:00.000Z");

  const roster: CompanyDriverLiveLocation[] = [
    driver({
      driver_id: 1,
      first_name: "Sam",
      last_name: "Suter",
      location_status: "live",
      recorded_at: recordedAt(10, now),
    }),
    driver({
      driver_id: 2,
      first_name: "Marc",
      last_name: "Tosca",
      location_status: "stale",
      recorded_at: recordedAt(300, now),
    }),
    driver({
      driver_id: 3,
      first_name: "Julie",
      last_name: "Borel",
      latitude: null,
      longitude: null,
    }),
  ];

  it("compte 1/3 sur un roster mixte", () => {
    expect(computeGpsCoverageCounts(roster, now)).toEqual({ liveCount: 1, totalCount: 3 });
    expect(formatGpsCoverageRatio(1, 3)).toBe("1/3");
  });

  it("exclut les comptes historiques du dénominateur T", () => {
    const withHistory = [
      ...roster,
      driver({
        driver_id: 99,
        first_name: "Ancien",
        last_name: "Compte",
        is_active: false,
        location_status: "offline",
        latitude: null,
        longitude: null,
      }),
    ];
    expect(isActiveFleetDriver({ is_active: false })).toBe(false);
    expect(isActiveFleetDriver({ is_active: true })).toBe(true);
    expect(isActiveFleetDriver({})).toBe(true);
    expect(computeGpsCoverageCounts(withHistory, now)).toEqual({ liveCount: 1, totalCount: 3 });
  });

  it("garde 0/7 et 7/7 comme ratio explicite, sans état visuel distinct", () => {
    expect(formatGpsCoverageRatio(0, 7)).toBe("0/7");
    expect(formatGpsCoverageRatio(7, 7)).toBe("7/7");
    expect(formatGpsCoverageA11y(0, 7)).toBe("0 chauffeurs sur 7 en direct");
    expect(formatGpsCoverageA11y(7, 7)).toBe("7 chauffeurs sur 7 en direct");
  });

  it("fait basculer un live périmé hors du compteur après le timeout local", () => {
    const staleLive = [
      driver({
        driver_id: 1,
        first_name: "Sam",
        last_name: "Suter",
        location_status: "live",
        recorded_at: recordedAt(200, now),
      }),
    ];
    expect(computeGpsCoverageCounts(staleLive, now)).toEqual({ liveCount: 0, totalCount: 1 });
  });

  it("rédige la synthèse au singulier et au pluriel", () => {
    expect(formatGpsCoverageSummary(1, 7)).toBe(
      "1 chauffeur sur 7 transmet actuellement sa position"
    );
    expect(formatGpsCoverageSummary(3, 7)).toBe(
      "3 chauffeurs sur 7 transmettent actuellement leur position"
    );
    expect(formatGpsCoverageSummary(0, 7)).toBe(
      "Aucun chauffeur sur 7 ne transmet actuellement sa position"
    );
    expect(formatGpsCoverageA11y(1, 7)).toBe("1 chauffeur sur 7 en direct");
  });

  it("classe les chauffeurs en direct en tête et expose la dernière position hors ligne", () => {
    const rows = buildLiveCoverageRows(roster, now);
    expect(rows.map((row) => row.initials)).toEqual(["SS", "JB", "MT"]);
    expect(rows[0]).toMatchObject({ isLive: true, statusLabel: "En direct", lastPositionLabel: null });
    expect(rows[1].isLive).toBe(false);
    expect(rows[2]).toMatchObject({
      isLive: false,
      statusLabel: "Hors ligne",
      lastPositionLabel: "Dernière position : il y a 5 min",
    });
  });
});
