import { describe, expect, it } from "@jest/globals";
import type { CompanyDispatchMission } from "../api/contracts";
import {
  filterDayMissionsForLocalSearch,
  filterMissionsByLocalSearch,
  getMissionNormalizedSearchIndex,
  normalizeMissionSearchText,
} from "./filterMissionsByLocalSearch";

function mission(partial: Partial<CompanyDispatchMission> & { mission_id: number }): CompanyDispatchMission {
  return {
    status: "assigned",
    ...partial,
  };
}

describe("filterMissionsByLocalSearch", () => {
  const sonia = mission({
    mission_id: 1,
    client_name: "Sonia BAUER",
    pickup_label: "Rue de la Gare 12, Genève",
    dropoff_label: "Hôpital",
    driver_name: "Karim",
    scheduled_at: "2026-09-05T15:30:00+02:00",
    identity: {
      source: { name: "Clinique des Grangettes" },
      ownership: { owner_company_name: "ATMR" },
    },
  });
  const other = mission({
    mission_id: 2,
    client_name: "Paul Martin",
    pickup_label: "Lausanne",
    dropoff_label: "Aéroport",
    driver_name: "Léa",
    scheduled_at: "2026-09-05T09:00:00+02:00",
  });
  const otherDay = mission({
    mission_id: 3,
    client_name: "Sonia HORS DATE",
    pickup_label: "Genève",
    scheduled_at: "2026-09-06T15:30:00+02:00",
  });

  it("neutralise casse et accents", () => {
    expect(normalizeMissionSearchText("Sónia")).toBe("sonia");
    expect(normalizeMissionSearchText("HÔPITAL")).toBe("hopital");
  });

  it("filtre patient, adresses, chauffeur et institution", () => {
    const day = [sonia, other];
    expect(filterMissionsByLocalSearch(day, "son").map((m) => m.mission_id)).toEqual([1]);
    expect(filterMissionsByLocalSearch(day, "GARE").map((m) => m.mission_id)).toEqual([1]);
    expect(filterMissionsByLocalSearch(day, "hopital").map((m) => m.mission_id)).toEqual([1]);
    expect(filterMissionsByLocalSearch(day, "karim").map((m) => m.mission_id)).toEqual([1]);
    expect(filterMissionsByLocalSearch(day, "grangettes").map((m) => m.mission_id)).toEqual([1]);
    expect(filterMissionsByLocalSearch(day, "atmr").map((m) => m.mission_id)).toEqual([1]);
  });

  it("effacer la recherche restaure toute la journée", () => {
    const day = [sonia, other];
    expect(filterMissionsByLocalSearch(day, "sonia")).toHaveLength(1);
    expect(filterMissionsByLocalSearch(day, "   ")).toEqual(day);
    expect(filterMissionsByLocalSearch(day, "")).toEqual(day);
  });

  it("n’inclut aucun résultat hors date", () => {
    const mixed = [sonia, otherDay];
    const found = filterDayMissionsForLocalSearch(mixed, "2026-09-05", "sonia");
    expect(found.map((m) => m.mission_id)).toEqual([1]);
    expect(found.some((m) => m.mission_id === 3)).toBe(false);
  });

  it("réutilise l’index normalisé et les références mission à chaque frappe", () => {
    const day = [sonia, other];
    const firstIndex = getMissionNormalizedSearchIndex(sonia);
    expect(getMissionNormalizedSearchIndex(sonia)).toBe(firstIndex);
    const withS = filterMissionsByLocalSearch(day, "S");
    const withSo = filterMissionsByLocalSearch(day, "So");
    expect(withS[0]).toBe(sonia);
    expect(withSo[0]).toBe(sonia);
    expect(withS[0]).toBe(withSo[0]);
  });

  it.each([10, 30, 60, 100] as const)(
    "filtre %s missions sans recréer les objets matchés",
    (count) => {
      const day = Array.from({ length: count }, (_, index) =>
        mission({
          mission_id: index + 1,
          client_name: index === 0 ? "Sonia DUPONT" : `Client ${index + 1}`,
          scheduled_at: "2026-09-06T10:00:00+02:00",
        })
      );
      const found = filterMissionsByLocalSearch(day, "sonia");
      expect(found).toHaveLength(1);
      expect(found[0]).toBe(day[0]);
    }
  );
});
