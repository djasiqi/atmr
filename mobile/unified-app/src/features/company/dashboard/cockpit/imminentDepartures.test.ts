import { buildImminentDepartures } from "./imminentDepartures";
import type { CompanyDispatchMission } from "../../api/contracts";

const baseMission = (
  partial: Partial<CompanyDispatchMission> & { mission_id: number }
): CompanyDispatchMission =>
  ({
    status: "assigned",
    driver_id: 1,
    pickup_lat: 46.204,
    pickup_lon: 6.143,
    dropoff_lat: 46.21,
    dropoff_lon: 6.15,
    ...partial,
  }) as CompanyDispatchMission;

describe("buildImminentDepartures", () => {
  const nowMs = Date.parse("2026-05-19T10:00:00+02:00");

  it("affiche un halo pickup rouge (risk critical) quand la mission est en retard", () => {
    const missions = [
      baseMission({
        mission_id: 42,
        status: "en_route",
        scheduled_at: "2026-05-19T09:30:00+02:00",
        assignment_pickup_delay_minutes: 18,
      }),
    ];

    const result = buildImminentDepartures(missions, nowMs);
    const items = [...result.individual, ...result.clustered];

    expect(items).toHaveLength(1);
    expect(items[0]?.missionId).toBe(42);
    expect(items[0]?.risk).toBe("critical");
    expect(items[0]?.minutesUntil).toBeLessThan(0);
  });

  it("inclut une mission en retard même après l'heure de prise en charge prévue", () => {
    const missions = [
      baseMission({
        mission_id: 7,
        status: "assigned",
        scheduled_at: "2026-05-19T08:45:00+02:00",
        assignment_pickup_delay_minutes: 12,
      }),
    ];

    const result = buildImminentDepartures(missions, nowMs);
    expect(result.individual.some((d) => d.missionId === 7)).toBe(true);
  });

  it("n'inclut plus le halo pickup après la fenêtre retard (4 h)", () => {
    const missions = [
      baseMission({
        mission_id: 8,
        status: "assigned",
        scheduled_at: "2026-05-19T05:00:00+02:00",
        assignment_pickup_delay_minutes: null,
      }),
    ];

    const result = buildImminentDepartures(missions, nowMs);
    expect([...result.individual, ...result.clustered]).toHaveLength(0);
  });
});
