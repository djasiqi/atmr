import { describe, expect, it } from "@jest/globals";
import {
  decideMissionMerge,
  mergeMissionsGuarded,
  replaceMissionsGuarded,
} from "./missionRevisionGuard";
import type { DriverMission } from "../types";

function mission(partial: Partial<DriverMission> & { id: number }): DriverMission {
  return { status: "ASSIGNED", ...partial } as DriverMission;
}

describe("decideMissionMerge (M2 anti-régression par revision)", () => {
  it("applique un snapshot plus récent (revision supérieure)", () => {
    const local = mission({ id: 1, status: "EN_ROUTE", assignment_id: 10, mission_revision: 3 });
    const incoming = mission({ id: 1, status: "IN_PROGRESS", assignment_id: 10, mission_revision: 4 });
    expect(decideMissionMerge(local, incoming)).toBe("apply");
  });

  it("ignore un snapshot strictement plus ancien (même lifecycle)", () => {
    const local = mission({ id: 1, status: "COMPLETED", assignment_id: 10, mission_revision: 6 });
    const incoming = mission({ id: 1, status: "EN_ROUTE", assignment_id: 10, mission_revision: 2 });
    expect(decideMissionMerge(local, incoming)).toBe("keep_local_stale_incoming");
  });

  it("applique un snapshot d'égale revision (idempotence serveur)", () => {
    const local = mission({ id: 1, status: "EN_ROUTE", assignment_id: 10, mission_revision: 3 });
    const incoming = mission({ id: 1, status: "EN_ROUTE", assignment_id: 10, mission_revision: 3 });
    expect(decideMissionMerge(local, incoming)).toBe("apply");
  });

  it("applique toujours un nouveau lifecycle (assignment_id différent), même à revision inférieure", () => {
    const local = mission({ id: 1, status: "COMPLETED", assignment_id: 10, mission_revision: 6 });
    const incoming = mission({ id: 1, status: "ASSIGNED", assignment_id: 11, mission_revision: 1 });
    expect(decideMissionMerge(local, incoming)).toBe("apply");
  });

  it("applique quand les revisions sont absentes (compat backend ancien)", () => {
    const local = mission({ id: 1, status: "EN_ROUTE" });
    const incoming = mission({ id: 1, status: "ASSIGNED" });
    expect(decideMissionMerge(local, incoming)).toBe("apply");
  });

  it("applique quand il n'y a pas d'état local", () => {
    const incoming = mission({ id: 1, status: "ASSIGNED", mission_revision: 1 });
    expect(decideMissionMerge(undefined, incoming)).toBe("apply");
  });
});

describe("replaceMissionsGuarded (polling plein)", () => {
  it("respecte l'appartenance serveur mais conserve l'état local plus récent", () => {
    const previous = [
      mission({ id: 1, status: "IN_PROGRESS", assignment_id: 10, mission_revision: 5 }),
      mission({ id: 2, status: "ASSIGNED", assignment_id: 20, mission_revision: 1 }),
    ];
    const incoming = [
      // Poll parti avant le PUT : revision périmée → état local conservé.
      mission({ id: 1, status: "EN_ROUTE", assignment_id: 10, mission_revision: 4 }),
      // Mission 3 nouvelle ; mission 2 disparue côté serveur.
      mission({ id: 3, status: "ASSIGNED", assignment_id: 30, mission_revision: 0 }),
    ];
    const { missions, staleIgnoredCount } = replaceMissionsGuarded(previous, incoming);
    expect(staleIgnoredCount).toBe(1);
    expect(missions.map((m) => m.id)).toEqual([1, 3]);
    expect(missions[0]!.status).toBe("IN_PROGRESS");
    expect(missions[0]!.mission_revision).toBe(5);
  });
});

describe("mergeMissionsGuarded (delta reconcile)", () => {
  it("conserve les missions absentes du delta et ignore les snapshots périmés", () => {
    const previous = [
      mission({ id: 1, status: "ARRIVED", assignment_id: 10, mission_revision: 3 }),
      mission({ id: 2, status: "ASSIGNED", assignment_id: 20, mission_revision: 1 }),
    ];
    const incoming = [
      mission({ id: 1, status: "EN_ROUTE", assignment_id: 10, mission_revision: 2 }),
    ];
    const { missions, staleIgnoredCount } = mergeMissionsGuarded(previous, incoming);
    expect(staleIgnoredCount).toBe(1);
    const byId = new Map(missions.map((m) => [m.id, m]));
    expect(byId.get(1)!.status).toBe("ARRIVED");
    expect(byId.get(2)!.status).toBe("ASSIGNED");
  });

  it("applique le delta plus récent", () => {
    const previous = [
      mission({ id: 1, status: "EN_ROUTE", assignment_id: 10, mission_revision: 2 }),
    ];
    const incoming = [
      mission({ id: 1, status: "IN_PROGRESS", assignment_id: 10, mission_revision: 3 }),
    ];
    const { missions, staleIgnoredCount } = mergeMissionsGuarded(previous, incoming);
    expect(staleIgnoredCount).toBe(0);
    expect(missions[0]!.status).toBe("IN_PROGRESS");
  });
});
