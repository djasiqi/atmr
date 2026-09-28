import { clusterCompanyRouteMissions } from "./clusterCompanyRouteMissions";

describe("clusterCompanyRouteMissions", () => {
  it("ordonne une mission et laisse une course simple à sa place", () => {
    const simple = { id: "simple", trip_flags: null };
    const returnLeg = {
      id: "return",
      dropoff_label: "Rue du Test 1",
      trip_flags: { route_group_id: "grp", leg_number: 3, return_leg: true },
    };
    const second = {
      id: "clinic",
      dropoff_label: "Clinique La Colline",
      trip_flags: { route_group_id: "grp", leg_number: 2 },
    };
    const first = {
      id: "hug",
      dropoff_label: "HUG",
      trip_flags: { route_group_id: "grp", leg_number: 1 },
    };

    const ordered = clusterCompanyRouteMissions([simple, returnLeg, second, first]);

    expect(ordered.map((mission) => mission.id)).toEqual(["simple", "hug", "clinic", "return"]);
    expect(ordered[3].dropoff_label).toBe("Rue du Test 1");
  });
});
