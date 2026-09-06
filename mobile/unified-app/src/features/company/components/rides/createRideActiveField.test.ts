import { describe, expect, it } from "@jest/globals";
import {
  applyCreateRideActiveField,
  createRideMissingHint,
  nextCreateRideFieldAfterSelection,
} from "./createRideActiveField";

describe("createRideActiveField", () => {
  it("n’autorise qu’un seul champ actif", () => {
    expect(applyCreateRideActiveField("client", "pickup", true)).toBe("pickup");
    expect(applyCreateRideActiveField("pickup", "pickup", false)).toBe(null);
    expect(applyCreateRideActiveField("pickup", "client", false)).toBe("pickup");
  });

  it("enchaîne Client → Départ → Destination → Date", () => {
    expect(nextCreateRideFieldAfterSelection("client")).toBe("pickup");
    expect(nextCreateRideFieldAfterSelection("pickup")).toBe("dropoff");
    expect(nextCreateRideFieldAfterSelection("dropoff")).toBe("schedule");
    expect(nextCreateRideFieldAfterSelection("schedule")).toBe(null);
  });

  it("compose un hint unique de complétion", () => {
    expect(
      createRideMissingHint({
        hasClient: false,
        hasPickup: true,
        hasDropoff: true,
        hasSchedule: false,
      })
    ).toBe("À compléter : client, date et heure");
    expect(
      createRideMissingHint({
        hasClient: true,
        hasPickup: true,
        hasDropoff: true,
        hasSchedule: true,
      })
    ).toBeNull();
  });
});
