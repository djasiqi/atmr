import { describe, expect, it } from "@jest/globals";

import { resolveMinMoveMeters } from "./driverLiveLocationMerge";
import type { CompanyDriverLiveLocation } from "../api/contracts";

const base: CompanyDriverLiveLocation = {
  driver_id: 1,
  latitude: 46.2,
  longitude: 6.14,
  location_status: "live",
  mission_id: null,
  is_background: false,
  accepted_observability_only: false,
};

describe("resolveMinMoveMeters", () => {
  it("uses legacy 5m when dynamic filter disabled", () => {
    expect(resolveMinMoveMeters({ ...base, accuracy: 20 })).toBe(5);
  });
});
