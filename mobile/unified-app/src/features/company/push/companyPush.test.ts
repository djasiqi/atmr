import { describe, expect, it, jest } from "@jest/globals";

import { navigateFromCompanyPush } from "./companyPush";

describe("navigateFromCompanyPush", () => {
  it("dirige new_request vers la liste Demandes institution", () => {
    const push = jest.fn();
    const router = { push } as never;

    navigateFromCompanyPush(router, {
      type: "new_request",
      offer_id: 12,
      request_id: 34,
    });

    expect(push).toHaveBeenCalledWith("/(app)/(company)/offers");
  });

  it("redirige le deep link offre vers la liste", () => {
    const push = jest.fn();
    const router = { push } as never;

    navigateFromCompanyPush(router, {
      type: "new_request",
      offer_id: 12,
      request_id: 34,
      deep_link: "lirie://enterprise/offers/12?request=34",
    });

    expect(push).toHaveBeenCalledWith("/(app)/(company)/offers");
  });

  it("conserve la navigation course pour booking_id", () => {
    const push = jest.fn();
    const router = { push } as never;

    navigateFromCompanyPush(router, {
      type: "booking_assigned",
      booking_id: 99,
    });

    expect(push).toHaveBeenCalledWith({
      pathname: "/(app)/(company)/ride-details",
      params: { rideId: "99" },
    });
  });
});
