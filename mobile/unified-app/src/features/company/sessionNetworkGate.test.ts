import { beforeEach, describe, expect, it } from "@jest/globals";
import { resetDriverSessionNetworkGateForTests, setDriverSessionNetworkReady } from "../../core/network/driverSessionNetworkGate";
import { isCompanySessionNetworkReady } from "../../core/network/companySessionNetworkGate";

describe("company sessionNetworkGate", () => {
  beforeEach(() => {
    resetDriverSessionNetworkGateForTests();
  });

  it("reste fermé tant que SESSION_READY n’a pas ouvert la barrière partagée", () => {
    expect(isCompanySessionNetworkReady()).toBe(false);
    setDriverSessionNetworkReady(true);
    expect(isCompanySessionNetworkReady()).toBe(true);
  });
});
