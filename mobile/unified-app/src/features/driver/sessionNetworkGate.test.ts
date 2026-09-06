import { beforeEach, describe, expect, it } from "@jest/globals";
import {
  resetDriverSessionNetworkGateForTests,
  setDriverSessionNetworkReady,
} from "../../core/network/driverSessionNetworkGate";
import { isDriverNetworkSessionReady } from "./sessionNetworkGate";

describe("isDriverNetworkSessionReady", () => {
  beforeEach(() => {
    resetDriverSessionNetworkGateForTests();
  });

  it("ignore le status React : seul le flag réseau bootstrap compte", () => {
    expect(isDriverNetworkSessionReady("ready")).toBe(false);
    expect(isDriverNetworkSessionReady("idle")).toBe(false);
    setDriverSessionNetworkReady(true);
    expect(isDriverNetworkSessionReady("idle")).toBe(true);
    expect(isDriverNetworkSessionReady("ready")).toBe(true);
  });
});
