import { describe, expect, it } from "@jest/globals";
import {
  canAttemptTrackingOperation,
  recordTrackingCircuitFailure,
  resetTrackingCircuitBreaker,
} from "./trackingCircuitBreaker";

describe("trackingCircuitBreaker", () => {
  it("opens after 3 failures", () => {
    resetTrackingCircuitBreaker();
    recordTrackingCircuitFailure(1000);
    recordTrackingCircuitFailure(2000);
    expect(canAttemptTrackingOperation(2500)).toBe(true);
    recordTrackingCircuitFailure(3000);
    expect(canAttemptTrackingOperation(3500)).toBe(false);
  });
});
