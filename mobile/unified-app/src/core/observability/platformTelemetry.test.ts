import { describe, expect, it, jest } from "@jest/globals";
import { emitPlatformTelemetry } from "./platformTelemetry";

const mockEmitDriverTelemetry = jest.fn();
const mockInfo = jest.spyOn(console, "info").mockImplementation(() => undefined);

jest.mock("./driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmitDriverTelemetry(...args),
}));

describe("platform telemetry", () => {
  it("routes known driver events to driver telemetry pipe", () => {
    emitPlatformTelemetry("driver", "tracking.send.failure", {
      source: "test",
      mission_id: 1,
    });

    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "tracking.send.failure",
      expect.objectContaining({ source: "test", mission_id: 1 })
    );
  });

  it("uses generic platform log pipe for non-driver events", () => {
    emitPlatformTelemetry("company", "dispatch.assignment.created", {
      source: "test-company",
      context_id: "company:42",
    });

    expect(mockInfo).toHaveBeenCalledWith(
      "[platform-telemetry] company.dispatch.assignment.created",
      expect.objectContaining({ source: "test-company", context_id: "company:42" })
    );
  });
});
