import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { emitCompanyDispatchTelemetry } from "./companyTelemetry";

const mockEmitPlatformTelemetry = jest.fn();

jest.mock("../../../core/observability/platformTelemetry", () => ({
  emitPlatformTelemetry: (...args: unknown[]) => mockEmitPlatformTelemetry(...args),
}));

describe("company telemetry (shadow-ready)", () => {
  beforeEach(() => {
    mockEmitPlatformTelemetry.mockReset();
  });

  it("does not emit while company dispatch flag is disabled", () => {
    const emitted = emitCompanyDispatchTelemetry("company.dispatch.opened", {
      source: "company.telemetry.test",
      context_id: "company:42",
    });

    expect(emitted).toBe(false);
    expect(mockEmitPlatformTelemetry).not.toHaveBeenCalled();
  });

  it("can emit only when explicitly forced for preparation checks", () => {
    const emitted = emitCompanyDispatchTelemetry(
      "company.dispatch.optimizer_status_requested",
      {
        source: "company.telemetry.test",
        context_id: "company:42",
      },
      { allowWhenDisabled: true }
    );

    expect(emitted).toBe(true);
    expect(mockEmitPlatformTelemetry).toHaveBeenCalledWith(
      "company",
      "company.dispatch.optimizer_status_requested",
      expect.objectContaining({
        source: "company.telemetry.test",
        context_id: "company:42",
        timestamp_client: expect.any(String),
      })
    );
  });

  it("supports all runtime dispatch telemetry events with required payload markers", () => {
    const events = [
      "company.dispatch.opened",
      "company.dispatch.driver_selected",
      "company.dispatch.delay_invalidated",
      "company.dispatch.optimizer_status_requested",
      "company.dispatch.socket_state_changed",
    ] as const;

    events.forEach((event) => {
      const emitted = emitCompanyDispatchTelemetry(
        event,
        {
          source: "company.telemetry.runtime",
          context_type: "company",
          context_id: "company:42",
          company_id: "42",
          mission_id: 101,
          driver_id: 7,
          previous_state: "connecting",
          state: "healthy",
          reason: "reconnect_success",
        },
        { allowWhenDisabled: true }
      );
      expect(emitted).toBe(true);
    });

    expect(mockEmitPlatformTelemetry).toHaveBeenCalledTimes(events.length);
  });
});
