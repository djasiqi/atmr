import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import {
  fetchFleetDirectionsPathNative,
} from "./fleetMapDirectionsNative";
import { resetFleetDirectionsCacheForTests } from "./fleetMapDirections";

const mockPost = jest.fn<(...args: any[]) => any>();
const mockEmitDriverTelemetry = jest.fn<(...args: any[]) => any>();

jest.mock("../../../../core/api/client", () => ({
  apiClient: {
    post: (...args: unknown[]) => mockPost(...args),
  },
}));

jest.mock("../../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmitDriverTelemetry(...args),
}));

const ORIGIN = { latitude: 46.205, longitude: 6.143 };
const DESTINATION = { latitude: 46.250, longitude: 6.180 };

const VALID_ENCODED_POLYLINE = "_p~iF~ps|U_ulLnnqC_mqNvxq`@";

describe("fetchFleetDirectionsPathNative (proxy backend)", () => {
  beforeEach(() => {
    mockPost.mockReset();
    mockEmitDriverTelemetry.mockReset();
    resetFleetDirectionsCacheForTests();
  });

  it("calls the backend proxy and decodes the polyline on success", async () => {
    mockPost.mockResolvedValueOnce({
      status: 200,
      data: {
        status: "OK",
        overview_polyline: VALID_ENCODED_POLYLINE,
        cached: false,
        source: "google_directions_v1",
      },
    });

    const path = await fetchFleetDirectionsPathNative({
      origin: ORIGIN,
      destination: DESTINATION,
    });

    expect(mockPost).toHaveBeenCalledTimes(1);
    expect(mockPost).toHaveBeenCalledWith(
      "/directions",
      expect.objectContaining({
        origin: expect.objectContaining({
          latitude: expect.any(Number),
          longitude: expect.any(Number),
        }),
        destination: expect.objectContaining({
          latitude: expect.any(Number),
          longitude: expect.any(Number),
        }),
        mode: "driving",
        region: "ch",
      })
    );
    expect(path.length).toBeGreaterThanOrEqual(2);
    expect(mockEmitDriverTelemetry).not.toHaveBeenCalled();
  });

  it("emits failure telemetry and returns [] when proxy reports REQUEST_DENIED", async () => {
    mockPost.mockResolvedValueOnce({
      status: 200,
      data: {
        status: "REQUEST_DENIED",
        overview_polyline: null,
        cached: false,
        error_message: "API key restricted",
        http_status: 200,
      },
    });

    const path = await fetchFleetDirectionsPathNative({
      origin: ORIGIN,
      destination: DESTINATION,
    });

    expect(path).toEqual([]);
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "company.fleet.directions.failed",
      expect.objectContaining({
        status: "REQUEST_DENIED",
        error_message: "API key restricted",
      })
    );
  });

  it("emits exception telemetry when the proxy call rejects", async () => {
    mockPost.mockRejectedValueOnce(new Error("network down"));

    const path = await fetchFleetDirectionsPathNative({
      origin: ORIGIN,
      destination: DESTINATION,
    });

    expect(path).toEqual([]);
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "company.fleet.directions.exception",
      expect.objectContaining({ error: "network down" })
    );
  });
});
