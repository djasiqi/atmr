import { describe, expect, it } from "@jest/globals";
import {
  resolveMissionTransportAuthority,
  shouldSkipMissionPolling,
} from "./transportAuthority";

describe("transportAuthority", () => {
  it("prefers socket when connected and healthy", () => {
    expect(
      resolveMissionTransportAuthority({
        connected: true,
        degradedMode: false,
        transportAuthority: "socket",
        actualTransport: "websocket",
      })
    ).toBe("socket");
    expect(
      shouldSkipMissionPolling({
        connected: true,
        degradedMode: false,
        transportAuthority: "socket",
        actualTransport: "websocket",
      })
    ).toBe(true);
  });

  it("falls back to polling when degraded", () => {
    expect(
      shouldSkipMissionPolling({
        connected: true,
        degradedMode: true,
        transportAuthority: "socket",
        actualTransport: "websocket",
      })
    ).toBe(false);
  });
});
