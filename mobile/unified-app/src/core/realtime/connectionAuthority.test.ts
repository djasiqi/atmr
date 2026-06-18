import { beforeEach, describe, expect, it, jest } from "@jest/globals";

jest.mock("@sentry/react-native", () => ({
  setTag: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports
const Sentry = require("@sentry/react-native") as { setTag: jest.Mock };
 
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { observeConnectionAuthority, getConnectionAuthorityMetricsSnapshot, resetConnectionAuthorityMetricsForTests } = require("./connectionAuthority");

const setTagMock = Sentry.setTag;

describe("connectionAuthority observer (gate D3.3)", () => {
  beforeEach(() => {
    resetConnectionAuthorityMetricsForTests();
    setTagMock.mockReset();
  });

  it("ignores empty payloads", () => {
    observeConnectionAuthority(undefined);
    observeConnectionAuthority(null);
    const snap = getConnectionAuthorityMetricsSnapshot();
    expect(snap.authorityObservedTotal).toBe(0);
    expect(setTagMock).not.toHaveBeenCalled();
  });

  it("tags Sentry with ws-service authority + canary + version", () => {
    observeConnectionAuthority({
      authority: "ws-service",
      canary: true,
      version: "phase2-b3",
    });
    expect(setTagMock).toHaveBeenCalledWith("realtime.authority", "ws-service");
    expect(setTagMock).toHaveBeenCalledWith("realtime.canary", "true");
    expect(setTagMock).toHaveBeenCalledWith("realtime.ws_version", "phase2-b3");
  });

  it("normalizes unknown authority values", () => {
    observeConnectionAuthority({ authority: "weird-string" });
    const snap = getConnectionAuthorityMetricsSnapshot();
    expect(snap.lastAuthority).toBe("unknown");
    expect(snap.authorityByName.unknown).toBe(1);
    expect(setTagMock).toHaveBeenCalledWith("realtime.authority", "unknown");
  });

  it("aggregates counters by authority across observations", () => {
    observeConnectionAuthority({ authority: "ws-service" });
    observeConnectionAuthority({ authority: "ws-service" });
    observeConnectionAuthority({ authority: "backend" });
    const snap = getConnectionAuthorityMetricsSnapshot();
    expect(snap.authorityObservedTotal).toBe(3);
    expect(snap.authorityByName).toEqual({ "ws-service": 2, backend: 1 });
    expect(snap.lastAuthority).toBe("backend");
  });

  it("does not set canary/version tags when fields are missing", () => {
    observeConnectionAuthority({ authority: "backend" });
    expect(setTagMock).toHaveBeenCalledWith("realtime.authority", "backend");
    expect(setTagMock).not.toHaveBeenCalledWith(
      "realtime.canary",
      expect.anything()
    );
    expect(setTagMock).not.toHaveBeenCalledWith(
      "realtime.ws_version",
      expect.anything()
    );
  });
});
