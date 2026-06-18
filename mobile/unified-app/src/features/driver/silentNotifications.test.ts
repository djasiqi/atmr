import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import {
  handleSilentPushPayload,
  isSilentPayload,
  shouldSuppressVisualPush,
} from "./silentNotifications";

const mockPost = jest.fn<() => Promise<void>>();
const mockRestart = jest.fn<() => Promise<void>>();
const mockEmit = jest.fn();

jest.mock("../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmit(...args),
}));

jest.mock("../../core/api/client", () => ({
  apiClient: { post: (...args: unknown[]) => mockPost(...args) },
}));

jest.mock("./services/backgroundLocationTask", () => ({
  restartNativeTrackingFromWake: (...args: unknown[]) => mockRestart(...args),
}));

describe("shouldSuppressVisualPush", () => {
  it("suppresses data-only mission_updated payloads", () => {
    expect(
      shouldSuppressVisualPush({ type: "mission_updated", title: "", body: "" })
    ).toBe(true);
  });
});

describe("isSilentPayload", () => {
  it.each([
    [{ type: "mission_refresh" }, true],
    [{ type: "silent_update" }, true],
    [{ content_available: 1 }, true],
    [{ silent: "true" }, true],
    [{ background: true }, true],
    [{ type: "booking_updated", title: "x" }, false],
  ])("detects silent payload %j -> %s", (input, expected) => {
    expect(isSilentPayload(input)).toBe(expected);
  });
});

describe("handleSilentPushPayload", () => {
  beforeEach(() => {
    mockPost.mockReset();
    mockRestart.mockReset();
    mockPost.mockResolvedValue(undefined);
    mockRestart.mockResolvedValue(undefined);
  });

  it("restarts tracking and acks with result=acked", async () => {
    const onResync = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    await handleSilentPushPayload({ type: "silent_update", mission_id: 12 }, onResync);

    expect(onResync).toHaveBeenCalledWith(12);
    expect(mockRestart).toHaveBeenCalledWith("silent_push_payload");
    expect(mockPost).toHaveBeenCalledWith("/driver/me/push-notifications/silent-ack", {
      sync_type: "silent_update",
      result: "acked",
    });
  });

  it("ignores non-silent payloads", async () => {
    const onResync = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    await handleSilentPushPayload({ type: "chat_message", title: "Hi" }, onResync);
    expect(onResync).not.toHaveBeenCalled();
    expect(mockRestart).not.toHaveBeenCalled();
    expect(mockPost).not.toHaveBeenCalled();
  });
});
