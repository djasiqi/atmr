import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockSetBackgroundHandler = jest.fn();
const mockGetMessaging = jest.fn();
const mockEmit = jest.fn();

jest.mock("react-native", () => ({
  Platform: { OS: "ios" },
}));

jest.mock("@react-native-firebase/app", () => ({
  getApp: jest.fn(),
}));

jest.mock("@react-native-firebase/messaging", () => ({
  getMessaging: (...args: unknown[]) => mockGetMessaging(...args),
  setBackgroundMessageHandler: (...args: unknown[]) => mockSetBackgroundHandler(...args),
  getToken: jest.fn(),
  onMessage: jest.fn(),
  onTokenRefresh: jest.fn(),
  requestPermission: jest.fn(),
}));

jest.mock("../../core/featureFlags/registry", () => ({
  isFeatureEnabled: () => true,
}));

jest.mock("../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmit(...args),
}));

jest.mock("./silentNotifications", () => ({
  isSilentPayload: (payload: Record<string, unknown>) => payload.type === "silent_update",
  shouldSuppressVisualPush: () => false,
}));

jest.mock("../../core/api/client", () => ({
  apiClient: { post: jest.fn().mockResolvedValue(undefined) },
}));

import {
  __resetDriverFcmBackgroundHandlerForTests,
  registerDriverFcmBackgroundHandler,
  setDriverFcmBackgroundCallback,
} from "./firebaseMessaging";

describe("firebaseMessaging background handler", () => {
  beforeEach(() => {
    __resetDriverFcmBackgroundHandlerForTests();
    mockSetBackgroundHandler.mockReset();
    mockGetMessaging.mockReturnValue({});
    mockEmit.mockReset();
  });

  it("registers handler once and resolves mutable callback at runtime", async () => {
    const first = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    const second = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);

    registerDriverFcmBackgroundHandler();
    registerDriverFcmBackgroundHandler(first);
    expect(mockSetBackgroundHandler).toHaveBeenCalledTimes(1);

    const handler = mockSetBackgroundHandler.mock.calls[0]?.[1] as (
      message: { data?: Record<string, unknown> }
    ) => Promise<void>;

    await handler({ data: { type: "silent_update" } });
    expect(first).toHaveBeenCalledTimes(1);

    setDriverFcmBackgroundCallback(second);
    await handler({ data: { type: "silent_update" } });
    expect(second).toHaveBeenCalledTimes(1);
  });

  it("emits no_callback telemetry when callback missing", async () => {
    registerDriverFcmBackgroundHandler();
    const handler = mockSetBackgroundHandler.mock.calls[0]?.[1] as (
      message: { data?: Record<string, unknown> }
    ) => Promise<void>;

    await handler({ data: { type: "silent_update" } });
    expect(mockEmit).toHaveBeenCalledWith(
      "push.fcm.background_handler_no_callback",
      expect.objectContaining({ platform: "ios" })
    );
  });
});
