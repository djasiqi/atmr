import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  buildStableDedupeKey,
  displayLocalDriverPush,
  LOCAL_PUSH_DEDUP_TTL_MS,
  resetPushLocalDisplayForTests,
  shouldSkipLocalPushDisplay,
} from "./pushLocalDisplay";
import { emitDriverTelemetry } from "../observability/driverTelemetry";

const mockGetItem = jest.fn() as jest.Mock<any>;
const mockSetItem = jest.fn() as jest.Mock<any>;
const mockCreateChannel = jest.fn() as jest.Mock<any>;
const mockDisplayNotification = jest.fn() as jest.Mock<any>;
const mockLoadNotifee = jest.fn() as jest.Mock<any>;

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: (...args: unknown[]) => mockGetItem(...args),
  setItem: (...args: unknown[]) => mockSetItem(...args),
}));

jest.mock("../../features/driver/notifeeCompat", () => ({
  loadNotifee: (...args: unknown[]) => mockLoadNotifee(...args),
}));

jest.mock("../observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("react-native", () => ({
  Platform: { OS: "android" },
}));

describe("pushLocalDisplay", () => {
  beforeEach(() => {
    resetPushLocalDisplayForTests();
    jest.clearAllMocks();
    mockGetItem.mockResolvedValue(null);
    mockSetItem.mockResolvedValue(undefined);
    mockCreateChannel.mockResolvedValue(undefined);
    mockDisplayNotification.mockResolvedValue(undefined);
    mockLoadNotifee.mockResolvedValue({
      default: {
        createChannel: mockCreateChannel,
        displayNotification: mockDisplayNotification,
      },
      AndroidImportance: { HIGH: 4 },
    });
  });

  it("priorise dedupe_key explicite puis clé mission stable puis event_id", () => {
    expect(buildStableDedupeKey({ dedupe_key: "custom:key" })).toBe("custom:key");
    expect(buildStableDedupeKey({ deduplication_key: "canon:key" })).toBe("canon:key");
    expect(
      buildStableDedupeKey({
        type: "booking_assigned",
        booking_id: 35438,
        event_id: "evt-99",
      })
    ).toBe("booking:35438:event:assigned");
    expect(
      buildStableDedupeKey({ type: "booking_reassigned", booking_id: 42 })
    ).toBe("booking:42:event:reassigned");
    expect(buildStableDedupeKey({ event_id: "evt-99", type: "chat_message" })).toBe(
      "event:evt-99"
    );
  });

  it("skip duplicate display within TTL", async () => {
    const key = "event:evt-dup";
    mockGetItem.mockResolvedValue(
      JSON.stringify({ [key]: Date.now() + LOCAL_PUSH_DEDUP_TTL_MS })
    );
    expect(await shouldSkipLocalPushDisplay(key)).toBe(true);
  });

  it("skip local display when remote notification block is present", async () => {
    const displayed = await displayLocalDriverPush(
      { type: "booking_assigned", title: "T", body: "B", dedupe_key: "event:1" },
      "background",
      { remoteNotification: { title: "Remote", body: "Body" } }
    );
    expect(displayed).toBe(false);
    expect(mockLoadNotifee).not.toHaveBeenCalled();
    expect(emitDriverTelemetry).toHaveBeenCalledWith(
      "push_remote_notification_payload_detected",
      expect.objectContaining({ dedupe_key: "event:1" })
    );
  });

  it("displayLocalDriverPush appelle notifee une seule fois", async () => {
    const payload = {
      type: "booking_assigned",
      title: "Nouvelle course",
      body: "Client",
      dedupe_key: "event:display-once",
    };

    const first = await displayLocalDriverPush(payload, "foreground");
    const second = await displayLocalDriverPush(payload, "background");

    expect(first).toBe(true);
    expect(second).toBe(false);
    expect(mockDisplayNotification).toHaveBeenCalledTimes(1);
    expect(emitDriverTelemetry).toHaveBeenCalledWith(
      "push_display_local",
      expect.objectContaining({ dedupe_key: "event:display-once" })
    );
  });
});
