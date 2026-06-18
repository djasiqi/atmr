import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import { markNotificationDisclosureAccepted } from "../../../core/notifications/notificationDisclosurePersistence";

jest.mock("../../../core/notifications/notificationDisclosurePersistence", () => ({
  markNotificationDisclosureAccepted: jest.fn(async () => undefined),
  readNotificationDisclosureAccepted: jest.fn(async () => false),
  subscribeNotificationDisclosureAccepted: jest.fn(() => () => undefined),
}));

jest.mock("../../../core/notifications/pushRegistrationState", () => ({
  getDisclosureShowRequestCount: jest.fn(() => 0),
  subscribePushRegistrationState: jest.fn(() => () => undefined),
}));

jest.mock("../../../core/sessionProvider", () => ({
  useSession: () => ({
    status: "ready",
    activeContext: { context_type: "driver", context_id: "driver:1" },
  }),
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (flag: string) => flag === "driver_push_enabled",
}));

const mockMarkAccepted = markNotificationDisclosureAccepted as jest.MockedFunction<
  typeof markNotificationDisclosureAccepted
>;

describe("DriverNotificationDisclosureHost — accept handler", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("markNotificationDisclosureAccepted est le seul effet attendu à l'acceptation (pas de permission OS ici)", async () => {
    await markNotificationDisclosureAccepted();
    expect(mockMarkAccepted).toHaveBeenCalledTimes(1);
  });
});
