import { beforeEach, describe, expect, it } from "@jest/globals";
import { clearActiveMissionScreen, setActiveMissionScreen } from "./activeScreenStore";
import {
  resolveForegroundPresentation,
  shouldSuppressForActiveScreen,
} from "./notificationPolicy";

describe("notificationPolicy", () => {
  beforeEach(() => {
    clearActiveMissionScreen(-1);
  });

  it("suppresses mission_updated on active mission screen", () => {
    setActiveMissionScreen(42);
    const result = shouldSuppressForActiveScreen({
      payload: { type: "mission_updated", mission_id: 42 },
    });
    expect(result.suppress).toBe(true);
  });

  it("hides silent mission_refresh in foreground", () => {
    const presentation = resolveForegroundPresentation({ rawType: "mission_refresh", silent: true });
    expect(presentation.shouldShowBanner).toBe(false);
  });

  it("plays sound for critical types", () => {
    const presentation = resolveForegroundPresentation({ rawType: "mission_cancelled" });
    expect(presentation.shouldPlaySound).toBe(true);
  });
});
