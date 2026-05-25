import { describe, expect, it } from "@jest/globals";
import { isUrgentHubMessage } from "./urgentAlertSound";

describe("isUrgentHubMessage", () => {
  it("detects urgent priority", () => {
    expect(isUrgentHubMessage({ priority: "urgent" })).toBe(true);
  });

  it("detects system emergency content", () => {
    expect(
      isUrgentHubMessage({
        message_type: "system",
        content: "⚠ Patient absent",
      })
    ).toBe(true);
  });

  it("detects driver_hub alert type", () => {
    expect(isUrgentHubMessage({ alert_type: "driver_hub_patient_absent" })).toBe(true);
  });

  it("ignores normal chat", () => {
    expect(isUrgentHubMessage({ priority: "normal", content: "Bonjour" })).toBe(false);
  });
});
