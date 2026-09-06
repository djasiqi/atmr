import { describe, expect, it } from "@jest/globals";
import {
  companyBootWorkByLane,
  isCompanyBootWorkAllowedAtLane,
  resolveCompanyTabLazy,
} from "./companyColdStartGraph";

describe("companyColdStartGraph", () => {
  it("Cockpit et Courses restent eager ; le reste est lazy", () => {
    expect(resolveCompanyTabLazy("dashboard")).toBe(false);
    expect(resolveCompanyTabLazy("rides")).toBe(false);
    expect(resolveCompanyTabLazy("clients-facturation")).toBe(true);
    expect(resolveCompanyTabLazy("invoices")).toBe(true);
    expect(resolveCompanyTabLazy("settings")).toBe(true);
    expect(resolveCompanyTabLazy("chat")).toBe(true);
    expect(resolveCompanyTabLazy("fleet-map")).toBe(true);
  });

  it("interdit billing / mode / optimizer / companies.me au boot", () => {
    const neverIds = companyBootWorkByLane("never");
    expect(neverIds).toEqual(
      expect.arrayContaining([
        "billing.invoices",
        "clients.list",
        "companies.me",
        "dispatch.mode.get",
        "optimizer.status",
      ])
    );
    expect(isCompanyBootWorkAllowedAtLane("optimizer.status", "critical")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("billing.invoices", "background")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("dispatch.mode.get", "critical")).toBe(false);
  });

  it("le premier écran n’attend pas inbox / unread / delays", () => {
    expect(isCompanyBootWorkAllowedAtLane("inbox.notifications", "critical")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("chat.unread", "critical")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("rides.delays", "critical")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("rides.j.page1", "critical")).toBe(true);
    expect(isCompanyBootWorkAllowedAtLane("drivers.live", "critical")).toBe(true);
    expect(isCompanyBootWorkAllowedAtLane("inbox.notifications", "background")).toBe(true);
    expect(isCompanyBootWorkAllowedAtLane("tabs.code.preload", "critical")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("tabs.code.preload", "background")).toBe(true);
  });
});
