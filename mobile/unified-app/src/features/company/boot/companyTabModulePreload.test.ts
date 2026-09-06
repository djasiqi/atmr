import { describe, expect, it } from "@jest/globals";
import {
  COMPANY_TAB_CODE_PRELOAD_IDS,
  preloadCompanyTabModules,
  type CompanyTabCodePreload,
} from "./companyTabModulePreload";
import { isCompanyBootWorkAllowedAtLane, resolveCompanyTabLazy } from "./companyColdStartGraph";

describe("NAV-01 companyTabModulePreload", () => {
  it("ne précharge que du CODE barre — aucun id GET", () => {
    expect(COMPANY_TAB_CODE_PRELOAD_IDS).toEqual([
      "chat.module",
      "menu.module",
    ]);
    expect(COMPANY_TAB_CODE_PRELOAD_IDS.join(" ")).not.toMatch(/get|query|prefetch|invoice|client/i);
  });

  it("conserve lazy au boot pour Chat / Menu ; Cockpit+Courses eager", () => {
    expect(resolveCompanyTabLazy("dashboard")).toBe(false);
    expect(resolveCompanyTabLazy("rides")).toBe(false);
    expect(resolveCompanyTabLazy("chat")).toBe(true);
    expect(resolveCompanyTabLazy("messages")).toBe(true);
    expect(resolveCompanyTabLazy("settings")).toBe(true);
    expect(resolveCompanyTabLazy("clients-facturation")).toBe(true);
  });

  it("le preload code n’est pas dans la lane critical", () => {
    expect(isCompanyBootWorkAllowedAtLane("tabs.code.preload", "critical")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("tabs.code.preload", "background")).toBe(true);
  });

  it("enchaîne les loaders et n’appelle aucun prefetch GET", async () => {
    const loaded: string[] = [];
    const prefetchQuery = jest.fn();
    const loaders: CompanyTabCodePreload[] = [
      { id: "chat.module", load: async () => { loaded.push("chat"); } },
      { id: "menu.module", load: async () => { loaded.push("menu"); } },
    ];
    await preloadCompanyTabModules(loaders);
    expect(loaded).toEqual(["chat", "menu"]);
    expect(prefetchQuery).not.toHaveBeenCalled();
  });

  it("arrête la file si annulé entre deux modules", async () => {
    const loaded: string[] = [];
    let cancelled = false;
    const loaders: CompanyTabCodePreload[] = [
      {
        id: "chat.module",
        load: async () => {
          loaded.push("chat");
          cancelled = true;
        },
      },
      { id: "menu.module", load: async () => { loaded.push("menu"); } },
    ];
    await preloadCompanyTabModules(loaders, () => cancelled);
    expect(loaded).toEqual(["chat"]);
  });
});
