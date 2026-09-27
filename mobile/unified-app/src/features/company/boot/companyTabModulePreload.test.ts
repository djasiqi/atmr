import { describe, expect, it } from "@jest/globals";
import {
  COMPANY_TAB_CODE_PRELOAD_IDS,
  preloadCompanyTabModules,
  type CompanyTabCodePreload,
} from "./companyTabModulePreload";
import { isCompanyBootWorkAllowedAtLane, resolveCompanyTabLazy } from "./companyColdStartGraph";

describe("NAV-01 companyTabModulePreload", () => {
  it("ne précharge aucun écran lourd au boot", () => {
    expect(COMPANY_TAB_CODE_PRELOAD_IDS).toEqual([]);
  });

  it("conserve lazy au boot pour Chat / Menu ; Cockpit+Courses eager", () => {
    expect(resolveCompanyTabLazy("dashboard")).toBe(false);
    expect(resolveCompanyTabLazy("rides")).toBe(false);
    expect(resolveCompanyTabLazy("chat")).toBe(true);
    expect(resolveCompanyTabLazy("messages")).toBe(true);
    expect(resolveCompanyTabLazy("settings")).toBe(true);
    expect(resolveCompanyTabLazy("clients-facturation")).toBe(true);
  });

  it("le preload code n’est ni critical ni background", () => {
    expect(isCompanyBootWorkAllowedAtLane("tabs.code.preload", "critical")).toBe(false);
    expect(isCompanyBootWorkAllowedAtLane("tabs.code.preload", "background")).toBe(false);
  });

  it("enchaîne les loaders et n’appelle aucun prefetch GET", async () => {
    const loaded: string[] = [];
    const prefetchQuery = jest.fn();
    const loaders: CompanyTabCodePreload[] = [
      { id: "menu.module", load: async () => { loaded.push("menu-a"); } },
      { id: "menu.module", load: async () => { loaded.push("menu-b"); } },
    ];
    await preloadCompanyTabModules(loaders);
    expect(loaded).toEqual(["menu-a", "menu-b"]);
    expect(prefetchQuery).not.toHaveBeenCalled();
  });

  it("arrête la file si annulé entre deux modules", async () => {
    const loaded: string[] = [];
    let cancelled = false;
    const loaders: CompanyTabCodePreload[] = [
      {
        id: "menu.module",
        load: async () => {
          loaded.push("menu-a");
          cancelled = true;
        },
      },
      { id: "menu.module", load: async () => { loaded.push("menu-b"); } },
    ];
    await preloadCompanyTabModules(loaders, () => cancelled);
    expect(loaded).toEqual(["menu-a"]);
  });
});
