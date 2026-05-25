import type { AuthContext } from "./contracts/auth";
import {
  isCompanyDriverSwitchAllowedForRequest,
  shouldShowCompanyDriverContextSwitch,
} from "./contextSwitchPolicy";

const driverCtx: AuthContext = {
  context_id: "driver:1",
  context_type: "driver",
  label: "Chauffeur",
  permissions: [],
  is_default: false,
  allow_mobile_context_switch: true,
};

const companyCtx: AuthContext = {
  context_id: "company:9",
  context_type: "company",
  label: "Entreprise",
  permissions: [],
  is_default: true,
  allow_mobile_context_switch: true,
};

const driverNoSwitch: AuthContext = {
  ...driverCtx,
  allow_mobile_context_switch: false,
};

describe("contextSwitchPolicy", () => {
  it("autorise la bascule pour compte COMPANY avec dispatch (flags serveur)", () => {
    expect(
      isCompanyDriverSwitchAllowedForRequest(driverCtx, companyCtx, "COMPANY")
    ).toBe(true);
    expect(shouldShowCompanyDriverContextSwitch(driverCtx, companyCtx, "COMPANY")).toBe(
      true
    );
  });

  it("refuse chauffeur seul (rôle DRIVER)", () => {
    expect(
      isCompanyDriverSwitchAllowedForRequest(driverCtx, companyCtx, "DRIVER")
    ).toBe(false);
    expect(shouldShowCompanyDriverContextSwitch(driverCtx, companyCtx, "DRIVER")).toBe(
      false
    );
  });

  it("masque le bouton si allow_mobile_context_switch absent", () => {
    expect(
      shouldShowCompanyDriverContextSwitch(driverNoSwitch, companyCtx, "COMPANY")
    ).toBe(false);
  });
});
