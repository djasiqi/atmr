import { describe, expect, it } from "@jest/globals";
import type { AuthContext, BootstrapResponse } from "../contracts/auth";
import { canEnterFromLocalSession } from "./canEnterFromLocalSession";

const context = {
  context_type: "company",
  context_id: "company:42",
  label: "ATMR",
  permissions: [],
  is_default: true,
} as AuthContext;

describe("canEnterFromLocalSession", () => {
  it("autorise l’entrée si bootstrap authentifié + contexte", () => {
    expect(
      canEnterFromLocalSession({
        bootstrap: { is_authenticated: true } as BootstrapResponse,
        activeContext: context,
      })
    ).toBe(true);
  });

  it("refuse anonyme, sans contexte, ou bootstrap incomplet", () => {
    expect(
      canEnterFromLocalSession({
        bootstrap: { is_authenticated: false } as BootstrapResponse,
        activeContext: context,
      })
    ).toBe(false);
    expect(
      canEnterFromLocalSession({
        bootstrap: { is_authenticated: true } as BootstrapResponse,
        activeContext: null,
      })
    ).toBe(false);
    expect(canEnterFromLocalSession({})).toBe(false);
  });
});
