import {
  assertDriverSurface,
  getAuthSurfaceRole,
  setAuthSurfaceRole,
} from "@/services/authSurface";

describe("authSurface", () => {
  afterEach(() => {
    setAuthSurfaceRole("enterprise");
  });

  it("expose le rôle courant", () => {
    setAuthSurfaceRole("driver");
    expect(getAuthSurfaceRole()).toBe("driver");
  });

  it("assertDriverSurface retourne false hors chauffeur", () => {
    setAuthSurfaceRole("enterprise");
    expect(assertDriverSurface("test")).toBe(false);
    setAuthSurfaceRole("driver");
    expect(assertDriverSurface("test")).toBe(true);
  });
});
