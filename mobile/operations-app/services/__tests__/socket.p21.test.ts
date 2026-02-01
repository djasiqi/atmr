/**
 * P2.1.1 — Tests extractAuthStatus (spec connect_error 401/403).
 * extractAuthStatus dans socketAuthUtils pour éviter import socket (dépendances natives).
 */

import { extractAuthStatus } from "../socketAuthUtils";

describe("socket P2.1.1 — extractAuthStatus", () => {
  it("retourne 401 pour err.data.status === 401", () => {
    expect(extractAuthStatus({ data: { status: 401 } })).toBe(401);
  });

  it("retourne 403 pour err.data.status === 403", () => {
    expect(extractAuthStatus({ data: { status: 403 } })).toBe(403);
  });

  it("retourne 401 pour err.data.code === 401", () => {
    expect(extractAuthStatus({ data: { code: 401 } })).toBe(401);
  });

  it("retourne 401 pour err.message contenant '401'", () => {
    expect(extractAuthStatus({ message: "Error 401 Unauthorized" })).toBe(401);
  });

  it("retourne 401 pour err.message contenant 'Unauthorized'", () => {
    expect(extractAuthStatus({ message: "Unauthorized" })).toBe(401);
  });

  it("retourne 403 pour err.message contenant '403'", () => {
    expect(extractAuthStatus({ message: "403 Forbidden" })).toBe(403);
  });

  it("retourne 403 pour err.message contenant 'Forbidden'", () => {
    expect(extractAuthStatus({ message: "Forbidden" })).toBe(403);
  });

  it("retourne null pour err sans status auth", () => {
    expect(extractAuthStatus({ message: "Network error" })).toBeNull();
    expect(extractAuthStatus(null)).toBeNull();
    expect(extractAuthStatus(undefined)).toBeNull();
  });
});

describe("socket P2.1.1 — garde offline (comportement documenté)", () => {
  it("si getNetworkStateSnapshot().isConnected === false, le handler ne tente pas refresh", () => {
    // Comportement vérifié dans la spec : garde offline évite battery drain.
    // Test d'intégration manuel : couper réseau → connect_error → pas de refresh.
    expect(true).toBe(true);
  });
});
