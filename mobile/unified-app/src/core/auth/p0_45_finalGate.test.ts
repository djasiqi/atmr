/**
 * P0-4/5 FINAL GATE — warm UI, AuthGuard mandatory status, BootBrand cold-only.
 */
import { describe, expect, it } from "@jest/globals";
import {
  resolveAuthGuardRedirect,
  shouldShowColdBootBrandSurface,
} from "./authGuardDecision";
import { resolveInitialRoute } from "../navigation/resolveInitialRoute";
import fs from "fs";
import path from "path";

const bootstrapAuth = {
  bootstrap_version: "1.0.0",
  is_authenticated: true,
  user: { id: "u-1", email: "a@b.c" },
  account_status: "active" as const,
  onboarding_status: { required: false },
  available_contexts: [
    {
      context_id: "driver:1",
      context_type: "driver" as const,
      label: "Chauffeur",
      permissions: ["mission:read"],
      is_default: true,
    },
  ],
  active_context_id: "driver:1",
  feature_flags: {},
  min_supported_app_version: "0.1.0",
  maintenance_mode: false,
  degraded_mode: false,
  server_time: new Date().toISOString(),
  request_id: "req-1",
};

describe("P0-5 FINAL warm recovery UI", () => {
  it("AUTHENTICATED → RECOVERING : AuthGuard redirect null + route app (pas public)", () => {
    expect(resolveAuthGuardRedirect(bootstrapAuth as any, "authenticated_online")).toBeNull();
    expect(resolveAuthGuardRedirect(bootstrapAuth as any, "auth_recovering")).toBeNull();
    expect(resolveInitialRoute(bootstrapAuth as any, null, "auth_recovering")).toBe(
      "/(app)/(driver)"
    );
    expect(
      resolveInitialRoute(
        { ...bootstrapAuth, is_authenticated: false } as any,
        null,
        "auth_recovering"
      )
    ).toBe("/(app)/(driver)");
  });

  it("BootBrandSurface cold-only : ready+recovering → false", () => {
    expect(
      shouldShowColdBootBrandSurface({
        status: "ready",
        mobileSessionStatus: "auth_recovering",
      })
    ).toBe(false);
    expect(
      shouldShowColdBootBrandSurface({
        status: "ready",
        mobileSessionStatus: "authenticated_offline",
      })
    ).toBe(false);
    expect(
      shouldShowColdBootBrandSurface({
        status: "bootstrapping",
        mobileSessionStatus: "initializing",
      })
    ).toBe(true);
  });

  it("login/public jamais via AuthGuard pendant recovering", () => {
    expect(resolveAuthGuardRedirect(bootstrapAuth as any, "auth_recovering")).toBeNull();
    expect(
      resolveAuthGuardRedirect(
        { ...bootstrapAuth, is_authenticated: false } as any,
        "auth_recovering"
      )
    ).toBeNull();
  });
});

describe("P0-4 FINAL AuthGuard legacy ban", () => {
  it("AuthGuard production passe toujours mobileSessionStatus", () => {
    const guardsPath = path.join(__dirname, "..", "guards.tsx");
    const src = fs.readFileSync(guardsPath, "utf8");
    expect(src).toMatch(
      /resolveAuthGuardRedirect\(\s*bootstrap\s*,\s*mobileSessionStatus\s*\)/
    );
    expect(src).not.toMatch(/resolveAuthGuardRedirect\(\s*bootstrap\s*\)/);
  });

  it("production callers resolveAuthGuardRedirect à 1 arg = 0 (src/core)", () => {
    const coreRoot = path.join(__dirname, "..");
    const offenders: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.name === "node_modules" || entry.name.endsWith(".test.ts") || entry.name.endsWith(".test.tsx")) {
          continue;
        }
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
          continue;
        }
        if (!/\.(ts|tsx)$/.test(entry.name)) continue;
        const text = fs.readFileSync(full, "utf8");
        // Appel à 1 argument : resolveAuthGuardRedirect(x) sans virgule avant )
        const re = /resolveAuthGuardRedirect\s*\(\s*[^,)]+\s*\)/g;
        let m: RegExpExecArray | null;
        while ((m = re.exec(text))) {
          offenders.push(`${full}: ${m[0]}`);
        }
      }
    };
    walk(coreRoot);
    expect(offenders).toEqual([]);
  });
});

describe("P0-5 FINAL terminal authority", () => {
  it("clearLocalAuthCredentialsLocked hors logout n'est appelé que via coordinator", () => {
    const coordinator = fs.readFileSync(
      path.join(__dirname, "authRecoveryCoordinator.ts"),
      "utf8"
    );
    expect(coordinator).toMatch(/applyTerminalRevocationIfCurrent/);
    expect(coordinator).toMatch(/clearLocalAuthCredentialsLocked/);

    // client.ts : clearLocalAuth uniquement dans logoutSession (pas sur 401 brut)
    const clientPath = path.join(__dirname, "..", "api", "client.ts");
    const client = fs.readFileSync(clientPath, "utf8");
    const clearCalls = [...client.matchAll(/clearLocalAuth\s*\(/g)];
    // export function clearLocalAuth + appel dans logoutSession finally
    expect(clearCalls.length).toBeLessThanOrEqual(3);
    expect(client).toMatch(/export async function logoutSession/);
    expect(client).toMatch(/opts\?\.skipLocalPurge/);
  });
});
