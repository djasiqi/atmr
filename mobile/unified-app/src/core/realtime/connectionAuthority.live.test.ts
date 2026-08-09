/**
 * Phase 2 mobile recovery — D3.3 LIVE test contre ws-service Docker.
 *
 * Connecte un VRAI `socket.io-client` à http://127.0.0.1:8001 (stack
 * docker-compose.phase2-validation.yml) et vérifie que `observeConnectionAuthority`
 * tag bien Sentry + incrémente le compteur local sur chaque connect + reconnect.
 *
 * Activé seulement quand RUN_LIVE_WS=1, sinon skip.
 */

import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

const RUN_LIVE = process.env.RUN_LIVE_WS === "1";
const describeLive = RUN_LIVE ? describe : describe.skip;

jest.mock("@sentry/react-native", () => ({
  setTag: jest.fn(),
}));

 
const { io } = require("socket.io-client");
 
const nodeCrypto = require("crypto");
 
const Sentry = require("@sentry/react-native") as { setTag: jest.Mock };
 
 
const { observeConnectionAuthority, getConnectionAuthorityMetricsSnapshot, resetConnectionAuthorityMetricsForTests } = require("./connectionAuthority");

const WS_URL = process.env.LIVE_WS_URL ?? "http://127.0.0.1:8001";
const JWT_SECRET = process.env.LIVE_JWT_SECRET ?? "validation-jwt-secret-only-for-local";

function base64UrlEncode(input: Buffer | string): string {
  const buf = Buffer.isBuffer(input) ? input : Buffer.from(input);
  return buf
    .toString("base64")
    .replace(/=+$/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

function makeToken(role: string, sub: string): string {
  const now = Math.floor(Date.now() / 1000);
  const header = base64UrlEncode(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const payload = base64UrlEncode(
    JSON.stringify({
      sub,
      role,
      iat: now,
      exp: now + 300,
      user_id: 100,
      driver_id: 100,
    })
  );
  const signingInput = `${header}.${payload}`;
  const signature = base64UrlEncode(
    nodeCrypto.createHmac("sha256", JWT_SECRET).update(signingInput).digest()
  );
  return `${signingInput}.${signature}`;
}

type AuthoritySocket = {
  connect: () => void;
  disconnect: () => void;
  on: (event: string, handler: (...args: unknown[]) => void) => void;
};

async function connectAndCaptureAuthority(timeoutMs = 5_000): Promise<void> {
  const token = makeToken("driver", `live-${Date.now()}`);
  const socket: AuthoritySocket = io(WS_URL, {
    transports: ["websocket"],
    auth: { token },
    reconnection: false,
  });
  await new Promise<void>((resolve, reject) => {
    const fail = setTimeout(() => reject(new Error("connect timeout")), timeoutMs);
    socket.on("connect", () => {
      clearTimeout(fail);
    });
    socket.on("connect_error", (err) => {
      clearTimeout(fail);
      reject(err instanceof Error ? err : new Error(String(err)));
    });
    socket.on("connection.authority", (payload) => {
      observeConnectionAuthority(payload as Record<string, unknown>);
      setTimeout(() => {
        socket.disconnect();
        resolve();
      }, 50);
    });
  });
}

describeLive("D3.3 connection.authority LIVE against ws-service", () => {
  beforeEach(() => {
    resetConnectionAuthorityMetricsForTests();
    Sentry.setTag.mockReset();
  });

  afterEach(() => {
    // small grace period to let docker socket cleanup
  });

  it("captures authority payload on a real socket.io connect", async () => {
    await connectAndCaptureAuthority();
    const snap = getConnectionAuthorityMetricsSnapshot();
    expect(snap.authorityObservedTotal).toBe(1);
    expect(snap.lastAuthority).toBe("ws-service");
    expect(snap.lastCanary).toBe(true);
    expect(snap.lastVersion).toBe("validation-v1");
  }, 20_000);

  it("aggregates authority count across 3 consecutive connects", async () => {
    await connectAndCaptureAuthority();
    await connectAndCaptureAuthority();
    await connectAndCaptureAuthority();
    const snap = getConnectionAuthorityMetricsSnapshot();
    expect(snap.authorityObservedTotal).toBe(3);
    expect(snap.authorityByName["ws-service"]).toBe(3);
    // Sentry.setTag a été appelé au moins 3x pour realtime.authority
    const authorityTagCalls = Sentry.setTag.mock.calls.filter(
      (call) => call[0] === "realtime.authority"
    );
    expect(authorityTagCalls.length).toBeGreaterThanOrEqual(3);
    expect(authorityTagCalls.every((c) => c[1] === "ws-service")).toBe(true);
  }, 30_000);
});

if (!RUN_LIVE) {
  describe("D3.3 live (skipped)", () => {
    it("requires RUN_LIVE_WS=1 + ws-service on 127.0.0.1:8001", () => {
      expect(RUN_LIVE).toBe(false);
    });
  });
}
