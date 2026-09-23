/**
 * Gate canary authentifié — ws-service only.
 *
 * Prérequis : ATMR_SOCKET_AB_TOKEN (JWT company) dans l'environnement.
 * Ne journalise jamais le token ni Authorization.
 *
 *   node backend/scripts/socketio_auth_canary.mjs
 *
 * Variables :
 *   ATMR_SOCKET_AB_TOKEN   (requis)
 *   ATMR_SOCKET_AB_CYCLES  (défaut 20)
 *   ATMR_SOCKET_AB_EVENT   (défaut urgent_alert — événement d'écoute)
 *   ATMR_SOCKET_AB_WAIT_EVENT_MS (défaut 0 = pas d'attente d'événement)
 *   ATMR_SIO_MODULE        (chemin socket.io-client optionnel)
 */
import http from "node:http";
import https from "node:https";
import { pathToFileURL } from "node:url";

const URL = process.env.ATMR_SOCKET_AB_URL || "https://api.lirie.ch";
const CYCLES = Number(process.env.ATMR_SOCKET_AB_CYCLES || "20");
const GAP_MS = Number(process.env.ATMR_SOCKET_AB_GAP_MS || "600");
const TIMEOUT_MS = Number(process.env.ATMR_SOCKET_AB_TIMEOUT_MS || "10000");
const WAIT_EVENT_MS = Number(process.env.ATMR_SOCKET_AB_WAIT_EVENT_MS || "0");
const EVENT_NAME = (process.env.ATMR_SOCKET_AB_EVENT || "urgent_alert").trim();
const TOKEN = (process.env.ATMR_SOCKET_AB_TOKEN || "").trim();
const CANARY = "1";

if (!TOKEN) {
  process.stdout.write(
    JSON.stringify(
      {
        error: "ATMR_SOCKET_AB_TOKEN manquant",
        token_present: false,
      },
      null,
      2
    ) + "\n"
  );
  process.exit(2);
}

function decodeClaimsUnsafe(token) {
  try {
    const part = token.split(".")[1];
    if (!part) return null;
    const pad = "=".repeat((4 - (part.length % 4)) % 4);
    const json = Buffer.from(part.replace(/-/g, "+").replace(/_/g, "/") + pad, "base64").toString(
      "utf8"
    );
    const payload = JSON.parse(json);
    return {
      role: typeof payload.role === "string" ? payload.role : null,
      company_id: typeof payload.company_id === "number" ? payload.company_id : null,
      has_user_id: payload.user_id != null || payload.sub != null,
      exp_present: typeof payload.exp === "number",
    };
  } catch {
    return null;
  }
}

const claims = decodeClaimsUnsafe(TOKEN);
const companyId = claims && claims.company_id != null ? claims.company_id : null;
const contextId = companyId != null ? `company:${companyId}` : "company:unknown";

const origHttp = http.request;
const origHttps = https.request;
let currentTrace = null;

function wrap(orig) {
  return function wrapped(options, callback) {
    const opts = typeof options === "string" ? new URL(options) : options;
    const path = String(opts.path || opts.pathname || "");
    const headers = opts.headers || {};
    const canary = headers["X-WS-Canary"] || headers["x-ws-canary"] || "";
    const row = {
      method: String(opts.method || "GET"),
      path: path.split(" ")[0],
      status: 0,
      canary: String(canary),
      hasSid: path.includes("sid="),
      body: "",
    };
    const track = Boolean(currentTrace && path.includes("/socket.io"));
    if (track) currentTrace.requests.push(row);
    const req = orig.call(this, options, (res) => {
      if (track) {
        row.status = res.statusCode;
        res.on("data", (chunk) => {
          if (row.body.length < 180) row.body += chunk.toString("utf8");
        });
      }
      if (typeof callback === "function") callback(res);
    });
    if (track) {
      const origEnd = req.end;
      req.end = function endWithHeaderSnapshot(...args) {
        row.canary = String(
          req.getHeader("X-WS-Canary") ||
            req.getHeader("x-ws-canary") ||
            row.canary ||
            ""
        );
        return origEnd.apply(this, args);
      };
    }
    return req;
  };
}

http.request = wrap(origHttp);
https.request = wrap(origHttps);

function clientSpecifier() {
  const given = process.env.ATMR_SIO_MODULE || "";
  if (!given) return "socket.io-client";
  if (given.startsWith("file:")) return given;
  return pathToFileURL(given).href;
}

const { io } = await import(clientSpecifier());

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function oneCycle({ waitEvent }) {
  const trace = { requests: [] };
  currentTrace = trace;
  const agent = new https.Agent({ keepAlive: false });
  const socket = io(URL, {
    path: "/socket.io",
    transports: ["polling"],
    upgrade: false,
    reconnection: false,
    timeout: TIMEOUT_MS,
    agent,
    extraHeaders: {
      "X-WS-Canary": CANARY,
      Authorization: `Bearer ${TOKEN}`,
    },
    auth: { token: TOKEN },
    query: {
      context_id: contextId,
      surface: "company",
      ...(companyId != null ? { company_id: String(companyId) } : {}),
    },
  });

  const result = {
    handshake_status: null,
    first_sid_request_status: null,
    connect_ok: false,
    join_company: false,
    join_ack: null,
    authority: null,
    event_received: false,
    event_type: null,
    event_id: null,
    event_received_at: null,
    invalid_session: false,
    rate_limit: false,
    error: "",
    header_missing: 0,
  };

  let settled = false;
  const outcome = new Promise((resolve) => {
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };

    socket.on("connection.authority", (payload) => {
      if (payload && typeof payload === "object") {
        result.authority = {
          authority: payload.authority ?? null,
          canary: payload.canary ?? null,
          version: payload.version ?? null,
        };
      }
    });

    socket.on(EVENT_NAME, (payload) => {
      result.event_received = true;
      result.event_type = EVENT_NAME;
      result.event_received_at = new Date().toISOString();
      if (payload && typeof payload === "object" && typeof payload.event_id === "string") {
        result.event_id = payload.event_id;
      }
      if (waitEvent) finish();
    });

    socket.on("connect", () => {
      result.connect_ok = true;
      socket.emit("join_company", {}, (ack) => {
        result.join_ack =
          ack && typeof ack === "object"
            ? { ok: Boolean(ack.ok) }
            : { ok: false };
        if (ack && ack.ok === true) result.join_company = true;
        if (!waitEvent) {
          finish();
          return;
        }
        setTimeout(finish, WAIT_EVENT_MS);
      });
      setTimeout(() => {
        if (!result.join_company && !waitEvent) finish();
      }, 3000);
    });

    socket.on("connect_error", (err) => {
      const message = err && err.message ? String(err.message) : String(err || "");
      result.error = message.slice(0, 180);
      const lower = message.toLowerCase();
      if (lower.includes("invalid session")) result.invalid_session = true;
      if (message.includes("RATE_LIMIT") || lower.includes("rate limit")) {
        result.rate_limit = true;
      }
      finish();
    });

    setTimeout(finish, waitEvent ? WAIT_EVENT_MS + TIMEOUT_MS : TIMEOUT_MS + 500);
  });

  await outcome;
  socket.close();
  agent.destroy();
  currentTrace = null;

  const handshake = trace.requests.find((row) => !row.hasSid);
  const withSid = trace.requests.find((row) => row.hasSid);
  result.handshake_status = handshake ? handshake.status : null;
  result.first_sid_request_status = withSid ? withSid.status : null;
  const bodies = trace.requests.map((row) => row.body || "").join("\n");
  if (bodies.toLowerCase().includes("invalid session")) result.invalid_session = true;
  if ((trace.requests || []).some((row) => row.hasSid && row.status === 400)) {
    result.invalid_session = true;
  }
  result.header_missing = trace.requests.filter((row) => row.canary !== CANARY).length;
  result.sid_request_400 = (trace.requests || []).filter(
    (row) => row.hasSid && row.status === 400
  ).length;
  result.request_count = trace.requests.length;
  return result;
}

const summary = {
  cycles: 0,
  handshake_200: 0,
  connect_ok: 0,
  join_company_ok: 0,
  sid_request_400: 0,
  invalid_session: 0,
  rate_limit: 0,
  header_missing_on_socket_request: 0,
  authority_ws_service: 0,
};

const samples = [];
for (let i = 0; i < CYCLES; i += 1) {
  const cycle = await oneCycle({ waitEvent: false });
  summary.cycles += 1;
  if (cycle.handshake_status === 200) summary.handshake_200 += 1;
  if (cycle.connect_ok) summary.connect_ok += 1;
  if (cycle.join_company) summary.join_company_ok += 1;
  summary.sid_request_400 += cycle.sid_request_400 || 0;
  if (cycle.invalid_session) summary.invalid_session += 1;
  if (cycle.rate_limit) summary.rate_limit += 1;
  summary.header_missing_on_socket_request += cycle.header_missing || 0;
  if (cycle.authority && cycle.authority.authority === "ws-service") {
    summary.authority_ws_service += 1;
  }
  if (samples.length < 5) {
    samples.push({
      i,
      handshake_status: cycle.handshake_status,
      connect_ok: cycle.connect_ok,
      join_company: cycle.join_company,
      join_ack: cycle.join_ack,
      authority: cycle.authority,
      sid_request_400: cycle.sid_request_400,
      invalid_session: cycle.invalid_session,
      error: cycle.error,
      header_missing: cycle.header_missing,
    });
  }
  if (cycle.rate_limit) break;
  if (i + 1 < CYCLES) await sleep(GAP_MS);
}

let eventProbe = null;
if (WAIT_EVENT_MS > 0) {
  const publishedAt = process.env.ATMR_SOCKET_AB_EVENT_PUBLISHED_AT || null;
  const startedWait = new Date().toISOString();
  const cycle = await oneCycle({ waitEvent: true });
  eventProbe = {
    listen_event: EVENT_NAME,
    wait_ms: WAIT_EVENT_MS,
    published_at: publishedAt,
    wait_started_at: startedWait,
    received: cycle.event_received,
    event_type: cycle.event_type,
    event_id: cycle.event_id,
    received_at: cycle.event_received_at,
    latency_ms:
      publishedAt && cycle.event_received_at
        ? Date.parse(cycle.event_received_at) - Date.parse(publishedAt)
        : null,
    connect_ok: cycle.connect_ok,
    join_company: cycle.join_company,
    invalid_session: cycle.invalid_session,
    sid_request_400: cycle.sid_request_400,
    error: cycle.error,
  };
}

const report = {
  mode: "auth_canary",
  started: new Date().toISOString(),
  url: URL,
  token_present: true,
  token_length: TOKEN.length,
  claims,
  context_id: contextId,
  cycles_requested: CYCLES,
  transports: ["polling"],
  upgrade: false,
  keepAlive: false,
  canary_header: CANARY,
  summary,
  samples,
  event_probe: eventProbe,
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
process.exit(
  summary.invalid_session === 0 &&
    summary.sid_request_400 === 0 &&
    summary.connect_ok === summary.cycles &&
    summary.join_company_ok === summary.cycles
    ? 0
    : 1
);
