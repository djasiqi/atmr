/**
 * Preuve A/B du long-polling Engine.IO.
 *
 * Même client que l'app (socket.io-client), transports polling seuls, upgrade coupé.
 * A : api.lirie.ch sans X-WS-Canary (backend Gunicorn, 6 workers).
 * B : le même hôte avec X-WS-Canary: 1 sur chaque requête Engine.IO (ws-service).
 *
 * Ne change aucun worker, aucune route, aucun déploiement.
 *
 * Prérequis :
 *   npm install --prefix backend/scripts socket.io-client@4.8.1
 * Lancement :
 *   node backend/scripts/socketio_polling_ab.mjs
 * Jeton optionnel (join_company) : ATMR_SOCKET_AB_TOKEN, non journalisé.
 */
import http from "node:http";
import https from "node:https";
import { pathToFileURL } from "node:url";

const URL = process.env.ATMR_SOCKET_AB_URL || "https://api.lirie.ch";
const CYCLES = Number(process.env.ATMR_SOCKET_AB_CYCLES || "50");
const GAP_MS = Number(process.env.ATMR_SOCKET_AB_GAP_MS || "800");
const TIMEOUT_MS = Number(process.env.ATMR_SOCKET_AB_TIMEOUT_MS || "8000");
const TOKEN = (process.env.ATMR_SOCKET_AB_TOKEN || "").trim();
const CANARY = "1";

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
          req.getHeader("X-WS-Canary") || req.getHeader("x-ws-canary") || row.canary || ""
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

function emptySummary() {
  return {
    cycles: 0,
    handshake_200: 0,
    sid_request_400: 0,
    invalid_session: 0,
    join_company_ok: 0,
    connect_ok: 0,
    rate_limit: 0,
    header_missing_on_socket_request: 0,
    stopped_early: "",
  };
}

async function oneCycle(label, withCanary) {
  const trace = { requests: [] };
  currentTrace = trace;
  const extraHeaders = withCanary ? { "X-WS-Canary": CANARY } : {};
  // Sans keep-alive, chaque requête est une nouvelle connexion TCP.
  // Traefik sticky ne choisit pas le worker Gunicorn : c'est le cas du client mobile.
  const agent = new https.Agent({ keepAlive: false });
  const socket = io(URL, {
    path: "/socket.io",
    transports: ["polling"],
    upgrade: false,
    reconnection: false,
    timeout: TIMEOUT_MS,
    agent,
    extraHeaders,
    auth: TOKEN ? { token: TOKEN } : {},
    query: {
      context_id: "ab-harness",
      surface: "company",
    },
  });

  const result = {
    label,
    handshake_status: null,
    sid: "",
    first_sid_request_status: null,
    join_company: false,
    disconnect: false,
    reconnect: false,
    invalid_session: false,
    connect_ok: false,
    rate_limit: false,
    error: "",
  };

  let settled = false;
  const outcome = new Promise((resolve) => {
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    socket.on("connect", () => {
      result.connect_ok = true;
      if (!TOKEN) {
        finish();
        return;
      }
      let acked = false;
      const timer = setTimeout(() => finish(), 2500);
      socket.once("joined_company", () => {
        acked = true;
        result.join_company = true;
        clearTimeout(timer);
        finish();
      });
      socket.emit("join_company", {}, (ack) => {
        if (ack && ack.ok === true) {
          acked = true;
          result.join_company = true;
          clearTimeout(timer);
          finish();
        }
      });
      setTimeout(() => {
        if (!acked) finish();
      }, 2500);
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
    socket.on("disconnect", () => {
      result.disconnect = true;
    });
    socket.io.on("reconnect_attempt", () => {
      result.reconnect = true;
    });
    setTimeout(finish, TIMEOUT_MS + 500);
  });

  await outcome;
  socket.close();
  agent.destroy();
  currentTrace = null;

  const handshake = trace.requests.find((row) => !row.hasSid);
  const withSid = trace.requests.find((row) => row.hasSid);
  result.handshake_status = handshake ? handshake.status : null;
  result.first_sid_request_status = withSid ? withSid.status : null;
  const sidFromUrl = (withSid?.path || "").match(/sid=([^&]+)/);
  const sidFromBody = (handshake?.body || "").match(/"sid"\s*:\s*"([^"]+)"/);
  const rawSid = sidFromUrl ? sidFromUrl[1] : sidFromBody ? sidFromBody[1] : "";
  result.sid = rawSid ? decodeURIComponent(rawSid) : "";
  const bodies = trace.requests.map((row) => row.body || "").join("\n");
  if (bodies.toLowerCase().includes("invalid session")) result.invalid_session = true;
  if (result.first_sid_request_status === 400) result.invalid_session = true;
  if (result.error.toLowerCase().includes("invalid session")) result.invalid_session = true;

  if (withCanary) {
    const missing = trace.requests.filter((row) => row.canary !== CANARY);
    result.header_missing = missing.length;
  } else {
    result.header_missing = 0;
  }
  result.request_count = trace.requests.length;
  result.requests = trace.requests.map((row) => ({
    method: row.method,
    status: row.status,
    hasSid: row.hasSid,
    canary: row.canary,
    invalid_body: (row.body || "").toLowerCase().includes("invalid session"),
  }));
  return result;
}

function accumulate(summary, cycle) {
  summary.cycles += 1;
  if (cycle.handshake_status === 200) summary.handshake_200 += 1;
  const sawSid400 = (cycle.requests || []).some((row) => row.hasSid && row.status === 400);
  if (sawSid400) summary.sid_request_400 += 1;
  if (cycle.invalid_session) summary.invalid_session += 1;
  if (cycle.join_company) summary.join_company_ok += 1;
  if (cycle.connect_ok) summary.connect_ok += 1;
  if (cycle.rate_limit) summary.rate_limit += 1;
  summary.header_missing_on_socket_request += cycle.header_missing || 0;
}

async function runSide(label, withCanary) {
  const summary = emptySummary();
  const samples = [];
  for (let i = 0; i < CYCLES; i += 1) {
    const cycle = await oneCycle(label, withCanary);
    accumulate(summary, cycle);
    if (samples.length < 3 || cycle.invalid_session || cycle.first_sid_request_status === 400) {
      if (samples.length < 8) {
        samples.push({
          i,
          handshake_status: cycle.handshake_status,
          first_sid_request_status: cycle.first_sid_request_status,
          invalid_session: cycle.invalid_session,
          connect_ok: cycle.connect_ok,
          join_company: cycle.join_company,
          request_count: cycle.request_count,
          header_missing: cycle.header_missing,
          error: cycle.error,
          sid_prefix: cycle.sid ? cycle.sid.slice(0, 4) : "",
          requests: cycle.requests,
        });
      }
    }
    if (cycle.rate_limit) {
      summary.stopped_early = "RATE_LIMIT";
      break;
    }
    if (i + 1 < CYCLES) await sleep(GAP_MS);
  }
  return { summary, samples };
}

const started = new Date().toISOString();
const sideA = await runSide("A", false);
const sideB = await runSide("B", true);
const report = {
  started,
  url: URL,
  cycles_requested: CYCLES,
  gap_ms: GAP_MS,
  token_present: Boolean(TOKEN),
  transports: ["polling"],
  upgrade: false,
  A: sideA,
  B: sideB,
};
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
process.exit(0);
