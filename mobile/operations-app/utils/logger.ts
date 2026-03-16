import Constants from "expo-constants";
import * as Application from "expo-application";

const APP_ID_PROD = "ch.liri.operations";
const appVariantRaw = String(
  Constants.expoConfig?.extra?.APP_VARIANT ??
    process.env.APP_VARIANT ??
    "dev"
).toLowerCase();
const appEnvRaw = String(
  process.env.EXPO_PUBLIC_APP_ENV ??
    Constants.expoConfig?.extra?.EXPO_PUBLIC_APP_ENV ??
    ""
).toLowerCase();
const nativeAppId = String(Application.applicationId ?? "");
const isProductionRuntime =
  !__DEV__ &&
  (nativeAppId === APP_ID_PROD ||
    appVariantRaw === "prod" ||
    appEnvRaw === "production");
const allowProdDebugLogs =
  String(process.env.EXPO_PUBLIC_ENABLE_PROD_LOGS ?? "").trim() === "1";
const isDevelopment = !isProductionRuntime || allowProdDebugLogs;

type LogLevel = "debug" | "info" | "warn" | "error";

const LEVEL_TAGS: Record<LogLevel, string> = {
  debug: "DBG",
  info: "INF",
  warn: "WRN",
  error: "ERR",
};

function ts(): string {
  return new Date().toISOString().slice(11, 23);
}

function fmt(level: LogLevel, module: string, msg: string): string {
  return `${ts()} ${LEVEL_TAGS[level]} [${module}] ${msg}`;
}

const SENSITIVE_KEY_RE =
  /(token|authorization|cookie|password|secret|api.?key|refresh|session)/i;

function redactString(value: string): string {
  if (value.length <= 12) return "***";
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

function sanitizeValue(value: unknown, keyHint?: string): unknown {
  if (value == null) return value;
  if (typeof value === "string") {
    if (keyHint && SENSITIVE_KEY_RE.test(keyHint)) {
      return redactString(value);
    }
    return value;
  }
  if (typeof value !== "object") return value;
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeValue(item));
  }
  const input = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(input)) {
    if (SENSITIVE_KEY_RE.test(k)) {
      if (typeof v === "string") out[k] = redactString(v);
      else if (v == null) out[k] = v;
      else out[k] = "***";
      continue;
    }
    out[k] = sanitizeValue(v, k);
  }
  return out;
}

function createModuleLogger(module: string) {
  return {
    debug(msg: string, data?: Record<string, unknown>) {
      if (!isDevelopment) return;
      if (data) console.debug(fmt("debug", module, msg), sanitizeValue(data));
      else console.debug(fmt("debug", module, msg));
    },
    info(msg: string, data?: Record<string, unknown>) {
      if (!isDevelopment) return;
      if (data) console.log(fmt("info", module, msg), sanitizeValue(data));
      else console.log(fmt("info", module, msg));
    },
    warn(msg: string, data?: Record<string, unknown>) {
      if (data) console.warn(fmt("warn", module, msg), sanitizeValue(data));
      else console.warn(fmt("warn", module, msg));
    },
    error(msg: string, data?: Record<string, unknown>) {
      if (data) console.error(fmt("error", module, msg), sanitizeValue(data));
      else console.error(fmt("error", module, msg));
    },
    success(msg: string, data?: Record<string, unknown>) {
      if (!isDevelopment) return;
      if (data)
        console.log(fmt("info", module, `OK ${msg}`), sanitizeValue(data));
      else console.log(fmt("info", module, `OK ${msg}`));
    },
  };
}

export type ModuleLogger = ReturnType<typeof createModuleLogger>;

export function getLogger(module: string): ModuleLogger {
  return createModuleLogger(module);
}

export const logger = {
  log: (...args: unknown[]) => {
    if (isDevelopment) console.log(...args.map((v) => sanitizeValue(v)));
  },
  info: (...args: unknown[]) => {
    if (isDevelopment) console.info(...args.map((v) => sanitizeValue(v)));
  },
  warn: (...args: unknown[]) => {
    console.warn(...args.map((v) => sanitizeValue(v)));
  },
  error: (...args: unknown[]) => {
    console.error(...args.map((v) => sanitizeValue(v)));
  },
  debug: (...args: unknown[]) => {
    if (isDevelopment) console.debug(...args.map((v) => sanitizeValue(v)));
  },
};

export const isDev = isDevelopment;
