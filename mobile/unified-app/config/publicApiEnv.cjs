/** Résolution URL API / Socket — aligné operations-app (legacy EXPO_PUBLIC_API_URL / SOCKET_URL). */

function trimEnv(name) {
  const v = process.env[name];
  return typeof v === "string" ? v.trim() : "";
}

function isPrivateOrLocalHost(url) {
  try {
    const host = new URL(url).hostname.toLowerCase();
    if (host === "localhost" || host === "127.0.0.1" || host === "[::1]" || host === "::1") {
      return true;
    }
    if (/^192\.168\./.test(host)) return true;
    if (/^10\./.test(host)) return true;
    const m = /^172\.(\d+)\./.exec(host);
    if (m) {
      const second = Number(m[1]);
      if (second >= 16 && second <= 31) return true;
    }
    return false;
  } catch {
    return false;
  }
}

/** unified-app : base REST avec suffixe /api/v1. Repli legacy EXPO_PUBLIC_API_URL (operations-app). */
function resolveApiBaseUrlFromEnv() {
  const direct = trimEnv("EXPO_PUBLIC_API_BASE_URL");
  if (direct) return direct;
  const legacy = trimEnv("EXPO_PUBLIC_API_URL");
  if (!legacy) return "";
  const normalized = legacy.replace(/\/$/, "");
  if (normalized.endsWith("/api/v1")) return normalized;
  return `${normalized}/api/v1`;
}

function resolveDriverSocketUrlFromEnv() {
  return trimEnv("EXPO_PUBLIC_DRIVER_SOCKET_URL") || trimEnv("EXPO_PUBLIC_SOCKET_URL");
}

function assertProdHttpsEnv(label, value) {
  if (!value) {
    throw new Error(`[app.config] Missing required ${label} for APP_VARIANT=prod EAS build`);
  }
  if (!value.startsWith("https://")) {
    throw new Error(`[app.config] ${label} must be HTTPS in production (got ${value})`);
  }
  if (isPrivateOrLocalHost(value)) {
    throw new Error(
      `[app.config] ${label} must not target localhost/LAN in production (got ${value})`
    );
  }
  return value;
}

const PROD_API_BASE_URL = "https://api.lirie.ch/api/v1";
const PROD_DRIVER_SOCKET_URL = "https://api.lirie.ch";

function isProdSafeHttpsUrl(value) {
  if (!value || typeof value !== "string") return false;
  const normalized = value.trim();
  if (!normalized.startsWith("https://")) return false;
  return !isPrivateOrLocalHost(normalized);
}

/** Build EAS prod : ignore une URL LAN/HTTP (ex. .env local injecté trop tôt) et force l'API prod. */
function resolveProdApiBaseUrlForEas(candidate) {
  if (isProdSafeHttpsUrl(candidate)) return candidate.trim();
  const legacy = trimEnv("EXPO_PUBLIC_API_URL");
  if (legacy) {
    const normalized = legacy.replace(/\/$/, "");
    const fromLegacy = normalized.endsWith("/api/v1") ? normalized : `${normalized}/api/v1`;
    if (isProdSafeHttpsUrl(fromLegacy)) return fromLegacy;
  }
  return PROD_API_BASE_URL;
}

function resolveProdDriverSocketUrlForEas(candidate) {
  if (isProdSafeHttpsUrl(candidate)) return candidate.trim();
  const legacy = trimEnv("EXPO_PUBLIC_SOCKET_URL");
  if (isProdSafeHttpsUrl(legacy)) return legacy.trim();
  return PROD_DRIVER_SOCKET_URL;
}

module.exports = {
  trimEnv,
  isPrivateOrLocalHost,
  isProdSafeHttpsUrl,
  resolveApiBaseUrlFromEnv,
  resolveDriverSocketUrlFromEnv,
  assertProdHttpsEnv,
  resolveProdApiBaseUrlForEas,
  resolveProdDriverSocketUrlForEas,
  PROD_API_BASE_URL,
  PROD_DRIVER_SOCKET_URL,
};
