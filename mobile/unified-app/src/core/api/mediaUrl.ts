import { apiClient } from "./client";

function getApiOrigin(): string {
  const base = String(apiClient.defaults.baseURL ?? "").trim();
  if (!base) return "";
  try {
    return new URL(base).origin;
  } catch {
    return base
      .replace(/\/api(?:\/v\d+)?(?:\/.*)?$/i, "")
      .replace(/\/+$/, "");
  }
}

const API_ORIGIN = getApiOrigin();

/**
 * Préfixe les chemins relatifs de médias (`/uploads/...`) avec l’origine de l’API.
 * Les URLs absolues, `file:`, `content:`, `data:` sont laissées telles quelles.
 */
export function resolveMediaUrl(input?: string | null): string | null {
  if (!input) return null;
  const value = String(input).trim();
  if (!value) return null;

  if (
    /^https?:\/\//i.test(value) ||
    value.startsWith("data:") ||
    value.startsWith("file:") ||
    value.startsWith("content:")
  ) {
    return value;
  }

  if (value.startsWith("//")) {
    return `https:${value}`;
  }

  if (!API_ORIGIN) return value;

  if (value.startsWith("/")) {
    return `${API_ORIGIN}${value}`;
  }

  return `${API_ORIGIN}/${value.replace(/^\/+/, "")}`;
}
