import { baseURL } from "@/services/api";

function getApiOrigin(): string {
  try {
    return new URL(baseURL).origin;
  } catch {
    const fallback = String(baseURL || "")
      .replace(/\/api(?:\/v\d+)?(?:\/.*)?$/i, "")
      .replace(/\/+$/, "");
    return fallback;
  }
}

const API_ORIGIN = getApiOrigin();

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

