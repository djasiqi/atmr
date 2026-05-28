/**
 * Phase 2 PR B/C — gate D3.3 (fix G4)
 *
 * Consumer pour l'event `connection.authority` émis par le ws-service à chaque
 * `connect`. Le payload `{ authority, canary, version }` permet de segmenter
 * les métriques canary par chemin (backend vs ws-service) :
 *   - tag Sentry `realtime.authority`, `realtime.canary`, `realtime.ws_version`
 *   - métrique locale incrémentale `authorityObservedTotal{authority}`
 *
 * Read-only : aucune logique métier ne dépend de cet event.
 */

import * as Sentry from "@sentry/react-native";

export type AuthoritySource = "ws-service" | "backend" | "unknown";

export type AuthorityPayload = {
  authority?: unknown;
  canary?: unknown;
  version?: unknown;
};

type AuthorityMetrics = {
  authorityObservedTotal: number;
  authorityByName: Record<string, number>;
  lastAuthority: AuthoritySource;
  lastCanary: boolean | null;
  lastVersion: string | null;
  lastObservedAt: string | null;
};

const metrics: {
  total: number;
  byName: Map<string, number>;
  last: {
    authority: AuthoritySource;
    canary: boolean | null;
    version: string | null;
    observedAt: string | null;
  };
} = {
  total: 0,
  byName: new Map(),
  last: {
    authority: "unknown",
    canary: null,
    version: null,
    observedAt: null,
  },
};

function resolveAuthority(input: unknown): AuthoritySource {
  if (input === "ws-service" || input === "backend") {
    return input;
  }
  return "unknown";
}

function resolveCanary(input: unknown): boolean | null {
  if (typeof input === "boolean") return input;
  return null;
}

function resolveVersion(input: unknown): string | null {
  if (typeof input === "string" && input.trim().length > 0) return input;
  return null;
}

/**
 * Met à jour les tags Sentry + les compteurs locaux selon le payload
 * `connection.authority` reçu. Best-effort : exceptions Sentry capturées.
 */
export function observeConnectionAuthority(payload: AuthorityPayload | undefined): void {
  if (!payload || typeof payload !== "object") return;
  const authority = resolveAuthority(payload.authority);
  const canary = resolveCanary(payload.canary);
  const version = resolveVersion(payload.version);

  metrics.total += 1;
  metrics.byName.set(authority, (metrics.byName.get(authority) ?? 0) + 1);
  metrics.last = {
    authority,
    canary,
    version,
    observedAt: new Date().toISOString(),
  };

  try {
    Sentry.setTag("realtime.authority", authority);
    if (canary !== null) {
      Sentry.setTag("realtime.canary", String(canary));
    }
    if (version) {
      Sentry.setTag("realtime.ws_version", version);
    }
  } catch {
    // monitoring ne doit pas casser le bridge
  }
}

export function getConnectionAuthorityMetricsSnapshot(): AuthorityMetrics {
  const byName: Record<string, number> = {};
  metrics.byName.forEach((value, key) => {
    byName[key] = value;
  });
  return {
    authorityObservedTotal: metrics.total,
    authorityByName: byName,
    lastAuthority: metrics.last.authority,
    lastCanary: metrics.last.canary,
    lastVersion: metrics.last.version,
    lastObservedAt: metrics.last.observedAt,
  };
}

export function resetConnectionAuthorityMetricsForTests(): void {
  metrics.total = 0;
  metrics.byName.clear();
  metrics.last = {
    authority: "unknown",
    canary: null,
    version: null,
    observedAt: null,
  };
}
