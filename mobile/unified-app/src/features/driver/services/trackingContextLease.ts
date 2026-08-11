/**
 * Autorité opérationnelle « GPS autorisé à émettre maintenant ».
 * Persistée et lisible par TaskManager — indépendante de activeContextIdForApi (mémoire Axios).
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";

export const TRACKING_CONTEXT_LEASE_STORAGE_KEY = "@driver:tracking_context_lease_v1";

export type TrackingContextLeaseDriverActive = {
  state: "driver_active";
  contextId: `driver:${string}`;
  driverId: number;
  sessionGenerationId: number;
  trackingGenerationId: string;
  trackingIdentityId: string;
  updatedAt: number;
};

export type TrackingContextLeaseSwitching = {
  state: "switching";
  /** true si le switch part d'un contexte chauffeur (capture locale encore OK). */
  fromDriver: boolean;
  updatedAt: number;
  /** Snapshot pour restore si le switch échoue. */
  previousDriverActive?: Omit<TrackingContextLeaseDriverActive, "state" | "updatedAt">;
};

export type TrackingContextLeaseInactive = {
  state: "inactive";
  updatedAt: number;
};

export type TrackingContextLease =
  | TrackingContextLeaseDriverActive
  | TrackingContextLeaseSwitching
  | TrackingContextLeaseInactive;

const inMemoryStorage = new Map<string, string>();
let memoryLease: TrackingContextLease | null = null;

async function readStorage(key: string): Promise<string | null> {
  const storage = AsyncStorage as unknown as {
    getItem?: (input: string) => Promise<string | null>;
  };
  if (typeof storage?.getItem === "function") {
    return storage.getItem(key);
  }
  return inMemoryStorage.get(key) ?? null;
}

async function writeStorage(key: string, value: string): Promise<void> {
  const storage = AsyncStorage as unknown as {
    setItem?: (k: string, v: string) => Promise<void>;
  };
  if (typeof storage?.setItem === "function") {
    await storage.setItem(key, value);
    return;
  }
  inMemoryStorage.set(key, value);
}

function isDriverContextId(value: string): value is `driver:${string}` {
  return /^driver:\d+$/.test(value);
}

function parseLease(raw: string | null): TrackingContextLease | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<TrackingContextLease>;
    if (!parsed || typeof parsed !== "object" || typeof parsed.state !== "string") {
      return null;
    }
    const updatedAt =
      typeof parsed.updatedAt === "number" && Number.isFinite(parsed.updatedAt)
        ? parsed.updatedAt
        : Date.now();
    if (parsed.state === "inactive") {
      return { state: "inactive", updatedAt };
    }
    if (parsed.state === "switching") {
      const switching = parsed as Partial<TrackingContextLeaseSwitching>;
      return {
        state: "switching",
        fromDriver: switching.fromDriver === true,
        updatedAt,
        previousDriverActive: switching.previousDriverActive,
      };
    }
    if (parsed.state === "driver_active") {
      const active = parsed as Partial<TrackingContextLeaseDriverActive>;
      if (
        typeof active.contextId === "string" &&
        isDriverContextId(active.contextId) &&
        typeof active.driverId === "number" &&
        Number.isFinite(active.driverId) &&
        typeof active.sessionGenerationId === "number" &&
        typeof active.trackingGenerationId === "string" &&
        typeof active.trackingIdentityId === "string"
      ) {
        return {
          state: "driver_active",
          contextId: active.contextId,
          driverId: active.driverId,
          sessionGenerationId: active.sessionGenerationId,
          trackingGenerationId: active.trackingGenerationId,
          trackingIdentityId: active.trackingIdentityId,
          updatedAt,
        };
      }
    }
    return null;
  } catch {
    return null;
  }
}

export async function readTrackingContextLease(): Promise<TrackingContextLease | null> {
  if (memoryLease) return memoryLease;
  const parsed = parseLease(await readStorage(TRACKING_CONTEXT_LEASE_STORAGE_KEY));
  memoryLease = parsed;
  return parsed;
}

async function persistLease(lease: TrackingContextLease): Promise<void> {
  memoryLease = lease;
  await writeStorage(TRACKING_CONTEXT_LEASE_STORAGE_KEY, JSON.stringify(lease));
  emitDriverTelemetry("tracking.context.lease.updated", {
    source: "driver.services.trackingContextLease",
    lease_state: lease.state,
    from_driver: lease.state === "switching" ? lease.fromDriver : undefined,
  });
}

export async function setTrackingContextLeaseDriverActive(params: {
  contextId: `driver:${string}` | string;
  driverId: number;
  sessionGenerationId: number;
  trackingGenerationId: string;
  trackingIdentityId: string;
}): Promise<TrackingContextLeaseDriverActive> {
  if (!isDriverContextId(params.contextId)) {
    throw new Error(`Invalid driver contextId for lease: ${params.contextId}`);
  }
  const lease: TrackingContextLeaseDriverActive = {
    state: "driver_active",
    contextId: params.contextId,
    driverId: params.driverId,
    sessionGenerationId: params.sessionGenerationId,
    trackingGenerationId: params.trackingGenerationId,
    trackingIdentityId: params.trackingIdentityId,
    updatedAt: Date.now(),
  };
  await persistLease(lease);
  return lease;
}

export async function setTrackingContextLeaseSwitching(opts?: {
  fromDriver?: boolean;
  previousDriverActive?: TrackingContextLeaseDriverActive | null;
}): Promise<TrackingContextLeaseSwitching> {
  const previous = opts?.previousDriverActive;
  const lease: TrackingContextLeaseSwitching = {
    state: "switching",
    fromDriver: opts?.fromDriver === true,
    updatedAt: Date.now(),
    previousDriverActive: previous
      ? {
          contextId: previous.contextId,
          driverId: previous.driverId,
          sessionGenerationId: previous.sessionGenerationId,
          trackingGenerationId: previous.trackingGenerationId,
          trackingIdentityId: previous.trackingIdentityId,
        }
      : undefined,
  };
  await persistLease(lease);
  return lease;
}

export async function setTrackingContextLeaseInactive(): Promise<TrackingContextLeaseInactive> {
  const lease: TrackingContextLeaseInactive = {
    state: "inactive",
    updatedAt: Date.now(),
  };
  await persistLease(lease);
  return lease;
}

export async function restoreTrackingContextLeaseDriverActiveFromSwitching(): Promise<
  TrackingContextLeaseDriverActive | null
> {
  const current = await readTrackingContextLease();
  if (current?.state !== "switching" || !current.previousDriverActive) {
    return null;
  }
  return setTrackingContextLeaseDriverActive(current.previousDriverActive);
}

/** Capture locale SQLite autorisée. */
export function leaseAllowsCapture(lease: TrackingContextLease | null): boolean {
  if (!lease) return false;
  if (lease.state === "driver_active") return true;
  if (lease.state === "switching" && lease.fromDriver) return true;
  return false;
}

/** Transport réseau / flush /driver/me/* autorisé. */
export function leaseAllowsTransport(lease: TrackingContextLease | null): boolean {
  return lease?.state === "driver_active";
}

/**
 * Réconciliation crash-safe au bootstrap.
 * Ne promeut jamais switching → driver_active sans confirmation du contexte bootstrap.
 */
export async function reconcileTrackingContextLeaseFromBootstrap(params: {
  activeContextId: string | null;
  activeContextType: string | null;
  isAuthenticated: boolean;
}): Promise<TrackingContextLease> {
  const { activeContextId, activeContextType, isAuthenticated } = params;
  const isDriver =
    isAuthenticated &&
    activeContextType === "driver" &&
    typeof activeContextId === "string" &&
    isDriverContextId(activeContextId);

  if (!isDriver) {
    return setTrackingContextLeaseInactive();
  }

  const existing = await readTrackingContextLease();
  if (
    existing?.state === "driver_active" &&
    existing.contextId === activeContextId
  ) {
    return existing;
  }

  // Crash pendant switching : si le serveur confirme encore driver, restaurer le snapshot.
  if (existing?.state === "switching") {
    if (
      existing.previousDriverActive &&
      existing.previousDriverActive.contextId === activeContextId
    ) {
      return setTrackingContextLeaseDriverActive(existing.previousDriverActive);
    }
    // Sinon fail-closed jusqu'à ce que le bridge reconstruise driver_active.
    return setTrackingContextLeaseInactive();
  }
  if (existing?.state === "driver_active" && existing.contextId !== activeContextId) {
    return setTrackingContextLeaseInactive();
  }
  // inactive / absente : le bridge posera driver_active au start GPS.
  return existing ?? (await setTrackingContextLeaseInactive());
}

/** Reset mémoire (tests). */
export function __resetTrackingContextLeaseForTests(): void {
  memoryLease = null;
  inMemoryStorage.clear();
}
