import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

// --- Mocks pour les scénarios SQLite natifs (chemin réel, pas mémoire Jest) ---

let mockPlatformOS: "android" | "ios" | "web" = "android";
jest.mock("react-native", () => ({
  get Platform() {
    return { OS: mockPlatformOS };
  },
}));

jest.mock("expo-modules-core", () => ({
  requireOptionalNativeModule: (name: string) => (name === "ExpoSQLite" ? {} : null),
}));

const mockOpenDatabaseAsync = jest.fn<(name: string) => Promise<unknown>>();
const mockDeleteDatabaseAsync = jest.fn<(name: string) => Promise<void>>();
jest.mock("expo-sqlite", () => ({
  openDatabaseAsync: (name: string) => mockOpenDatabaseAsync(name),
  deleteDatabaseAsync: (name: string) => mockDeleteDatabaseAsync(name),
}));

// Import après les jest.mock ci-dessus (hoistés par Jest de toute façon).
import { trackingQueueStore } from "./trackingQueueStore";

type FakeRow = Record<string, unknown>;

/** Fabrique une fausse connexion SQLite conforme au sous-ensemble utilisé par le store. */
function createFakeDb(overrides: Partial<{
  runAsyncImpl: (sql: string, ...params: unknown[]) => Promise<unknown>;
  getFirstAsyncImpl: (sql: string, ...params: unknown[]) => Promise<unknown>;
  withExclusiveTransactionAsync: boolean;
}> = {}) {
  const meta = new Map<string, string>();
  const withExclusive = overrides.withExclusiveTransactionAsync ?? true;

  const getFirstAsync = jest.fn(async (sql: string, ...params: unknown[]): Promise<unknown> => {
    if (overrides.getFirstAsyncImpl) return overrides.getFirstAsyncImpl(sql, ...params);
    if (sql.includes("quick_check")) return { quick_check: "ok" };
    if (sql === "SELECT 1") return { "1": 1 };
    if (sql.includes("migration_completed")) return meta.get("migration_completed") ? { value: "1" } : null;
    if (sql.includes("COUNT(*)")) return { c: 0 };
    return null;
  });

  const runAsync = jest.fn(async (sql: string, ...params: unknown[]): Promise<unknown> => {
    if (overrides.runAsyncImpl) return overrides.runAsyncImpl(sql, ...params);
    if (sql.includes("migration_completed")) {
      meta.set("migration_completed", "1");
    }
    return undefined;
  });

  const execAsync = jest.fn(async (_sql: string): Promise<void> => undefined);

  const withTransactionAsync = jest.fn(async (fn: () => Promise<void>): Promise<void> => {
    await fn();
  });

  const db: Record<string, unknown> = {
    execAsync,
    runAsync,
    getFirstAsync,
    getAllAsync: jest.fn(async (): Promise<FakeRow[]> => []),
    withTransactionAsync,
  };

  if (withExclusive) {
    db.withExclusiveTransactionAsync = jest.fn(async (fn: (txn: unknown) => Promise<void>): Promise<void> => {
      await fn(db);
    });
  }

  return db;
}

describe("trackingQueueStore (Annexe A.4)", () => {
  beforeEach(() => {
    trackingQueueStore._resetMemoryForTests();
  });

  it("conserve ingested_non_persisted après upsert (pas d'expiration silencieuse)", async () => {
    const queuedAt = Date.now() - 86_400_000 * 2;
    await trackingQueueStore.upsert({
      locationEventId: "e1",
      trackingSessionId: "s1",
      sessionGeneration: 1,
      sequenceId: 1,
      payloadJson: "{}",
      state: "ingested_non_persisted",
      queuedAt,
      lastAttemptAt: null,
      retryCount: 0,
      deliveryState: "backend_acked",
      missionId: null,
      locationMode: "mission_live",
      batchId: "b1",
      positionId: "p1",
      appState: "active",
      lastError: null,
      ackedAt: Date.now(),
    });
    const active = await trackingQueueStore.listActive();
    expect(active.some((r) => r.locationEventId === "e1")).toBe(true);
    expect(active.find((r) => r.locationEventId === "e1")?.state).toBe(
      "ingested_non_persisted"
    );
    await trackingQueueStore.upsert({
      ...(active.find((r) => r.locationEventId === "e1")!),
      queuedAt: queuedAt - 86_400_000,
    });
    const again = await trackingQueueStore.listActive();
    expect(again.some((r) => r.locationEventId === "e1")).toBe(true);
  });

  it("expose durable_unavailable après force test", () => {
    trackingQueueStore._forceDurableUnavailableForTests();
    expect(trackingQueueStore.isDurableUnavailable()).toBe(true);
    expect(trackingQueueStore.isDurableBackendAvailable()).toBe(false);
  });

  it("importLegacyOnce conserve les lignes déjà importées (mémoire Jest)", async () => {
    await trackingQueueStore.importLegacyOnce([
      {
        locationEventId: "m1",
        trackingSessionId: "s1",
        sessionGeneration: 1,
        sequenceId: 1,
        payloadJson: "{}",
        state: "non_ingested",
        queuedAt: 1,
        lastAttemptAt: null,
        retryCount: 0,
        deliveryState: "queued",
        missionId: null,
        locationMode: "mission_live",
        batchId: "b",
        positionId: "p",
        appState: "active",
        lastError: null,
        ackedAt: null,
      },
    ]);
    const active = await trackingQueueStore.listActive();
    expect(active.some((r) => r.locationEventId === "m1")).toBe(true);
  });

  it("reste utilisable en mémoire si le backend durable est forcé KO puis réinit tests", async () => {
    trackingQueueStore._resetMemoryForTests();
    await trackingQueueStore.upsert({
      locationEventId: "e2",
      trackingSessionId: "s1",
      sessionGeneration: 1,
      sequenceId: 2,
      payloadJson: "{}",
      state: "non_ingested",
      queuedAt: Date.now(),
      lastAttemptAt: null,
      retryCount: 0,
      deliveryState: "queued",
      missionId: null,
      locationMode: "mission_live",
      batchId: "b",
      positionId: "p",
      appState: "active",
      lastError: null,
      ackedAt: null,
    });
    const active = await trackingQueueStore.listActive();
    expect(active.some((r) => r.locationEventId === "e2")).toBe(true);
  });
});

describe("trackingQueueStore — backend SQLite natif mocké", () => {
  beforeEach(() => {
    trackingQueueStore._resetMemoryForTests();
    trackingQueueStore._setForceNativeSqliteForTests(true);
    mockPlatformOS = "android";
    mockOpenDatabaseAsync.mockReset();
    mockDeleteDatabaseAsync.mockReset();
  });

  afterEach(() => {
    trackingQueueStore._setForceNativeSqliteForTests(false);
  });

  function sampleRow(id: string): Parameters<typeof trackingQueueStore.upsert>[0] {
    return {
      locationEventId: id,
      trackingSessionId: "s1",
      sessionGeneration: 1,
      sequenceId: 1,
      payloadJson: "{}",
      state: "non_ingested",
      queuedAt: Date.now(),
      lastAttemptAt: null,
      retryCount: 0,
      deliveryState: "queued",
      missionId: null,
      locationMode: "mission_live",
      batchId: "b",
      positionId: "p",
      appState: "active",
      lastError: null,
      ackedAt: null,
    };
  }

  it("20 init() concurrents ne déclenchent qu'un seul openDatabaseAsync", async () => {
    const db = createFakeDb();
    mockOpenDatabaseAsync.mockResolvedValue(db);

    await Promise.all(Array.from({ length: 20 }, () => trackingQueueStore.init()));

    expect(mockOpenDatabaseAsync).toHaveBeenCalledTimes(1);
    expect(trackingQueueStore.isDurableBackendAvailable()).toBe(true);
  });

  it("importLegacyOnce ne fait pas de deadlock (transaction exclusive mockée)", async () => {
    const db = createFakeDb();
    mockOpenDatabaseAsync.mockResolvedValue(db);

    const rows = [sampleRow("l1"), sampleRow("l2")];
    // On force un COUNT après insertion cohérent avec le nombre de lignes importées.
    (db.getFirstAsync as jest.Mock).mockImplementation(async (sql: string) => {
      if (sql.includes("quick_check")) return { quick_check: "ok" };
      if (sql.includes("migration_completed")) return null;
      if (sql.includes("COUNT(*)")) return { c: rows.length };
      return null;
    });

    const ok = await trackingQueueStore.importLegacyOnce(rows);

    expect(ok).toBe(true);
    expect(db.withExclusiveTransactionAsync).toHaveBeenCalledTimes(1);
    expect(db.withTransactionAsync).not.toHaveBeenCalled();
  });

  it("une NPE sur prepareAsync déclenche une réouverture unique puis réussit", async () => {
    const npeError = new Error(
      "Call to function 'NativeDatabase.prepareAsync' has been rejected -> Caused by: java.lang.NullPointerException"
    );
    let openCount = 0;
    let firstDbUsed = false;

    mockOpenDatabaseAsync.mockImplementation(async () => {
      openCount += 1;
      if (openCount === 1) {
        const db1 = createFakeDb({
          runAsyncImpl: async () => {
            firstDbUsed = true;
            throw npeError;
          },
        });
        return db1;
      }
      return createFakeDb();
    });

    await trackingQueueStore.upsert(sampleRow("npe1"));

    expect(firstDbUsed).toBe(true);
    expect(openCount).toBe(2);
    expect(trackingQueueStore.isDurableUnavailable()).toBe(false);
    expect(trackingQueueStore.isDurableBackendAvailable()).toBe(true);
  });

  it("une seconde NPE après réouverture bascule en fail-closed", async () => {
    const npeError = new Error(
      "Call to function 'NativeDatabase.prepareAsync' has been rejected -> Caused by: java.lang.NullPointerException"
    );
    mockOpenDatabaseAsync.mockImplementation(async () =>
      createFakeDb({
        runAsyncImpl: async () => {
          throw npeError;
        },
      })
    );

    await expect(trackingQueueStore.upsert(sampleRow("npe2"))).rejects.toThrow(
      "durable_unavailable"
    );

    expect(trackingQueueStore.isDurableUnavailable()).toBe(true);
    expect(trackingQueueStore.isDurableBackendAvailable()).toBe(false);
    expect(mockDeleteDatabaseAsync).not.toHaveBeenCalled();
  });

  it("n'appelle jamais deleteDatabaseAsync, même en cas d'échec durable", async () => {
    mockOpenDatabaseAsync.mockRejectedValue(new Error("boom"));

    await expect(trackingQueueStore.upsert(sampleRow("x1"))).rejects.toThrow(
      "durable_unavailable"
    );

    expect(mockDeleteDatabaseAsync).not.toHaveBeenCalled();
  });

  it("quick_check ne tourne qu'à l'ouverture à froid, pas sur un healthcheck à chaud", async () => {
    const db = createFakeDb();
    mockOpenDatabaseAsync.mockResolvedValue(db);

    const first = await trackingQueueStore.initAndHealthcheckHeadless();
    const second = await trackingQueueStore.initAndHealthcheckHeadless();

    expect(first).toEqual({ durable: true, schemaReady: true, recovered: true });
    expect(second).toEqual({ durable: true, schemaReady: true, recovered: false });

    const quickCheckCalls = (db.getFirstAsync as jest.Mock).mock.calls.filter(([sql]) =>
      String(sql).includes("quick_check")
    );
    expect(quickCheckCalls).toHaveLength(1);

    const selectOneCalls = (db.getFirstAsync as jest.Mock).mock.calls.filter(
      ([sql]) => sql === "SELECT 1"
    );
    expect(selectOneCalls.length).toBeGreaterThanOrEqual(2);
  });

  it("initAndHealthcheckHeadless retourne durable/schemaReady cohérents", async () => {
    const db = createFakeDb();
    mockOpenDatabaseAsync.mockResolvedValue(db);

    const health = await trackingQueueStore.initAndHealthcheckHeadless();

    expect(health.durable).toBe(true);
    expect(health.schemaReady).toBe(true);
    expect(health.recovered).toBe(true);
  });
});
