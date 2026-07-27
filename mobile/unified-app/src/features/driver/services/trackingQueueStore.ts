/**
 * Stockage transactionnel de la file GPS (Annexe A.4).
 * SQLite = source de vérité native ; mémoire uniquement Jest/web (explicite).
 *
 * Architecture (P0) :
 * - Une seule ouverture SQLite en vol à la fois (`sqliteOpenPromise` + `getOrOpenDatabase`).
 * - Toute méthode publique acquiert le mutex une seule fois via `runSerialized` (API non réentrante).
 * - Les primitives internes (`*WithExecutor`) prennent un `executor` et n'acquièrent JAMAIS le
 *   mutex ni n'appellent une méthode publique — elles s'utilisent depuis l'intérieur d'une
 *   transaction déjà ouverte pour éviter tout deadlock (ex. importLegacyOnce, markState, deleteIds).
 * - Une NullPointerException native (`NativeDatabase.prepareAsync`) déclenche une récupération
 *   à l'intérieur du mutex : invalidation du handle, réouverture unique, sondage `SELECT 1`,
 *   puis un seul essai de ré-exécution avant fail-closed. On ne supprime/ne wipe jamais la base.
 */

export type TrackingQueueRowState =
  | "non_ingested"
  | "ingested_non_persisted"
  | "persisted"
  | "rejected"
  | "tombstone";

export type TrackingQueueRow = {
  locationEventId: string;
  trackingSessionId: string;
  sessionGeneration: number | null;
  sequenceId: number;
  payloadJson: string;
  state: TrackingQueueRowState;
  queuedAt: number;
  lastAttemptAt: number | null;
  retryCount: number;
  deliveryState: string;
  missionId: number | null;
  locationMode: string;
  batchId: string;
  positionId: string;
  appState: string;
  lastError: string | null;
  ackedAt: number | null;
};

export type LocalGapRecord = {
  trackingSessionId: string;
  sequenceFrom: number;
  sequenceTo: number;
  reason: string;
  createdAt: number;
};

export type ContiguousCursor = {
  trackingSessionId: string;
  contiguousIngestedThrough: number;
  contiguousPersistedThrough: number;
};

/** Sous-ensemble minimal de l'API SQLite utilisé par les primitives (db réelle ou txn). */
export type SqliteExecutor = {
  execAsync: (sql: string) => Promise<void>;
  runAsync: (sql: string, ...params: unknown[]) => Promise<unknown>;
  getAllAsync: <T>(sql: string, ...params: unknown[]) => Promise<T[]>;
  getFirstAsync: <T>(sql: string, ...params: unknown[]) => Promise<T | null>;
};

/** Handle de connexion complet (peut en plus démarrer des transactions). */
type SqliteDatabaseHandle = SqliteExecutor & {
  withTransactionAsync: (fn: () => Promise<void>) => Promise<void>;
  /** Verrou exclusif natif (préféré) — absent sur certains mocks Jest, d'où le fallback. */
  withExclusiveTransactionAsync?: (
    fn: (txn: SqliteExecutor) => Promise<void>
  ) => Promise<void>;
};

/** Résultat du sondage de santé headless (tâche background, avant d'enfiler des points). */
export type TrackingQueueHealth = {
  durable: boolean;
  schemaReady: boolean;
  recovered: boolean;
};

type TrackingQueueStateExtras = Partial<
  Pick<TrackingQueueRow, "ackedAt" | "deliveryState" | "lastError" | "retryCount" | "lastAttemptAt">
>;

type MemoryDb = {
  rows: Map<string, TrackingQueueRow>;
  gaps: LocalGapRecord[];
  cursors: Map<string, ContiguousCursor>;
  quarantineIdentity: string | null;
  migrationCompleted: boolean;
};

const memory: MemoryDb = {
  rows: new Map(),
  gaps: [],
  cursors: new Map(),
  quarantineIdentity: null,
  migrationCompleted: false,
};

let useMemory = true;
let durableUnavailable = false;
let sqliteDb: SqliteDatabaseHandle | null = null;
/** Singleton d'ouverture : garantit un seul `openDatabaseAsync` en vol, même sous concurrence. */
let sqliteOpenPromise: Promise<SqliteDatabaseHandle> | null = null;
/** Schéma créé + vérifié pour le handle courant. */
let schemaReady = false;
/** true tant qu'un `PRAGMA quick_check` n'a pas encore été fait pour le handle courant
 * (ouverture à froid ou juste après une réouverture suite à NPE). */
let needsDeepCheck = true;
/** true uniquement après une récupération NPE réussie (consommé au prochain healthcheck). */
let lastNpeRecovered = false;
/** Uniquement pour les tests : force le chemin natif même sous Jest (bypass `allowMemoryBackend`). */
let forceNativeForTests = false;

/** Mutex simple par chaînage de promesses — sérialise les appels publics (non réentrant). */
let mutexChain: Promise<unknown> = Promise.resolve();

function runSerialized<T>(fn: () => Promise<T>): Promise<T> {
  const result = mutexChain.then(() => fn());
  mutexChain = result.then(
    () => undefined,
    () => undefined
  );
  return result;
}

function isNativePlatform(): boolean {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Platform = require("react-native").Platform as { OS?: string };
    return Platform.OS === "ios" || Platform.OS === "android";
  } catch {
    return false;
  }
}

function allowMemoryBackend(): boolean {
  if (forceNativeForTests) return false;
  if (typeof jest !== "undefined") return true;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Platform = require("react-native").Platform as { OS?: string };
    return Platform.OS === "web";
  } catch {
    return true;
  }
}

/**
 * Binaire store sans ExpoSQLite (OTA JS seul) : ne jamais `import("expo-sqlite")`
 * sinon crash fatal « Cannot find native module 'ExpoSQLite' » au switch chauffeur.
 */
function isExpoSqliteNativeAvailable(): boolean {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { requireOptionalNativeModule } = require("expo-modules-core") as {
      requireOptionalNativeModule?: (name: string) => unknown;
    };
    if (typeof requireOptionalNativeModule !== "function") return false;
    return requireOptionalNativeModule("ExpoSQLite") != null;
  } catch {
    return false;
  }
}

function emitCriticalTelemetry(event: string, detail?: Record<string, unknown>): void {
  try {
    console.error(`[trackingQueueStore] CRITICAL ${event}`, detail ?? {});
  } catch {
    // ignore
  }
}

/**
 * Détecte la NPE native `NativeDatabase.prepareAsync` observée sur certains devices Android
 * (handle corrompu après kill process). Seule cette signature déclenche la récupération —
 * les autres erreurs (contrainte SQL, etc.) remontent telles quelles.
 */
function isNpeError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return (
    message.includes("NativeDatabase") &&
    message.includes("prepareAsync") &&
    (message.includes("NullPointerException") || message.includes("NullPointer"))
  );
}

/** Ouverture à froid : crée le schéma puis vérifie l'intégrité via `quick_check` (rapide, non bloquant). */
async function openAndInitSchema(): Promise<SqliteDatabaseHandle> {
  if (!isExpoSqliteNativeAvailable()) {
    throw new Error("expo_sqlite_native_module_missing");
  }
  // `require` paresseux (jamais d'import statique en tête de fichier) : le module natif n'est
  // touché qu'ici, après le garde `isExpoSqliteNativeAvailable()` ci-dessus.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ExpoSqlite = require("expo-sqlite") as typeof import("expo-sqlite");
  const db = (await ExpoSqlite.openDatabaseAsync(
    "driver_tracking_queue_v5.db"
  )) as unknown as SqliteDatabaseHandle;
  await db.execAsync(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS tracking_queue (
        location_event_id TEXT PRIMARY KEY NOT NULL,
        tracking_session_id TEXT NOT NULL,
        session_generation INTEGER,
        sequence_id INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        state TEXT NOT NULL,
        queued_at INTEGER NOT NULL,
        last_attempt_at INTEGER,
        retry_count INTEGER NOT NULL DEFAULT 0,
        delivery_state TEXT NOT NULL,
        mission_id INTEGER,
        location_mode TEXT NOT NULL,
        batch_id TEXT NOT NULL,
        position_id TEXT NOT NULL,
        app_state TEXT NOT NULL,
        last_error TEXT,
        acked_at INTEGER
      );
      CREATE INDEX IF NOT EXISTS ix_tq_session_seq
        ON tracking_queue(tracking_session_id, sequence_id);
      CREATE INDEX IF NOT EXISTS ix_tq_state ON tracking_queue(state);
      CREATE TABLE IF NOT EXISTS tracking_local_gaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_session_id TEXT NOT NULL,
        sequence_from INTEGER NOT NULL,
        sequence_to INTEGER NOT NULL,
        reason TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS tracking_cursors (
        tracking_session_id TEXT PRIMARY KEY NOT NULL,
        contiguous_ingested_through INTEGER NOT NULL DEFAULT 0,
        contiguous_persisted_through INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE IF NOT EXISTS tracking_quarantine_meta (
        key TEXT PRIMARY KEY NOT NULL,
        value TEXT NOT NULL
      );
    `);
  try {
    // quick_check(1) : borné à 1 erreur, bien plus rapide qu'un integrity_check complet
    // (qui peut bloquer plusieurs secondes sur une grosse base au démarrage).
    const check = await db.getFirstAsync<{ quick_check: string }>("PRAGMA quick_check(1)");
    const ok = String(check?.quick_check ?? "").toLowerCase() === "ok";
    if (!ok) {
      throw new Error(`sqlite_integrity_failed:${check?.quick_check}`);
    }
  } catch (err) {
    if (String(err).includes("sqlite_integrity_failed")) throw err;
    // PRAGMA non supporté selon runtime (ou mock de test) — ignorer.
  }
  return db;
}

/** Singleton d'ouverture — un seul `openDatabaseAsync` en vol, réutilisé par tous les appelants. */
function getOrOpenDatabase(): Promise<SqliteDatabaseHandle> {
  if (sqliteDb) return Promise.resolve(sqliteDb);
  if (!sqliteOpenPromise) {
    sqliteOpenPromise = openAndInitSchema().then(
      (db) => {
        sqliteDb = db;
        schemaReady = true;
        needsDeepCheck = false;
        return db;
      },
      (err) => {
        sqliteOpenPromise = null;
        throw err;
      }
    );
  }
  return sqliteOpenPromise;
}

type BackendMode = "memory" | "sqlite" | "unavailable";

/**
 * Détermine (et si besoin établit) le backend actif. Doit être appelé depuis l'intérieur
 * d'un `runSerialized` — ne pose jamais le verrou lui-même.
 */
async function ensureBackendMode(): Promise<BackendMode> {
  if (sqliteDb) return "sqlite";
  if (durableUnavailable && isNativePlatform()) return "unavailable";
  if (allowMemoryBackend()) {
    useMemory = true;
    return "memory";
  }
  // OTA sur binaire sans ExpoSQLite : dégradé mémoire + AsyncStorage (pas de crash).
  if (isNativePlatform() && !isExpoSqliteNativeAvailable()) {
    useMemory = true;
    durableUnavailable = false;
    sqliteDb = null;
    emitCriticalTelemetry("sqlite_native_module_missing", {
      platform: "native",
      degraded: "memory_async_storage",
    });
    return "memory";
  }
  // Natif avec module : fail-closed si ouverture KO, jamais DELETE DB.
  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      await getOrOpenDatabase();
      useMemory = false;
      durableUnavailable = false;
      return "sqlite";
    } catch (err) {
      lastError = err;
      if (String(err).includes("expo_sqlite_native_module_missing")) {
        break;
      }
    }
  }
  // Module présent mais open KO → fail-closed.
  // Module manquant (course) → déjà géré ci-dessus ; filet de sécurité mémoire.
  if (String(lastError).includes("expo_sqlite_native_module_missing")) {
    useMemory = true;
    durableUnavailable = false;
    sqliteDb = null;
    emitCriticalTelemetry("sqlite_native_module_missing", {
      platform: "native",
      degraded: "memory_async_storage",
    });
    return "memory";
  }
  durableUnavailable = true;
  useMemory = false;
  sqliteDb = null;
  schemaReady = false;
  needsDeepCheck = true;
  emitCriticalTelemetry("sqlite_open_failed", {
    error: String(lastError),
    platform: "native",
  });
  return "unavailable";
}

/**
 * Exécute une opération SQLite avec récupération NPE : invalide le handle, réouvre une seule
 * fois, sonde `SELECT 1`, puis retente l'opération une seule fois avant fail-closed.
 * Ne supprime/ne wipe JAMAIS la base (pas de `deleteDatabaseAsync`, pas de `DROP`).
 */
async function runSqliteOperation<T>(op: (db: SqliteDatabaseHandle) => Promise<T>): Promise<T> {
  const db = sqliteDb;
  if (!db) {
    throw new Error("durable_unavailable");
  }
  try {
    return await op(db);
  } catch (err) {
    if (!isNpeError(err)) {
      throw err;
    }
    emitCriticalTelemetry("sqlite_npe_detected", { error: String(err) });
    sqliteDb = null;
    sqliteOpenPromise = null;
    schemaReady = false;
    needsDeepCheck = true;
    durableUnavailable = false;
    let reopened: SqliteDatabaseHandle;
    try {
      reopened = await getOrOpenDatabase();
      await reopened.getFirstAsync("SELECT 1");
      lastNpeRecovered = true;
    } catch (reopenErr) {
      durableUnavailable = true;
      sqliteDb = null;
      emitCriticalTelemetry("sqlite_npe_recovery_failed", { error: String(reopenErr) });
      throw new Error("durable_unavailable");
    }
    try {
      return await op(reopened);
    } catch (retryErr) {
      if (isNpeError(retryErr)) {
        durableUnavailable = true;
        sqliteDb = null;
        schemaReady = false;
        needsDeepCheck = true;
        emitCriticalTelemetry("sqlite_npe_recovery_exhausted", { error: String(retryErr) });
        throw new Error("durable_unavailable");
      }
      throw retryErr;
    }
  }
}

/** Transaction exclusive (préférée) avec fallback `withTransactionAsync` pour les mocks. */
async function withExclusiveOrFallbackTransaction(
  db: SqliteDatabaseHandle,
  fn: (executor: SqliteExecutor) => Promise<void>
): Promise<void> {
  if (typeof db.withExclusiveTransactionAsync === "function") {
    await db.withExclusiveTransactionAsync(async (txn) => {
      await fn(txn);
    });
    return;
  }
  await db.withTransactionAsync(async () => {
    await fn(db);
  });
}

function rowFromSqlite(r: Record<string, unknown>): TrackingQueueRow {
  return {
    locationEventId: String(r.location_event_id),
    trackingSessionId: String(r.tracking_session_id),
    sessionGeneration:
      r.session_generation == null ? null : Number(r.session_generation),
    sequenceId: Number(r.sequence_id),
    payloadJson: String(r.payload_json),
    state: String(r.state) as TrackingQueueRowState,
    queuedAt: Number(r.queued_at),
    lastAttemptAt: r.last_attempt_at == null ? null : Number(r.last_attempt_at),
    retryCount: Number(r.retry_count ?? 0),
    deliveryState: String(r.delivery_state),
    missionId: r.mission_id == null ? null : Number(r.mission_id),
    locationMode: String(r.location_mode),
    batchId: String(r.batch_id),
    positionId: String(r.position_id),
    appState: String(r.app_state),
    lastError: r.last_error == null ? null : String(r.last_error),
    ackedAt: r.acked_at == null ? null : Number(r.acked_at),
  };
}

// --- Primitives (executor fourni, jamais de verrou, jamais d'appel à l'API publique) ---

async function upsertWithExecutor(executor: SqliteExecutor, row: TrackingQueueRow): Promise<void> {
  await executor.runAsync(
    `INSERT INTO tracking_queue (
      location_event_id, tracking_session_id, session_generation, sequence_id,
      payload_json, state, queued_at, last_attempt_at, retry_count, delivery_state,
      mission_id, location_mode, batch_id, position_id, app_state, last_error, acked_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(location_event_id) DO UPDATE SET
      tracking_session_id=excluded.tracking_session_id,
      session_generation=excluded.session_generation,
      sequence_id=excluded.sequence_id,
      payload_json=excluded.payload_json,
      state=excluded.state,
      queued_at=excluded.queued_at,
      last_attempt_at=excluded.last_attempt_at,
      retry_count=excluded.retry_count,
      delivery_state=excluded.delivery_state,
      mission_id=excluded.mission_id,
      location_mode=excluded.location_mode,
      batch_id=excluded.batch_id,
      position_id=excluded.position_id,
      app_state=excluded.app_state,
      last_error=excluded.last_error,
      acked_at=excluded.acked_at`,
    row.locationEventId,
    row.trackingSessionId,
    row.sessionGeneration,
    row.sequenceId,
    row.payloadJson,
    row.state,
    row.queuedAt,
    row.lastAttemptAt,
    row.retryCount,
    row.deliveryState,
    row.missionId,
    row.locationMode,
    row.batchId,
    row.positionId,
    row.appState,
    row.lastError,
    row.ackedAt
  );
}

function upsertMemory(row: TrackingQueueRow): void {
  memory.rows.set(row.locationEventId, { ...row });
}

async function markStateWithExecutor(
  executor: SqliteExecutor,
  id: string,
  state: TrackingQueueRowState,
  extras?: TrackingQueueStateExtras
): Promise<void> {
  await executor.runAsync(
    `UPDATE tracking_queue SET
      state = ?,
      delivery_state = COALESCE(?, delivery_state),
      acked_at = COALESCE(?, acked_at),
      last_error = COALESCE(?, last_error),
      retry_count = COALESCE(?, retry_count),
      last_attempt_at = COALESCE(?, last_attempt_at)
     WHERE location_event_id = ?`,
    state,
    extras?.deliveryState ?? null,
    extras?.ackedAt ?? null,
    extras?.lastError ?? null,
    extras?.retryCount ?? null,
    extras?.lastAttemptAt ?? null,
    id
  );
}

function markStateMemory(
  id: string,
  state: TrackingQueueRowState,
  extras?: TrackingQueueStateExtras
): void {
  const existing = memory.rows.get(id);
  if (!existing) return;
  memory.rows.set(id, {
    ...existing,
    state,
    deliveryState: extras?.deliveryState ?? existing.deliveryState,
    ackedAt: extras?.ackedAt ?? existing.ackedAt,
    lastError: extras?.lastError ?? existing.lastError,
    retryCount: extras?.retryCount ?? existing.retryCount,
    lastAttemptAt: extras?.lastAttemptAt ?? existing.lastAttemptAt,
  });
}

async function deleteIdWithExecutor(executor: SqliteExecutor, id: string): Promise<void> {
  await executor.runAsync("DELETE FROM tracking_queue WHERE location_event_id = ?", id);
}

function deleteIdMemory(id: string): void {
  memory.rows.delete(id);
}

async function getCursorWithExecutor(
  executor: SqliteExecutor,
  trackingSessionId: string
): Promise<ContiguousCursor> {
  const row = await executor.getFirstAsync<Record<string, unknown>>(
    "SELECT * FROM tracking_cursors WHERE tracking_session_id = ?",
    trackingSessionId
  );
  if (!row) {
    return {
      trackingSessionId,
      contiguousIngestedThrough: 0,
      contiguousPersistedThrough: 0,
    };
  }
  return {
    trackingSessionId,
    contiguousIngestedThrough: Number(row.contiguous_ingested_through ?? 0),
    contiguousPersistedThrough: Number(row.contiguous_persisted_through ?? 0),
  };
}

function getCursorMemory(trackingSessionId: string): ContiguousCursor {
  return (
    memory.cursors.get(trackingSessionId) ?? {
      trackingSessionId,
      contiguousIngestedThrough: 0,
      contiguousPersistedThrough: 0,
    }
  );
}

/**
 * Migration AsyncStorage → SQLite one-shot. Primitive complète (marqueur + comptage + transaction) :
 * appelée uniquement depuis l'intérieur d'un `runSqliteOperation` déjà sous verrou.
 */
async function performLegacyImport(
  db: SqliteDatabaseHandle,
  rows: TrackingQueueRow[]
): Promise<boolean> {
  const marker = await db.getFirstAsync<{ value: string }>(
    "SELECT value FROM tracking_quarantine_meta WHERE key = 'migration_completed'"
  );
  if (marker?.value === "1") {
    return true;
  }
  const countRow = await db.getFirstAsync<{ c: number }>(
    "SELECT COUNT(*) AS c FROM tracking_queue"
  );
  const existingCount = Number(countRow?.c ?? 0);
  if (existingCount > 0 && rows.length === 0) {
    await db.runAsync(
      `INSERT INTO tracking_quarantine_meta (key, value) VALUES ('migration_completed', '1')
       ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    );
    return true;
  }
  await withExclusiveOrFallbackTransaction(db, async (txn) => {
    for (const row of rows) {
      await upsertWithExecutor(txn, row);
    }
    const after = await txn.getFirstAsync<{ c: number }>(
      "SELECT COUNT(*) AS c FROM tracking_queue"
    );
    const afterCount = Number(after?.c ?? 0);
    if (afterCount < rows.length) {
      throw new Error("migration_count_mismatch");
    }
    await txn.runAsync(
      `INSERT INTO tracking_quarantine_meta (key, value) VALUES ('migration_completed', '1')
       ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    );
  });
  return true;
}

export const trackingQueueStore = {
  async init(): Promise<void> {
    return runSerialized(async () => {
      await ensureBackendMode();
    });
  },

  isMemoryBackend(): boolean {
    return useMemory && !durableUnavailable;
  },

  isDurableBackendAvailable(): boolean {
    return !durableUnavailable && !!sqliteDb && !useMemory;
  },

  isDurableUnavailable(): boolean {
    return durableUnavailable;
  },

  /**
   * Sondage de santé headless (avant d'enfiler/flush en tâche background), sans effet de bord
   * sur la file : ouverture à froid = schéma + `SELECT 1` + `quick_check(1)` ; handle déjà chaud
   * et sain = `SELECT 1` seul (pas de re-vérification d'intégrité à chaque tick).
   */
  async initAndHealthcheckHeadless(): Promise<TrackingQueueHealth> {
    return runSerialized(async () => {
      const recovered = lastNpeRecovered;
      lastNpeRecovered = false;
      const mode = await ensureBackendMode();
      if (mode !== "sqlite") {
        return { durable: false, schemaReady: false, recovered: false };
      }
      try {
        await runSqliteOperation((db) => db.getFirstAsync("SELECT 1"));
      } catch (err) {
        emitCriticalTelemetry("sqlite_headless_healthcheck_failed", { error: String(err) });
        return { durable: false, schemaReady: false, recovered };
      }
      return { durable: true, schemaReady, recovered };
    });
  },

  /**
   * Migration AsyncStorage → SQLite one-shot.
   * Ne wipe jamais une SQLite déjà peuplée avec une copie AsyncStorage obsolète.
   */
  async importLegacyOnce(rows: TrackingQueueRow[]): Promise<boolean> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "memory") {
        for (const row of rows) upsertMemory(row);
        memory.migrationCompleted = true;
        return true;
      }
      if (mode === "unavailable") {
        return false;
      }
      return runSqliteOperation((db) => performLegacyImport(db, rows));
    });
  },

  /** @deprecated Prefer importLegacyOnce — replaceAll wipe SQLite. */
  async replaceAll(rows: TrackingQueueRow[]): Promise<void> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "unavailable") {
        throw new Error("durable_unavailable");
      }
      if (mode === "memory") {
        memory.rows.clear();
        for (const row of rows) {
          memory.rows.set(row.locationEventId, row);
        }
        return;
      }
      await runSqliteOperation((db) => performLegacyImport(db, rows));
    });
  },

  async upsert(row: TrackingQueueRow): Promise<void> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "unavailable") {
        emitCriticalTelemetry("upsert_rejected_durable_unavailable", {
          locationEventId: row.locationEventId,
        });
        throw new Error("durable_unavailable");
      }
      if (mode === "memory") {
        upsertMemory(row);
        return;
      }
      await runSqliteOperation((db) => upsertWithExecutor(db, row));
    });
  },

  async listActive(): Promise<TrackingQueueRow[]> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        return runSqliteOperation(async (db) => {
          const rows = await db.getAllAsync<Record<string, unknown>>(
            `SELECT * FROM tracking_queue
             WHERE state NOT IN ('persisted', 'tombstone', 'rejected')
             ORDER BY COALESCE(session_generation, 0) ASC, sequence_id ASC`
          );
          return rows.map(rowFromSqlite);
        });
      }
      return [...memory.rows.values()]
        .filter((r) => r.state !== "persisted" && r.state !== "tombstone" && r.state !== "rejected")
        .sort((a, b) => {
          const ga = a.sessionGeneration ?? 0;
          const gb = b.sessionGeneration ?? 0;
          if (ga !== gb) return ga - gb;
          return a.sequenceId - b.sequenceId;
        });
    });
  },

  async markState(
    locationEventIds: string[],
    state: TrackingQueueRowState,
    extras?: TrackingQueueStateExtras
  ): Promise<void> {
    if (locationEventIds.length === 0) return;
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        await runSqliteOperation((db) =>
          withExclusiveOrFallbackTransaction(db, async (txn) => {
            for (const id of locationEventIds) {
              await markStateWithExecutor(txn, id, state, extras);
            }
          })
        );
        return;
      }
      // Backend mémoire OU durable indisponible : comportement historique conservé
      // (écriture mémoire best-effort, jamais de perte silencieuse de la file locale).
      for (const id of locationEventIds) {
        markStateMemory(id, state, extras);
      }
    });
  },

  async deleteIds(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        await runSqliteOperation((db) =>
          withExclusiveOrFallbackTransaction(db, async (txn) => {
            for (const id of ids) {
              await deleteIdWithExecutor(txn, id);
            }
          })
        );
        return;
      }
      for (const id of ids) deleteIdMemory(id);
    });
  },

  async recordGap(gap: LocalGapRecord): Promise<void> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        await runSqliteOperation((db) =>
          db.runAsync(
            `INSERT INTO tracking_local_gaps
              (tracking_session_id, sequence_from, sequence_to, reason, created_at)
             VALUES (?, ?, ?, ?, ?)`,
            gap.trackingSessionId,
            gap.sequenceFrom,
            gap.sequenceTo,
            gap.reason,
            gap.createdAt
          )
        );
        return;
      }
      memory.gaps.push(gap);
    });
  },

  async getCursor(trackingSessionId: string): Promise<ContiguousCursor> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        return runSqliteOperation((db) => getCursorWithExecutor(db, trackingSessionId));
      }
      return getCursorMemory(trackingSessionId);
    });
  },

  async setContiguousIngested(
    trackingSessionId: string,
    through: number
  ): Promise<void> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        await runSqliteOperation(async (db) => {
          const cur = await getCursorWithExecutor(db, trackingSessionId);
          const next = Math.max(cur.contiguousIngestedThrough, through);
          await db.runAsync(
            `INSERT INTO tracking_cursors
              (tracking_session_id, contiguous_ingested_through, contiguous_persisted_through)
             VALUES (?, ?, ?)
             ON CONFLICT(tracking_session_id) DO UPDATE SET
               contiguous_ingested_through = MAX(
                 tracking_cursors.contiguous_ingested_through, excluded.contiguous_ingested_through
               )`,
            trackingSessionId,
            next,
            cur.contiguousPersistedThrough
          );
        });
        return;
      }
      const cur = getCursorMemory(trackingSessionId);
      const next = Math.max(cur.contiguousIngestedThrough, through);
      memory.cursors.set(trackingSessionId, { ...cur, contiguousIngestedThrough: next });
    });
  },

  async setContiguousPersisted(
    trackingSessionId: string,
    through: number
  ): Promise<void> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        await runSqliteOperation(async (db) => {
          const cur = await getCursorWithExecutor(db, trackingSessionId);
          const next = Math.max(cur.contiguousPersistedThrough, through);
          await db.runAsync(
            `INSERT INTO tracking_cursors
              (tracking_session_id, contiguous_ingested_through, contiguous_persisted_through)
             VALUES (?, ?, ?)
             ON CONFLICT(tracking_session_id) DO UPDATE SET
               contiguous_persisted_through = MAX(
                 tracking_cursors.contiguous_persisted_through, excluded.contiguous_persisted_through
               )`,
            trackingSessionId,
            cur.contiguousIngestedThrough,
            next
          );
        });
        return;
      }
      const cur = getCursorMemory(trackingSessionId);
      const next = Math.max(cur.contiguousPersistedThrough, through);
      memory.cursors.set(trackingSessionId, { ...cur, contiguousPersistedThrough: next });
    });
  },

  /**
   * Quarantaine logout : marque l'identité ; ne purge jamais les points non ACKés.
   * Réconciliation uniquement si la même identité se reconnecte.
   */
  async quarantineForIdentity(identityKey: string): Promise<void> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        await runSqliteOperation((db) =>
          db.runAsync(
            `INSERT INTO tracking_quarantine_meta (key, value) VALUES ('identity', ?)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
            identityKey
          )
        );
        return;
      }
      memory.quarantineIdentity = identityKey;
    });
  },

  async getQuarantineIdentity(): Promise<string | null> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        const row = await runSqliteOperation((db) =>
          db.getFirstAsync<{ value: string }>(
            "SELECT value FROM tracking_quarantine_meta WHERE key = 'identity'"
          )
        );
        return row?.value ?? null;
      }
      return memory.quarantineIdentity;
    });
  },

  async clearQuarantineIfMatch(identityKey: string): Promise<boolean> {
    return runSerialized(async () => {
      const mode = await ensureBackendMode();
      if (mode === "sqlite") {
        return runSqliteOperation(async (db) => {
          const row = await db.getFirstAsync<{ value: string }>(
            "SELECT value FROM tracking_quarantine_meta WHERE key = 'identity'"
          );
          const current = row?.value ?? null;
          if (current == null) return true;
          if (current !== identityKey) return false;
          await db.runAsync("DELETE FROM tracking_quarantine_meta WHERE key = 'identity'");
          return true;
        });
      }
      const current = memory.quarantineIdentity;
      if (current == null) return true;
      if (current !== identityKey) return false;
      memory.quarantineIdentity = null;
      return true;
    });
  },

  /** Tests uniquement. */
  _resetMemoryForTests(): void {
    memory.rows.clear();
    memory.gaps = [];
    memory.cursors.clear();
    memory.quarantineIdentity = null;
    memory.migrationCompleted = false;
    useMemory = true;
    durableUnavailable = false;
    sqliteDb = null;
    sqliteOpenPromise = null;
    schemaReady = false;
    needsDeepCheck = true;
    lastNpeRecovered = false;
    mutexChain = Promise.resolve();
  },

  /** Tests : simule ouverture SQLite KO sur natif. */
  _forceDurableUnavailableForTests(): void {
    durableUnavailable = true;
    useMemory = false;
    sqliteDb = null;
    sqliteOpenPromise = null;
  },

  /** Tests uniquement : force le chemin SQLite natif même sous Jest (mock `expo-sqlite`). */
  _setForceNativeSqliteForTests(enabled: boolean): void {
    forceNativeForTests = enabled;
  },
};

declare const jest: unknown | undefined;
