/**
 * Stockage transactionnel de la file GPS (Annexe A.4).
 * SQLite = source de vérité native ; mémoire uniquement Jest/web (explicite).
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
let sqliteDb: {
  execAsync: (sql: string) => Promise<void>;
  runAsync: (sql: string, ...params: unknown[]) => Promise<unknown>;
  getAllAsync: <T>(sql: string, ...params: unknown[]) => Promise<T[]>;
  getFirstAsync: <T>(sql: string, ...params: unknown[]) => Promise<T | null>;
  withTransactionAsync: (fn: () => Promise<void>) => Promise<void>;
} | null = null;

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

async function openAndInitSchema(): Promise<typeof sqliteDb> {
  if (!isExpoSqliteNativeAvailable()) {
    throw new Error("expo_sqlite_native_module_missing");
  }
  const ExpoSqlite = await import("expo-sqlite");
  const db = await ExpoSqlite.openDatabaseAsync("driver_tracking_queue_v5.db");
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
    const integrity = await db.getFirstAsync<{ integrity_check: string }>(
      "PRAGMA integrity_check"
    );
    const ok = String(integrity?.integrity_check ?? "").toLowerCase() === "ok";
    if (!ok) {
      throw new Error(`sqlite_integrity_failed:${integrity?.integrity_check}`);
    }
  } catch (err) {
    if (String(err).includes("sqlite_integrity_failed")) throw err;
    // PRAGMA non supporté selon runtime — ignorer
  }
  return db as typeof sqliteDb;
}

async function ensureSqlite(): Promise<boolean> {
  if (sqliteDb) return true;
  if (durableUnavailable && isNativePlatform()) return false;
  if (allowMemoryBackend()) {
    useMemory = true;
    return false;
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
    return false;
  }
  // Natif avec module : fail-closed si ouverture KO, jamais DELETE DB.
  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const db = await openAndInitSchema();
      sqliteDb = db;
      useMemory = false;
      durableUnavailable = false;
      return true;
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
    return false;
  }
  durableUnavailable = true;
  useMemory = false;
  sqliteDb = null;
  emitCriticalTelemetry("sqlite_open_failed", {
    error: String(lastError),
    platform: "native",
  });
  return false;
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

export const trackingQueueStore = {
  async init(): Promise<void> {
    await ensureSqlite();
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
   * Migration AsyncStorage → SQLite one-shot.
   * Ne wipe jamais une SQLite déjà peuplée avec une copie AsyncStorage obsolète.
   */
  async importLegacyOnce(rows: TrackingQueueRow[]): Promise<boolean> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      if (allowMemoryBackend()) {
        for (const row of rows) memory.rows.set(row.locationEventId, row);
        memory.migrationCompleted = true;
        return true;
      }
      return false;
    }
    const marker = await sqliteDb.getFirstAsync<{ value: string }>(
      "SELECT value FROM tracking_quarantine_meta WHERE key = 'migration_completed'"
    );
    if (marker?.value === "1") {
      return true;
    }
    const countRow = await sqliteDb.getFirstAsync<{ c: number }>(
      "SELECT COUNT(*) AS c FROM tracking_queue"
    );
    const existingCount = Number(countRow?.c ?? 0);
    if (existingCount > 0 && rows.length === 0) {
      await sqliteDb.runAsync(
        `INSERT INTO tracking_quarantine_meta (key, value) VALUES ('migration_completed', '1')
         ON CONFLICT(key) DO UPDATE SET value = excluded.value`
      );
      return true;
    }
    await sqliteDb.withTransactionAsync(async () => {
      for (const row of rows) {
        await trackingQueueStore.upsert(row);
      }
      const after = await sqliteDb!.getFirstAsync<{ c: number }>(
        "SELECT COUNT(*) AS c FROM tracking_queue"
      );
      const afterCount = Number(after?.c ?? 0);
      if (afterCount < rows.length) {
        throw new Error("migration_count_mismatch");
      }
      await sqliteDb!.runAsync(
        `INSERT INTO tracking_quarantine_meta (key, value) VALUES ('migration_completed', '1')
         ON CONFLICT(key) DO UPDATE SET value = excluded.value`
      );
    });
    return true;
  },

  /** @deprecated Prefer importLegacyOnce — replaceAll wipe SQLite. */
  async replaceAll(rows: TrackingQueueRow[]): Promise<void> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      if (durableUnavailable && isNativePlatform()) {
        throw new Error("durable_unavailable");
      }
      memory.rows.clear();
      for (const row of rows) {
        memory.rows.set(row.locationEventId, row);
      }
      return;
    }
    // Natif : ne pas DELETE+réinsert si migration déjà faite
    const marker = await sqliteDb.getFirstAsync<{ value: string }>(
      "SELECT value FROM tracking_quarantine_meta WHERE key = 'migration_completed'"
    );
    if (marker?.value === "1") {
      return;
    }
    await trackingQueueStore.importLegacyOnce(rows);
  },

  async upsert(row: TrackingQueueRow): Promise<void> {
    await ensureSqlite();
    if (durableUnavailable && isNativePlatform()) {
      emitCriticalTelemetry("upsert_rejected_durable_unavailable", {
        locationEventId: row.locationEventId,
      });
      throw new Error("durable_unavailable");
    }
    if (useMemory || !sqliteDb) {
      memory.rows.set(row.locationEventId, { ...row });
      return;
    }
    await sqliteDb.runAsync(
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
  },

  async listActive(): Promise<TrackingQueueRow[]> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      return [...memory.rows.values()]
        .filter((r) => r.state !== "persisted" && r.state !== "tombstone" && r.state !== "rejected")
        .sort((a, b) => {
          const ga = a.sessionGeneration ?? 0;
          const gb = b.sessionGeneration ?? 0;
          if (ga !== gb) return ga - gb;
          return a.sequenceId - b.sequenceId;
        });
    }
    const rows = await sqliteDb.getAllAsync<Record<string, unknown>>(
      `SELECT * FROM tracking_queue
       WHERE state NOT IN ('persisted', 'tombstone', 'rejected')
       ORDER BY COALESCE(session_generation, 0) ASC, sequence_id ASC`
    );
    return rows.map(rowFromSqlite);
  },

  async withTransaction(fn: () => Promise<void>): Promise<void> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      await fn();
      return;
    }
    await sqliteDb.withTransactionAsync(fn);
  },

  async markState(
    locationEventIds: string[],
    state: TrackingQueueRowState,
    extras?: Partial<Pick<TrackingQueueRow, "ackedAt" | "deliveryState" | "lastError" | "retryCount" | "lastAttemptAt">>
  ): Promise<void> {
    if (locationEventIds.length === 0) return;
    await ensureSqlite();
    await trackingQueueStore.withTransaction(async () => {
      for (const id of locationEventIds) {
        if (useMemory || !sqliteDb) {
          const existing = memory.rows.get(id);
          if (!existing) continue;
          memory.rows.set(id, {
            ...existing,
            state,
            deliveryState: extras?.deliveryState ?? existing.deliveryState,
            ackedAt: extras?.ackedAt ?? existing.ackedAt,
            lastError: extras?.lastError ?? existing.lastError,
            retryCount: extras?.retryCount ?? existing.retryCount,
            lastAttemptAt: extras?.lastAttemptAt ?? existing.lastAttemptAt,
          });
          if (state === "persisted" || state === "tombstone" || state === "rejected") {
            // Conservés pour audit gaps ; retirés du drain via listActive
          }
          continue;
        }
        await sqliteDb!.runAsync(
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
    });
  },

  async deleteIds(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    await ensureSqlite();
    await trackingQueueStore.withTransaction(async () => {
      for (const id of ids) {
        if (useMemory || !sqliteDb) {
          memory.rows.delete(id);
          continue;
        }
        await sqliteDb!.runAsync(
          "DELETE FROM tracking_queue WHERE location_event_id = ?",
          id
        );
      }
    });
  },

  async recordGap(gap: LocalGapRecord): Promise<void> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      memory.gaps.push(gap);
      return;
    }
    await sqliteDb.runAsync(
      `INSERT INTO tracking_local_gaps
        (tracking_session_id, sequence_from, sequence_to, reason, created_at)
       VALUES (?, ?, ?, ?, ?)`,
      gap.trackingSessionId,
      gap.sequenceFrom,
      gap.sequenceTo,
      gap.reason,
      gap.createdAt
    );
  },

  async getCursor(trackingSessionId: string): Promise<ContiguousCursor> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      return (
        memory.cursors.get(trackingSessionId) ?? {
          trackingSessionId,
          contiguousIngestedThrough: 0,
          contiguousPersistedThrough: 0,
        }
      );
    }
    const row = await sqliteDb.getFirstAsync<Record<string, unknown>>(
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
  },

  async setContiguousIngested(
    trackingSessionId: string,
    through: number
  ): Promise<void> {
    const cur = await trackingQueueStore.getCursor(trackingSessionId);
    const next = Math.max(cur.contiguousIngestedThrough, through);
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      memory.cursors.set(trackingSessionId, {
        ...cur,
        contiguousIngestedThrough: next,
      });
      return;
    }
    await sqliteDb.runAsync(
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
  },

  async setContiguousPersisted(
    trackingSessionId: string,
    through: number
  ): Promise<void> {
    const cur = await trackingQueueStore.getCursor(trackingSessionId);
    const next = Math.max(cur.contiguousPersistedThrough, through);
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      memory.cursors.set(trackingSessionId, {
        ...cur,
        contiguousPersistedThrough: next,
      });
      return;
    }
    await sqliteDb.runAsync(
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
  },

  /**
   * Quarantaine logout : marque l'identité ; ne purge jamais les points non ACKés.
   * Réconciliation uniquement si la même identité se reconnecte.
   */
  async quarantineForIdentity(identityKey: string): Promise<void> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      memory.quarantineIdentity = identityKey;
      return;
    }
    await sqliteDb.runAsync(
      `INSERT INTO tracking_quarantine_meta (key, value) VALUES ('identity', ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
      identityKey
    );
  },

  async getQuarantineIdentity(): Promise<string | null> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      return memory.quarantineIdentity;
    }
    const row = await sqliteDb.getFirstAsync<{ value: string }>(
      "SELECT value FROM tracking_quarantine_meta WHERE key = 'identity'"
    );
    return row?.value ?? null;
  },

  async clearQuarantineIfMatch(identityKey: string): Promise<boolean> {
    const current = await trackingQueueStore.getQuarantineIdentity();
    if (current == null) return true;
    if (current !== identityKey) return false;
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      memory.quarantineIdentity = null;
      return true;
    }
    await sqliteDb.runAsync(
      "DELETE FROM tracking_quarantine_meta WHERE key = 'identity'"
    );
    return true;
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
  },

  /** Tests : simule ouverture SQLite KO sur natif. */
  _forceDurableUnavailableForTests(): void {
    durableUnavailable = true;
    useMemory = false;
    sqliteDb = null;
  },
};

declare const jest: unknown | undefined;
