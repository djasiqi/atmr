/**
 * Stockage transactionnel de la file GPS (Annexe A.4).
 * SQLite en runtime natif ; backend mémoire pour Jest / web.
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
};

const memory: MemoryDb = {
  rows: new Map(),
  gaps: [],
  cursors: new Map(),
  quarantineIdentity: null,
};

let useMemory = true;
let sqliteDb: {
  execAsync: (sql: string) => Promise<void>;
  runAsync: (sql: string, ...params: unknown[]) => Promise<unknown>;
  getAllAsync: <T>(sql: string, ...params: unknown[]) => Promise<T[]>;
  getFirstAsync: <T>(sql: string, ...params: unknown[]) => Promise<T | null>;
  withTransactionAsync: (fn: () => Promise<void>) => Promise<void>;
} | null = null;

async function ensureSqlite(): Promise<boolean> {
  if (sqliteDb) return true;
  if (typeof jest !== "undefined") {
    useMemory = true;
    return false;
  }
  try {
    // Chargement dynamique — évite le crash Jest sans native module.
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
    sqliteDb = db as typeof sqliteDb;
    useMemory = false;
    return true;
  } catch {
    useMemory = true;
    return false;
  }
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
    return useMemory;
  },

  /** Remplace toute la file (migration AsyncStorage → store). */
  async replaceAll(rows: TrackingQueueRow[]): Promise<void> {
    await ensureSqlite();
    if (useMemory || !sqliteDb) {
      memory.rows.clear();
      for (const row of rows) {
        memory.rows.set(row.locationEventId, row);
      }
      return;
    }
    await sqliteDb.withTransactionAsync(async () => {
      await sqliteDb!.runAsync("DELETE FROM tracking_queue");
      for (const row of rows) {
        await trackingQueueStore.upsert(row);
      }
    });
  },

  async upsert(row: TrackingQueueRow): Promise<void> {
    await ensureSqlite();
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
    useMemory = true;
    sqliteDb = null;
  },
};

declare const jest: unknown | undefined;
