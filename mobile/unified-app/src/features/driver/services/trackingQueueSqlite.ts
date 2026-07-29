/**
 * Accès Expo SQLite (iOS / Android / Jest).
 * Le variant web est `trackingQueueSqlite.web.ts` (aucun import expo-sqlite → pas de wa-sqlite.wasm).
 */

export type TrackingQueueSqliteHandle = {
  execAsync: (sql: string) => Promise<void>;
  runAsync: (sql: string, ...params: unknown[]) => Promise<unknown>;
  getAllAsync: <T>(sql: string, ...params: unknown[]) => Promise<T[]>;
  getFirstAsync: <T>(sql: string, ...params: unknown[]) => Promise<T | null>;
  withTransactionAsync: (fn: () => Promise<void>) => Promise<void>;
  withExclusiveTransactionAsync?: (
    fn: (txn: {
      execAsync: (sql: string) => Promise<void>;
      runAsync: (sql: string, ...params: unknown[]) => Promise<unknown>;
      getAllAsync: <T>(sql: string, ...params: unknown[]) => Promise<T[]>;
      getFirstAsync: <T>(sql: string, ...params: unknown[]) => Promise<T | null>;
    }) => Promise<void>
  ) => Promise<void>;
};

/**
 * Binaire store sans ExpoSQLite (OTA JS seul) : ne jamais importer expo-sqlite
 * sinon crash fatal « Cannot find native module 'ExpoSQLite' ».
 */
export function isExpoSqliteNativeAvailable(): boolean {
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

export async function openExpoSqliteDatabase(
  name: string
): Promise<TrackingQueueSqliteHandle> {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ExpoSqlite = require("expo-sqlite") as typeof import("expo-sqlite");
  return (await ExpoSqlite.openDatabaseAsync(name)) as unknown as TrackingQueueSqliteHandle;
}
