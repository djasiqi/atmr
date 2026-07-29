/**
 * Stub web : jamais d'import `expo-sqlite` (évite wa-sqlite.wasm dans le bundle SSR Metro).
 * La file GPS utilise le backend mémoire sur web.
 */

export function isExpoSqliteNativeAvailable(): boolean {
  return false;
}

export async function openExpoSqliteDatabase(_name: string): Promise<never> {
  throw new Error("expo_sqlite_native_module_missing");
}
