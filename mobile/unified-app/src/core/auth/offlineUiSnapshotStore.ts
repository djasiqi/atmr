import AsyncStorage from "@react-native-async-storage/async-storage";
import type { AuthContext, BootstrapResponse } from "../contracts/auth";

/** Cache UI de cold start. Pas un secret : hors SecureStore. */
export const OFFLINE_UI_SNAPSHOT_KEY = "@atmr/auth/offline_ui_snapshot";

export type OfflineUiSnapshot = {
  schema_version: 1;
  session_id: string;
  active_context: AuthContext | null;
  bootstrap: BootstrapResponse | null;
};

export async function writeOfflineUiSnapshot(
  snapshot: Omit<OfflineUiSnapshot, "schema_version">
): Promise<void> {
  const payload: OfflineUiSnapshot = { schema_version: 1, ...snapshot };
  await AsyncStorage.setItem(OFFLINE_UI_SNAPSHOT_KEY, JSON.stringify(payload));
}

export async function readOfflineUiSnapshot(): Promise<OfflineUiSnapshot | null> {
  try {
    const raw = await AsyncStorage.getItem(OFFLINE_UI_SNAPSHOT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<OfflineUiSnapshot>;
    if (parsed.schema_version !== 1 || typeof parsed.session_id !== "string") return null;
    return {
      schema_version: 1,
      session_id: parsed.session_id,
      active_context: parsed.active_context ?? null,
      bootstrap: parsed.bootstrap ?? null,
    };
  } catch {
    return null;
  }
}

export async function deleteOfflineUiSnapshot(): Promise<void> {
  try {
    await AsyncStorage.removeItem(OFFLINE_UI_SNAPSHOT_KEY);
  } catch {
    /* best-effort */
  }
}
