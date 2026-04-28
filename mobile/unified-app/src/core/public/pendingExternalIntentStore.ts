import * as SecureStore from "../storage/secureStoreCompat";

const STORE_KEY = "pending_external_intent_v1";

export type PendingExternalIntentRecord = {
  intent_id: string;
  intent_type: string;
  payload: Record<string, unknown>;
  received_at: number;
};

function safeParse(raw: string | null): PendingExternalIntentRecord | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as PendingExternalIntentRecord;
    if (!parsed || typeof parsed !== "object") return null;
    if (!parsed.intent_id || !parsed.intent_type || !parsed.received_at) return null;
    return parsed;
  } catch {
    return null;
  }
}

export async function loadPendingExternalIntentRecord(): Promise<PendingExternalIntentRecord | null> {
  const raw = await SecureStore.getItemAsync(STORE_KEY);
  return safeParse(raw);
}

export async function savePendingExternalIntentRecord(
  value: PendingExternalIntentRecord
): Promise<void> {
  await SecureStore.setItemAsync(STORE_KEY, JSON.stringify(value));
}

export async function clearPendingExternalIntentRecord(): Promise<void> {
  await SecureStore.deleteItemAsync(STORE_KEY);
}
