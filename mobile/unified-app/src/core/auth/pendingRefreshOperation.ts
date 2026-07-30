/**
 * Idempotency-Key de refresh persistée (survit crash / timeout / redémarrage).
 * Aucun token stocké — AsyncStorage uniquement.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";

const PENDING_REFRESH_KEY = "@atmr/auth/pending_refresh_operation";

export type PendingRefreshOperation = {
  operationId: string;
  sessionId: string;
  sourceRefreshGeneration: number;
  createdAt: string;
};

export async function readPendingRefreshOperation(): Promise<PendingRefreshOperation | null> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_REFRESH_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PendingRefreshOperation;
    if (
      typeof parsed?.operationId !== "string" ||
      typeof parsed?.sessionId !== "string"
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export async function writePendingRefreshOperation(
  op: PendingRefreshOperation
): Promise<void> {
  await AsyncStorage.setItem(PENDING_REFRESH_KEY, JSON.stringify(op));
}

export async function clearPendingRefreshOperation(): Promise<void> {
  await AsyncStorage.removeItem(PENDING_REFRESH_KEY);
}

export async function ensurePendingRefreshOperation(params: {
  sessionId: string;
  sourceRefreshGeneration: number;
}): Promise<PendingRefreshOperation> {
  const existing = await readPendingRefreshOperation();
  if (
    existing &&
    existing.sessionId === params.sessionId &&
    existing.sourceRefreshGeneration === params.sourceRefreshGeneration
  ) {
    return existing;
  }
  const op: PendingRefreshOperation = {
    operationId: `ref-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    sessionId: params.sessionId,
    sourceRefreshGeneration: params.sourceRefreshGeneration,
    createdAt: new Date().toISOString(),
  };
  await writePendingRefreshOperation(op);
  return op;
}
