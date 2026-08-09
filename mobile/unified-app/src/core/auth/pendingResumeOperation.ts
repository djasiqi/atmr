/**
 * Idempotency-Key de session-resume persistée (survit crash / timeout / redémarrage).
 * Aucun token stocké — AsyncStorage uniquement.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";

const PENDING_RESUME_KEY = "@atmr/auth/pending_resume_operation";

export type PendingResumeOperation = {
  operationId: string;
  sessionId: string;
  /** Génération recovery connue au démarrage de l'op (peut être absente). */
  sourceCredentialGeneration: number | null;
  createdAt: string;
};

export async function readPendingResumeOperation(): Promise<PendingResumeOperation | null> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_RESUME_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PendingResumeOperation;
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

export async function writePendingResumeOperation(
  op: PendingResumeOperation
): Promise<void> {
  await AsyncStorage.setItem(PENDING_RESUME_KEY, JSON.stringify(op));
}

export async function clearPendingResumeOperation(): Promise<void> {
  await AsyncStorage.removeItem(PENDING_RESUME_KEY);
}

export async function ensurePendingResumeOperation(params: {
  sessionId: string;
  sourceCredentialGeneration: number | null;
}): Promise<PendingResumeOperation> {
  const existing = await readPendingResumeOperation();
  if (
    existing &&
    existing.sessionId === params.sessionId &&
    existing.sourceCredentialGeneration === params.sourceCredentialGeneration
  ) {
    return existing;
  }
  const op: PendingResumeOperation = {
    operationId: `res-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    sessionId: params.sessionId,
    sourceCredentialGeneration: params.sourceCredentialGeneration,
    createdAt: new Date().toISOString(),
  };
  await writePendingResumeOperation(op);
  return op;
}
