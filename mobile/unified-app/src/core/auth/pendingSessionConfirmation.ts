/**
 * Confirmation provisional de session appareil — survit un crash post-login.
 * Aucun secret stocké : session_id + installation id uniquement (AsyncStorage).
 */
import AsyncStorage from "@react-native-async-storage/async-storage";

const PENDING_CONFIRM_KEY = "@atmr/auth/pending_session_confirmation";

export type PendingSessionConfirmation = {
  sessionId: string;
  deviceInstallationId: string;
  createdAt: string;
};

export async function readPendingSessionConfirmation(): Promise<PendingSessionConfirmation | null> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_CONFIRM_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PendingSessionConfirmation;
    if (
      typeof parsed?.sessionId !== "string" ||
      !parsed.sessionId.trim() ||
      typeof parsed?.deviceInstallationId !== "string" ||
      !parsed.deviceInstallationId.trim()
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export async function writePendingSessionConfirmation(params: {
  sessionId: string;
  deviceInstallationId: string;
}): Promise<void> {
  const payload: PendingSessionConfirmation = {
    sessionId: params.sessionId.trim(),
    deviceInstallationId: params.deviceInstallationId.trim(),
    createdAt: new Date().toISOString(),
  };
  await AsyncStorage.setItem(PENDING_CONFIRM_KEY, JSON.stringify(payload));
}

export async function clearPendingSessionConfirmation(): Promise<void> {
  await AsyncStorage.removeItem(PENDING_CONFIRM_KEY);
}

/**
 * POST confirm best-effort si une confirmation est en attente.
 * Nettoie le pending uniquement en cas de succès.
 */
export async function flushPendingSessionConfirmation(): Promise<boolean> {
  const pending = await readPendingSessionConfirmation();
  if (!pending) return false;
  try {
     
    const { confirmDeviceSession } = require("../api/client") as {
      confirmDeviceSession: (sessionId: string) => Promise<void>;
    };
    await confirmDeviceSession(pending.sessionId);
    await clearPendingSessionConfirmation();
    return true;
  } catch {
    return false;
  }
}
