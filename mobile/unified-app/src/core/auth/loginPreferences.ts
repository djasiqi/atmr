import * as SecureStore from "../storage/secureStoreCompat";
import { getItem, removeItem, setItem } from "../storage/typedStorage";
import { STORAGE_KEYS } from "../storage/storageKeys";

export type LoginPreferences = {
  rememberMe: boolean;
  email: string | null;
  /** Toujours null — conservé pour compat type / lecture legacy. */
  password: string | null;
};

const DEFAULT_PREFERENCES: LoginPreferences = {
  rememberMe: true,
  email: null,
  password: null,
};

function normalizeEmail(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  return null;
}

/** Purge défensive de tout mot de passe legacy encore en SecureStore. */
async function clearLegacyRememberedPassword(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD);
  } catch {
    /* ignore */
  }
}

async function clearRememberedCredentials(): Promise<void> {
  await Promise.all([
    removeItem(STORAGE_KEYS.LOGIN_PREFERENCES),
    clearLegacyRememberedPassword(),
  ]);
}

export async function readLoginPreferences(): Promise<LoginPreferences> {
  const stored = await getItem<Partial<LoginPreferences>>(STORAGE_KEYS.LOGIN_PREFERENCES);
  // Toujours purger un éventuel mot de passe legacy.
  await clearLegacyRememberedPassword();
  if (!stored || typeof stored !== "object") {
    return { ...DEFAULT_PREFERENCES };
  }
  const rememberMe = stored.rememberMe !== false;
  if (!rememberMe) {
    return { rememberMe: false, email: null, password: null };
  }
  return {
    rememberMe: true,
    email: normalizeEmail(stored.email),
    password: null,
  };
}

export async function writeLoginPreferences(preferences: LoginPreferences): Promise<void> {
  if (!preferences.rememberMe) {
    await clearRememberedCredentials();
    return;
  }
  await setItem<Omit<LoginPreferences, "password">>(STORAGE_KEYS.LOGIN_PREFERENCES, {
    rememberMe: true,
    email: normalizeEmail(preferences.email),
  });
  await clearLegacyRememberedPassword();
}

export async function persistLoginRememberMe(
  email: string,
  _password: string,
  rememberMe: boolean
): Promise<void> {
  await writeLoginPreferences({
    rememberMe,
    email: rememberMe ? email.trim() : null,
    password: null,
  });
}
