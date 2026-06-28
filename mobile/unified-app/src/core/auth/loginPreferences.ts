import * as SecureStore from "../storage/secureStoreCompat";
import { getItem, removeItem, setItem } from "../storage/typedStorage";
import { STORAGE_KEYS } from "../storage/storageKeys";

export type LoginPreferences = {
  rememberMe: boolean;
  email: string | null;
  password: string | null;
};

const DEFAULT_PREFERENCES: LoginPreferences = {
  rememberMe: true,
  email: null,
  password: null,
};

function normalizeEmail(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizePassword(value: unknown): string | null {
  if (typeof value !== "string") return null;
  return value.length > 0 ? value : null;
}

async function readRememberedPassword(): Promise<string | null> {
  const value = await SecureStore.getItemAsync(STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD);
  return normalizePassword(value);
}

async function writeRememberedPassword(password: string | null): Promise<void> {
  if (!password) {
    await SecureStore.deleteItemAsync(STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD);
    return;
  }
  await SecureStore.setItemAsync(STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD, password);
}

async function clearRememberedCredentials(): Promise<void> {
  await Promise.all([
    removeItem(STORAGE_KEYS.LOGIN_PREFERENCES),
    SecureStore.deleteItemAsync(STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD),
  ]);
}

export async function readLoginPreferences(): Promise<LoginPreferences> {
  const stored = await getItem<Partial<LoginPreferences>>(STORAGE_KEYS.LOGIN_PREFERENCES);
  if (!stored || typeof stored !== "object") {
    return { ...DEFAULT_PREFERENCES };
  }
  const rememberMe = stored.rememberMe !== false;
  if (!rememberMe) {
    return { rememberMe: false, email: null, password: null };
  }
  const [email, password] = await Promise.all([
    Promise.resolve(normalizeEmail(stored.email)),
    readRememberedPassword(),
  ]);
  return { rememberMe: true, email, password };
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
  await writeRememberedPassword(normalizePassword(preferences.password));
}

export async function persistLoginRememberMe(
  email: string,
  password: string,
  rememberMe: boolean
): Promise<void> {
  await writeLoginPreferences({
    rememberMe,
    email: rememberMe ? email.trim() : null,
    password: rememberMe ? password : null,
  });
}
