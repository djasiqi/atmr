import AsyncStorage from "@react-native-async-storage/async-storage";
import { runStorageMigrations } from "./storageMigrations";
import type { StorageKey } from "./storageKeys";

export function getParsedOrNull<T>(raw: string | null): T | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function getItem<T>(key: StorageKey | string): Promise<T | null> {
  const raw = await AsyncStorage.getItem(key);
  return getParsedOrNull<T>(raw);
}

export async function setItem<T>(key: StorageKey | string, value: T): Promise<void> {
  const payload = JSON.stringify(value);
  await AsyncStorage.setItem(key, payload);
}

export async function removeItem(key: StorageKey | string): Promise<void> {
  await AsyncStorage.removeItem(key);
}

export async function mergeItem<T extends Record<string, unknown>>(
  key: StorageKey | string,
  value: Partial<T>
): Promise<void> {
  const current = await getItem<T>(key);
  const merged = {
    ...(current ?? {}),
    ...value,
  } as T;
  await setItem(key, merged);
}

export async function migrateIfNeeded(): Promise<void> {
  await runStorageMigrations();
}

