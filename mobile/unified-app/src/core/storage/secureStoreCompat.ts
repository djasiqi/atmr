import * as SecureStore from "expo-secure-store";
import { Platform } from "react-native";

const WEB_STORAGE_PREFIX = "unified_secure_store:";
const memoryStore = new Map<string, string>();

function buildWebKey(key: string): string {
  return `${WEB_STORAGE_PREFIX}${key}`;
}

function canUseLocalStorage(): boolean {
  return typeof globalThis !== "undefined" && typeof globalThis.localStorage !== "undefined";
}

/**
 * Supprime les secrets éventuellement persistés en clair (legacy web).
 * Ne relit jamais la valeur pour la réinjecter.
 */
function purgeLegacyWebPersistence(key?: string): void {
  if (!canUseLocalStorage()) return;
  try {
    if (key) {
      globalThis.localStorage.removeItem(buildWebKey(key));
      return;
    }
    const toRemove: string[] = [];
    for (let i = 0; i < globalThis.localStorage.length; i += 1) {
      const storedKey = globalThis.localStorage.key(i);
      if (storedKey && storedKey.startsWith(WEB_STORAGE_PREFIX)) {
        toRemove.push(storedKey);
      }
    }
    toRemove.forEach((storedKey) => globalThis.localStorage.removeItem(storedKey));
  } catch {
    // no-op
  }
}

function hasNativeSecureStore(): boolean {
  return (
    typeof SecureStore?.getItemAsync === "function" &&
    typeof SecureStore?.setItemAsync === "function" &&
    typeof SecureStore?.deleteItemAsync === "function"
  );
}

export async function getItemAsync(key: string): Promise<string | null> {
  purgeLegacyWebPersistence(key);
  if (Platform.OS === "web") {
    return memoryStore.get(key) ?? null;
  }
  if (!hasNativeSecureStore()) {
    return memoryStore.get(key) ?? null;
  }
  try {
    return await SecureStore.getItemAsync(key);
  } catch {
    return memoryStore.get(key) ?? null;
  }
}

export async function setItemAsync(key: string, value: string): Promise<void> {
  purgeLegacyWebPersistence(key);
  if (Platform.OS === "web") {
    memoryStore.set(key, value);
    return;
  }
  if (!hasNativeSecureStore()) {
    memoryStore.set(key, value);
    return;
  }
  try {
    await SecureStore.setItemAsync(key, value);
    memoryStore.delete(key);
  } catch {
    // Native only : mémoire de session, jamais localStorage / AsyncStorage.
    memoryStore.set(key, value);
  }
}

export async function deleteItemAsync(key: string): Promise<void> {
  memoryStore.delete(key);
  purgeLegacyWebPersistence(key);
  if (Platform.OS === "web" || !hasNativeSecureStore()) {
    return;
  }
  try {
    await SecureStore.deleteItemAsync(key);
  } catch {
    // déjà retiré de la mémoire
  }
}

purgeLegacyWebPersistence();
