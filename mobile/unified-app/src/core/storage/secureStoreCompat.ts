import * as SecureStore from "expo-secure-store";
import { Platform } from "react-native";

const WEB_STORAGE_PREFIX = "unified_secure_store:";
const memoryFallback = new Map<string, string>();

function buildWebKey(key: string): string {
  return `${WEB_STORAGE_PREFIX}${key}`;
}

function canUseLocalStorage(): boolean {
  return typeof globalThis !== "undefined" && typeof globalThis.localStorage !== "undefined";
}

function readWebFallback(key: string): string | null {
  const mem = memoryFallback.get(key);
  if (typeof mem === "string") return mem;
  if (!canUseLocalStorage()) return null;
  try {
    return globalThis.localStorage.getItem(buildWebKey(key));
  } catch {
    return null;
  }
}

function writeWebFallback(key: string, value: string): void {
  memoryFallback.set(key, value);
  if (!canUseLocalStorage()) return;
  try {
    globalThis.localStorage.setItem(buildWebKey(key), value);
  } catch {
    // no-op: best effort fallback only
  }
}

function deleteWebFallback(key: string): void {
  memoryFallback.delete(key);
  if (!canUseLocalStorage()) return;
  try {
    globalThis.localStorage.removeItem(buildWebKey(key));
  } catch {
    // no-op: best effort fallback only
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
  if (Platform.OS === "web") {
    return readWebFallback(key);
  }
  if (!hasNativeSecureStore()) {
    return readWebFallback(key);
  }
  try {
    return await SecureStore.getItemAsync(key);
  } catch {
    return readWebFallback(key);
  }
}

export async function setItemAsync(key: string, value: string): Promise<void> {
  if (Platform.OS === "web") {
    writeWebFallback(key, value);
    return;
  }
  if (!hasNativeSecureStore()) {
    writeWebFallback(key, value);
    return;
  }
  try {
    await SecureStore.setItemAsync(key, value);
  } catch {
    writeWebFallback(key, value);
  }
}

export async function deleteItemAsync(key: string): Promise<void> {
  if (Platform.OS === "web") {
    deleteWebFallback(key);
    return;
  }
  if (!hasNativeSecureStore()) {
    deleteWebFallback(key);
    return;
  }
  try {
    await SecureStore.deleteItemAsync(key);
  } catch {
    deleteWebFallback(key);
  }
}
