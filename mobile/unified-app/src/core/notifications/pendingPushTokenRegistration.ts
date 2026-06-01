import { getItem, removeItem, setItem } from "../storage/typedStorage";
import { STORAGE_KEYS } from "../storage/storageKeys";

export type PushTokenProvider = "expo" | "fcm";

export type PendingPushTokenRegistration = {
  provider: PushTokenProvider;
  token: string;
  deviceId: string;
  platform: "ios" | "android";
  savedAt: number;
};

type PendingPushTokenStore = {
  items: PendingPushTokenRegistration[];
};

const DEFAULT_MAX_ATTEMPTS = 3;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Retry avec backoff exponentiel (2s, 4s, …). */
export async function registerWithRetry(
  registerFn: () => Promise<void>,
  maxAttempts = DEFAULT_MAX_ATTEMPTS
): Promise<void> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await registerFn();
      return;
    } catch (error) {
      lastError = error;
      if (attempt < maxAttempts) {
        await sleep(Math.pow(2, attempt) * 1000);
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error("push token registration failed");
}

async function readStore(): Promise<PendingPushTokenStore> {
  const raw = await getItem<PendingPushTokenStore>(STORAGE_KEYS.PENDING_PUSH_TOKEN_REGISTRATION);
  if (!raw?.items || !Array.isArray(raw.items)) {
    return { items: [] };
  }
  return raw;
}

export async function persistPendingPushTokenRegistration(
  entry: Omit<PendingPushTokenRegistration, "savedAt">
): Promise<void> {
  const store = await readStore();
  const next: PendingPushTokenRegistration = { ...entry, savedAt: Date.now() };
  const withoutProvider = store.items.filter((item) => item.provider !== entry.provider);
  await setItem(STORAGE_KEYS.PENDING_PUSH_TOKEN_REGISTRATION, {
    items: [...withoutProvider, next],
  });
}

export async function clearPendingPushTokenRegistration(provider: PushTokenProvider): Promise<void> {
  const store = await readStore();
  const items = store.items.filter((item) => item.provider !== provider);
  if (items.length === store.items.length) return;
  if (items.length === 0) {
    await removeItem(STORAGE_KEYS.PENDING_PUSH_TOKEN_REGISTRATION);
    return;
  }
  await setItem(STORAGE_KEYS.PENDING_PUSH_TOKEN_REGISTRATION, { items });
}

export async function flushPendingPushTokenRegistrations(handlers: {
  registerExpo: (input: {
    token: string;
    deviceId: string;
    platform: "ios" | "android";
  }) => Promise<void>;
  registerFcm: (input: {
    token: string;
    deviceId: string;
    platform: "ios" | "android";
  }) => Promise<void>;
}): Promise<void> {
  const store = await readStore();
  if (store.items.length === 0) return;

  for (const pending of store.items) {
    const registerFn =
      pending.provider === "expo"
        ? () =>
            handlers.registerExpo({
              token: pending.token,
              deviceId: pending.deviceId,
              platform: pending.platform,
            })
        : () =>
            handlers.registerFcm({
              token: pending.token,
              deviceId: pending.deviceId,
              platform: pending.platform,
            });
    try {
      await registerWithRetry(registerFn);
      await clearPendingPushTokenRegistration(pending.provider);
    } catch {
      // Conserver en file pour un prochain démarrage.
    }
  }
}
