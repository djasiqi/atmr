export const STORAGE_KEYS = {
  DRIVER_PROFILE: "driver.profile.cache",
  NOTIFICATIONS: "notifications.cache",
  SESSION_RUNTIME: "session.runtime",
  STORAGE_MIGRATION_VERSION: "storage.migration.version",
} as const;

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS];

