export const STORAGE_KEYS = {
  DRIVER_PROFILE: "driver.profile.cache",
  NOTIFICATIONS: "notifications.cache",
  SESSION_RUNTIME: "session.runtime",
  STORAGE_MIGRATION_VERSION: "storage.migration.version",
  PENDING_PUSH_TOKEN_REGISTRATION: "notifications.pending_push_token_registration",
  DRIVER_TRACKING_ONBOARDED: "driver.tracking_onboarded",
  DRIVER_TRACKING_NEEDS_ATTENTION: "driver.tracking_needs_attention",
  DRIVER_NOTIFICATION_DISCLOSURE_ACCEPTED: "driver.notification_disclosure_accepted",
} as const;

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS];

