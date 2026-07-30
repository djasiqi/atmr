export const STORAGE_KEYS = {
  DRIVER_PROFILE: "driver.profile.cache",
  NOTIFICATIONS: "notifications.cache",
  SESSION_RUNTIME: "session.runtime",
  STORAGE_MIGRATION_VERSION: "storage.migration.version",
  PENDING_PUSH_TOKEN_REGISTRATION: "notifications.pending_push_token_registration",
  PENDING_COMPANY_PUSH_PRESS: "notifications.pending_company_push_press",
  DRIVER_TRACKING_ONBOARDED: "driver.tracking_onboarded",
  DRIVER_TRACKING_NEEDS_ATTENTION: "driver.tracking_needs_attention",
  DRIVER_OEM_GUIDANCE_ACKNOWLEDGED: "driver.oem_guidance_acknowledged",
  DRIVER_NOTIFICATION_DISCLOSURE_ACCEPTED: "driver.notification_disclosure_accepted",
  DRIVER_BIOMETRIC_ENABLED: "driver.biometric.enabled",
  AUTH_BIOMETRIC_ENABLED: "auth.biometric.enabled",
  LOGIN_PREFERENCES: "auth.login.preferences",
  LOGIN_REMEMBERED_PASSWORD: "auth.login.remembered_password",
} as const;

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS];

