function envEnabled(value: string | undefined, fallback = true): boolean {
  if (value == null) return fallback;
  return !['0', 'false', 'off', 'no'].includes(value.trim().toLowerCase());
}

export function getFeatureFlags(
  env: Partial<Record<string, string | undefined>>,
) {
  return {
    institutionMobileRequestSendEnabled: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_REQUEST_SEND_ENABLED,
    ),
    institutionMobileRealtimeEnabled: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_REALTIME_ENABLED,
    ),
    institutionMobileFieldsetTerrainRequired: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_FIELDSET_TERRAIN_REQUIRED,
    ),
    institutionMobileRoleGuardsEnabled: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_ROLE_GUARDS_ENABLED,
    ),
    institutionMobileRequestFiltersEnabled: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_REQUEST_FILTERS_ENABLED,
    ),
    institutionMobileBillingIntentEnabled: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_BILLING_INTENT_ENABLED,
    ),
    institutionMobilePatientDetailEnabled: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_PATIENT_DETAIL_ENABLED,
    ),
    institutionMobileSettingsNotificationsEnabled: envEnabled(
      env.EXPO_PUBLIC_INSTITUTION_MOBILE_SETTINGS_NOTIFICATIONS_ENABLED,
    ),
  };
}

export const featureFlags = getFeatureFlags(process.env);
