const toBool = (value, fallback = false) => {
  if (value == null) return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  return fallback;
};

const envMap = {
  FF_PRICING_WIZARD_V1: process.env.REACT_APP_FF_PRICING_WIZARD_V1,
  FF_ADMIN_ZONESETS_READONLY: process.env.REACT_APP_FF_ADMIN_ZONESETS_READONLY,
};

export const isFeatureEnabled = (flagKey, fallback = true) => {
  try {
    const rawOverrides = localStorage.getItem('featureFlags');
    if (rawOverrides) {
      const parsed = JSON.parse(rawOverrides);
      if (parsed && Object.prototype.hasOwnProperty.call(parsed, flagKey)) {
        return toBool(parsed[flagKey], fallback);
      }
    }
  } catch (_err) {
    // Ignore local overrides parse errors.
  }
  return toBool(envMap[flagKey], fallback);
};

