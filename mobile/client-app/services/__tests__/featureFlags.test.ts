describe('featureFlags', () => {
  it('defaults flags to true when env vars are absent', () => {
    const mod = require('../featureFlags') as typeof import('../featureFlags');
    const flags = mod.getFeatureFlags({});

    expect(flags.institutionMobileRequestSendEnabled).toBe(true);
    expect(flags.institutionMobileRealtimeEnabled).toBe(true);
    expect(flags.institutionMobileRoleGuardsEnabled).toBe(true);
  });

  it('turns flags off for false-like values', () => {
    const mod = require('../featureFlags') as typeof import('../featureFlags');
    const flags = mod.getFeatureFlags({
      EXPO_PUBLIC_INSTITUTION_MOBILE_REQUEST_SEND_ENABLED: 'false',
      EXPO_PUBLIC_INSTITUTION_MOBILE_REALTIME_ENABLED: '0',
      EXPO_PUBLIC_INSTITUTION_MOBILE_PATIENT_DETAIL_ENABLED: 'off',
    });

    expect(flags.institutionMobileRequestSendEnabled).toBe(false);
    expect(flags.institutionMobileRealtimeEnabled).toBe(false);
    expect(flags.institutionMobilePatientDetailEnabled).toBe(false);
  });
});
