import { shouldTriggerCompanyDriversWatchdog } from '../../hooks/enterprise/useCompanyDriversLiveOverlay';

describe('company drivers watchdog policy', () => {
  it('déclenche un refetch après 60s de silence', () => {
    const now = 120000;
    const lastSocketEventAt = 50000;
    const lastWatchdogInvalidateAt = 0;

    expect(
      shouldTriggerCompanyDriversWatchdog({
        now,
        lastSocketEventAt,
        lastWatchdogInvalidateAt,
        silenceMs: 60000,
      })
    ).toBe(true);
  });

  it('n’invalide pas trop tôt quand le silence est inférieur au seuil', () => {
    expect(
      shouldTriggerCompanyDriversWatchdog({
        now: 100000,
        lastSocketEventAt: 50001,
        lastWatchdogInvalidateAt: 0,
        silenceMs: 60000,
      })
    ).toBe(false);
  });

  it('évite la boucle d’invalidation pendant un même silence prolongé', () => {
    expect(
      shouldTriggerCompanyDriversWatchdog({
        now: 180000,
        lastSocketEventAt: 10000,
        lastWatchdogInvalidateAt: 150000,
        silenceMs: 60000,
      })
    ).toBe(false);
  });
});
