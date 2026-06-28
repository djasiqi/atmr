import { shouldTriggerCompanyDriversWatchdog } from '../../hooks/enterprise/useCompanyDriversLiveOverlay';

describe('company drivers watchdog policy', () => {
  it('déclenche un refetch après 60s sans événement de position', () => {
    const now = 120000;
    const lastLocationEventAt = 50000;
    const lastWatchdogInvalidateAt = 0;

    expect(
      shouldTriggerCompanyDriversWatchdog({
        now,
        lastLocationEventAt,
        lastWatchdogInvalidateAt,
        silenceMs: 60000,
      })
    ).toBe(true);
  });

  it('n’invalide pas trop tôt quand le silence position est inférieur au seuil', () => {
    expect(
      shouldTriggerCompanyDriversWatchdog({
        now: 100000,
        lastLocationEventAt: 50001,
        lastWatchdogInvalidateAt: 0,
        silenceMs: 60000,
      })
    ).toBe(false);
  });

  it('évite la boucle d’invalidation pendant un même silence prolongé', () => {
    expect(
      shouldTriggerCompanyDriversWatchdog({
        now: 180000,
        lastLocationEventAt: 10000,
        lastWatchdogInvalidateAt: 150000,
        silenceMs: 60000,
      })
    ).toBe(false);
  });

  it('déclenche si positions silencieuses même quand d’autres événements socket sont récents', () => {
    expect(
      shouldTriggerCompanyDriversWatchdog({
        now: 130000,
        lastLocationEventAt: 50000,
        lastWatchdogInvalidateAt: 0,
        silenceMs: 60000,
      })
    ).toBe(true);
  });
});
