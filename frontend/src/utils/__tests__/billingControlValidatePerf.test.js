import {
  markBillingPayerChange,
  markBillingPayerRequestEnd,
  markBillingPayerRequestStart,
  markBillingPayerUiCommit,
  markBillingValidateClick,
  markBillingValidateRequestEnd,
  markBillingValidateRequestStart,
  markBillingValidateUiCommit,
} from '../billingControlValidatePerf';

describe('billingControlValidatePerf', () => {
  it('marque le clic et le commit UI sans lever', () => {
    const clickAt = markBillingValidateClick(45708);
    expect(typeof clickAt).toBe('number');
    const commitAt = markBillingValidateUiCommit(45708, clickAt);
    expect(commitAt).toBeGreaterThanOrEqual(clickAt);
  });

  it('mesure la durée de requête', () => {
    const clickAt = markBillingValidateClick(1);
    const startAt = markBillingValidateRequestStart(1);
    const endAt = markBillingValidateRequestEnd(1, {
      clickAt,
      requestStartAt: startAt,
      ok: true,
    });
    expect(endAt).toBeGreaterThanOrEqual(startAt);
  });

  it('marque un changement de payeur sans lever', () => {
    const clickAt = markBillingPayerChange(45708);
    const commitAt = markBillingPayerUiCommit(45708, clickAt);
    const startAt = markBillingPayerRequestStart(45708);
    const endAt = markBillingPayerRequestEnd(45708, {
      clickAt,
      requestStartAt: startAt,
      ok: true,
    });
    expect(commitAt).toBeGreaterThanOrEqual(clickAt);
    expect(endAt).toBeGreaterThanOrEqual(startAt);
  });
});
