const { test, expect } = require('@playwright/test');
const { loginInstitution } = require('./helpers/auth');
const { attachCriticalGuards, waitForDashboardReady } = require('./helpers/waitReady');
const { captureLocator } = require('./helpers/screenshot');

test.describe('Socle documentation Institution (Lot 1)', () => {
  test('smoke: login → dashboard Institution déterministe', async ({ page }) => {
    const guards = attachCriticalGuards(page);

    await loginInstitution(page);
    await waitForDashboardReady(page);

    const dashboard = page.locator('[data-tour-id="institution-dashboard"]');
    await expect(dashboard).toBeVisible();
    await expect(dashboard).toContainText(/Bonjour/);
    await expect(dashboard).toContainText(/16 mars 2026/i);
    await expect(page.locator('[data-tour-id="institution-kpi-grid"]')).toBeVisible();

    expect(guards.pageErrors, `Erreurs JS: ${guards.pageErrors.join(' | ')}`).toEqual([]);
    expect(guards.failedApi, `5xx API: ${guards.failedApi.join(' | ')}`).toEqual([]);

    // Capture temporaire de validation — pas le 01-dashboard.png officiel.
    await captureLocator(page, dashboard, '_smoke-dashboard.png');
  });
});
