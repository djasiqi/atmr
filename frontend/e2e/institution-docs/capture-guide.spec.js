const { test, expect } = require('@playwright/test');
const { loginInstitution } = require('./helpers/auth');
const {
  attachCriticalGuards,
  waitForDashboardReady,
  waitForRequestsListReady,
  waitForRequestDetailReady,
} = require('./helpers/waitReady');
const {
  captureMainUntil,
  captureColumnUntil,
  captureElementUntil,
} = require('./helpers/screenshot');
const { assertDocsTenantOnly } = require('./helpers/privacy');
const { inspectCapture } = require('./helpers/inspectCapture');
const { OFFICIAL_LOT3, captureReviewSheet } = require('./helpers/reviewSheet');
const { measureDetailPanel, measureTextOverflow } = require('./helpers/measurePanel');

/** Hash figé de 10-patients-list.png (Lot 3 validé) — ne plus régénérer. */
const FROZEN_PATIENTS_SHA256 = 'b5978fa7336c44c36412ba02feac6c6342d826008acdf51babce998a801fdbad';

test.describe('Captures documentation Institution (Lot 3.1)', () => {
  test('01 dashboard · 07 liste · 08 détail (10 figée)', async ({ page }) => {
    const guards = attachCriticalGuards(page);

    await test.step('login + horloge figée + dashboard prêt', async () => {
      await loginInstitution(page);
      await waitForDashboardReady(page);
    });

    await test.step('01-dashboard.png — contenu sans sidebar', async () => {
      const greeting = page.locator('[data-tour-id="institution-dashboard"]');
      const kpi = page.locator('[data-tour-id="institution-kpi-grid"]');
      const todayCard = page.locator('[data-tour-id="institution-create-request"]');
      const recentCard = page.locator('[data-tour-id="institution-history"]');

      await expect(greeting).toContainText(/Bonjour/);
      await expect(greeting).toContainText(/16 mars 2026/i);
      await expect(kpi).toBeVisible();
      await expect(kpi).not.toContainText('—');
      await expect(todayCard).toContainText('Transports du jour');
      await expect(recentCard).toContainText('Demandes récentes');
      await expect(page.getByRole('button', { name: 'Nouvelle demande' })).toBeVisible();
      await expect(todayCard.getByText('Brouillon')).toHaveCount(0);
      await assertDocsTenantOnly(page);

      await captureMainUntil(
        page,
        [greeting, kpi, todayCard, recentCard],
        '01-dashboard.png',
        { pad: 20 }
      );
    });

    await test.step('07-transport-list.png — liste sans diffusion expirée', async () => {
      await page.locator('[data-tour-id="institution-sidebar"]')
        .getByRole('link', { name: 'Transports' })
        .click();
      await waitForRequestsListReady(page);

      const list = page.locator('[data-tour-id="institution-history"]');
      const cards = list.locator('[class*="requestCard"]');
      await expect(cards).toHaveCount(4);
      await expect(list.getByText('LIVRAISON', { exact: true })).toBeVisible();
      await expect(list.getByText('En attente d\'offre').first()).toBeVisible();
      await assertDocsTenantOnly(page, list);

      await captureColumnUntil(page, list, cards.last(), '07-transport-list.png', { pad: 20 });
    });

    await test.step('08-transport-detail.png — vrai panneau 440 CSS', async () => {
      const list = page.locator('[data-tour-id="institution-history"]');
      const cards = list.locator('[class*="requestCard"]');
      await cards.filter({ hasText: '10:30' }).click();
      await waitForRequestDetailReady(page);

      const column = page.locator('[data-tour-id="institution-request-detail-panel"]').first();
      const panel = page.locator('[data-tour-id="institution-request-detail-panel"]').last();
      const measurements = await measureDetailPanel(page);
      // eslint-disable-next-line no-console
      console.log('[institution-docs] detail-panel boundingBox', JSON.stringify(measurements, null, 2));

      await assertDocsTenantOnly(page, panel);
      await expect(panel.getByText('DOCS-REQ-001')).toBeVisible();
      await expect(panel.getByText(/Transport patient/i)).toBeVisible();

      const overflowRef = await measureTextOverflow(panel, 'DOCS-REQ-001');
      const overflowType = await measureTextOverflow(panel, 'Transport patient');
      // eslint-disable-next-line no-console
      console.log('[institution-docs] overflow', JSON.stringify({ overflowRef, overflowType }));

      const besoinsChips = panel.getByText('Assistance').first();
      await besoinsChips.scrollIntoViewIfNeeded();
      await captureElementUntil(
        page,
        column,
        besoinsChips,
        '08-transport-detail.png',
        { pad: 16 }
      );

      const report = inspectCapture('08-transport-detail.png');
      expect(report.width, '08 doit capturer le panneau ~440 CSS (880 px @2x)').toBeGreaterThanOrEqual(800);

      if (overflowRef.overflows || overflowType.overflows) {
        throw new Error(
          'PRODUCT UI ISSUE CONFIRMED: overflow dans le panneau détail à 440 CSS px. '
          + `DOCS-REQ-001 overflows=${overflowRef.overflows} `
          + `(client=${overflowRef.clientWidth}, scroll=${overflowRef.scrollWidth}); `
          + `Transport patient overflows=${overflowType.overflows} `
          + `(client=${overflowType.clientWidth}, scroll=${overflowType.scrollWidth}). `
          + 'Aucun contournement documentaire.'
        );
      }
    });

    expect(guards.pageErrors, `Erreurs JS: ${guards.pageErrors.join(' | ')}`).toEqual([]);
    expect(guards.failedApi, `5xx API: ${guards.failedApi.join(' | ')}`).toEqual([]);

    await test.step('planche de revue + 10 figée', async () => {
      const frozen = inspectCapture('10-patients-list.png');
      expect(frozen.sha256, '10-patients-list.png ne doit plus être régénérée').toBe(
        FROZEN_PATIENTS_SHA256
      );
      await captureReviewSheet(page);
      const reports = OFFICIAL_LOT3.map((name) => inspectCapture(name));
      for (const report of reports) {
        expect(report.width, report.filename).toBeGreaterThan(600);
        expect(report.height, report.filename).toBeGreaterThan(400);
        // eslint-disable-next-line no-console
        console.log(
          `[institution-docs] ${report.filename} ${report.width}x${report.height} `
          + `${report.bytes}o sha256=${report.sha256}`
        );
      }
    });
  });
});
