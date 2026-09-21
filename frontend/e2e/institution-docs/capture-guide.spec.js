const { test, expect } = require('@playwright/test');
const { loginInstitution } = require('./helpers/auth');
const {
  attachCriticalGuards,
  waitForDashboardReady,
  waitForRequestsListReady,
  waitForRequestDetailReady,
  waitForPatientsListReady,
} = require('./helpers/waitReady');
const {
  capturePortalUntil,
  captureColumnUntil,
  captureColumnUntilBox,
  unionBoxes,
} = require('./helpers/screenshot');
const { assertDocsTenantOnly } = require('./helpers/privacy');
const { inspectCapture } = require('./helpers/inspectCapture');
const { OFFICIAL_LOT3, captureReviewSheet } = require('./helpers/reviewSheet');

test.describe('Captures documentation Institution (Lot 3)', () => {
  test('01 dashboard · 07 liste · 08 détail · 10 patients', async ({ page }) => {
    const guards = attachCriticalGuards(page);

    await test.step('login + horloge figée + dashboard prêt', async () => {
      await loginInstitution(page);
      await waitForDashboardReady(page);
    });

    await test.step('01-dashboard.png — portail compact', async () => {
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
      await assertDocsTenantOnly(page);

      await capturePortalUntil(
        page,
        [greeting, kpi, todayCard, recentCard],
        '01-dashboard.png',
        { pad: 20 }
      );
    });

    await test.step('07-transport-list.png — liste des 4 demandes', async () => {
      await page.locator('[data-tour-id="institution-sidebar"]')
        .getByRole('link', { name: 'Transports' })
        .click();
      await waitForRequestsListReady(page);

      const list = page.locator('[data-tour-id="institution-history"]');
      const cards = list.locator('[class*="requestCard"]');
      await expect(cards).toHaveCount(4);
      await expect(list.getByText('LIVRAISON', { exact: true })).toBeVisible();
      await assertDocsTenantOnly(page, list);

      await captureColumnUntil(page, list, cards.last(), '07-transport-list.png', { pad: 20 });
    });

    await test.step('08-transport-detail.png — DOCS-REQ-001 jusqu’aux besoins', async () => {
      const list = page.locator('[data-tour-id="institution-history"]');
      const cards = list.locator('[class*="requestCard"]');
      await cards.filter({ hasText: '10:30' }).click();
      await waitForRequestDetailReady(page);

      const panel = page.locator('[data-tour-id="institution-request-detail-panel"]').last();
      await assertDocsTenantOnly(page, panel);

      const besoins = panel.getByRole('heading', { name: 'Besoins' });
      const chips = panel.getByText(/Fauteuil|Assistance/).first();
      await chips.scrollIntoViewIfNeeded();
      const lastBox = unionBoxes(
        await besoins.boundingBox(),
        await chips.boundingBox(),
      );
      await captureColumnUntilBox(page, panel, lastBox, '08-transport-detail.png', { pad: 16 });
    });

    await test.step('10-patients-list.png — trois patients docs', async () => {
      await page.locator('[data-tour-id="institution-sidebar"]')
        .getByRole('link', { name: 'Patients' })
        .click();
      await waitForPatientsListReady(page);

      const search = page.getByPlaceholder(/Rechercher par nom/i);
      const listColumn = search.locator('xpath=ancestor::div[contains(@class,"listColumn")][1]');
      await expect(listColumn).toBeVisible();
      await expect(page.locator('[data-tour-id="institution-request-detail-panel"]')).toHaveCount(0);
      await assertDocsTenantOnly(page, listColumn);

      await captureColumnUntil(
        page,
        listColumn,
        listColumn.getByText('Test TEST'),
        '10-patients-list.png',
        { pad: 20 }
      );
    });

    expect(guards.pageErrors, `Erreurs JS: ${guards.pageErrors.join(' | ')}`).toEqual([]);
    expect(guards.failedApi, `5xx API: ${guards.failedApi.join(' | ')}`).toEqual([]);

    await test.step('planche de revue temporaire + métadonnées', async () => {
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
