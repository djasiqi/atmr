/**
 * Entreprise : attribution réelle via la modale Assigner (pas d'API directe).
 */
const { test, expect } = require('@playwright/test');
const { loadAccounts, loginExpectUrl } = require('./helpers');

const accounts = loadAccounts();
const TODAY = '2026-11-13';

test.describe('LOT-RECETTE-2 — Entreprise modale Assigner', () => {
  test('ouvrir modale, sélectionner chauffeur, POST assign', async ({ page }) => {
    const apiLog = [];
    page.on('response', (res) => {
      if (!res.url().includes('/api/')) return;
      apiLog.push({
        status: res.status(),
        method: res.request().method(),
        path: new URL(res.url()).pathname,
      });
    });

    await loginExpectUrl(
      page,
      accounts.companyEmail,
      accounts.password,
      /\/dashboard\/company\//
    );

    const base = `/dashboard/company/${accounts.companyPublicId}`;

    await page.goto(`${base}/reservations?date=${TODAY}`);
    await expect(page.locator('body')).toContainText(
      /Mont-Blanc|Cornavin|CLIENT\.RI1|Réservation/i,
      { timeout: 45000 }
    );

    const accept = page.getByRole('button', { name: /^Accepter$/i });
    if (await accept.count()) {
      await page.getByText(/Mont-Blanc|Cornavin/i).first().click();
      if (await accept.count()) {
        await accept.first().click();
        await expect
          .poll(
            () =>
              apiLog.some(
                (e) => e.method === 'POST' && /accept/i.test(e.path) && e.status < 400
              ),
            { timeout: 30000 }
          )
          .toBeTruthy();
      }
    }

    await page.goto(`${base}?date=${TODAY}`);
    await expect
      .poll(
        () =>
          apiLog.some(
            (e) => e.method === 'GET' && /drivers/i.test(e.path) && e.status < 400
          ),
        { timeout: 45000 }
      )
      .toBeTruthy();

    await page.getByRole('button', { name: /Sans chauffeur/i }).click();
    await expect(page.locator('body')).toContainText(/Manuel/i);

    const assignBtn = page
      .getByRole('button', { name: /Assigner un chauffeur/i })
      .first();
    await expect(assignBtn).toBeVisible({ timeout: 30000 });
    await assignBtn.click();

    const dialog = page.getByRole('dialog', { name: /Assigner un chauffeur/i });
    await expect(dialog).toBeVisible({ timeout: 15000 });
    await expect(dialog.getByText(/Aucun chauffeur disponible/i)).toHaveCount(0);
    await expect(dialog.getByText('Recette driver.ri1')).toBeVisible();

    // Bouton « Assigner » de la ligne chauffeur (pas le titre de la modale)
    await dialog.getByRole('button', { name: /^Assigner$/i }).click();

    await expect
      .poll(
        () =>
          apiLog.some(
            (e) =>
              e.method === 'POST' &&
              /assign/i.test(e.path) &&
              e.status < 400
          ),
        { timeout: 45000 }
      )
      .toBeTruthy();

    await expect(dialog).toBeHidden({ timeout: 15000 });
  });
});
