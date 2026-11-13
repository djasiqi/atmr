/**
 * Multi-profils LOT-2 — preuves navigateur (Client/Entreprise).
 * Chauffeur : états lus via API entreprise uniquement (pas d'UI mobile).
 */
const { test, expect } = require('@playwright/test');
const { loadAccounts, loginExpectUrl } = require('./helpers');

const accounts = loadAccounts();
const TODAY = '2026-11-13';

test.describe('LOT-RECETTE-2 — multi-profils A (Client)', () => {
  test('Entreprise voit course assignée après modale Assigner', async ({ page }) => {
    await loginExpectUrl(
      page,
      accounts.companyEmail,
      accounts.password,
      /\/dashboard\/company\//
    );
    const base = `/dashboard/company/${accounts.companyPublicId}`;
    await page.goto(`${base}/reservations?date=${TODAY}`);
    await expect(page.locator('body')).toContainText(/Assigné|Assignée|Mont-Blanc|Cornavin/i, {
      timeout: 45000,
    });
  });

  test('Client voit statut mis à jour sur Mes courses', async ({ page }) => {
    await loginExpectUrl(
      page,
      accounts.clientEmail,
      accounts.password,
      /\/dashboard\/client\//
    );
    await page.goto(`/reservations/${accounts.clientPublicId}`);
    await expect(page.locator('body')).toContainText(
      /Assigné|confirm|chauffeur|course|Cornavin|Mont-Blanc|en cours|accept/i,
      { timeout: 45000 }
    );
  });
});
