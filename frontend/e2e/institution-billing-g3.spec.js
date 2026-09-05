const { test, expect } = require('@playwright/test');
const {
  DUPONT_ID,
  KLEIN_ID,
  installApiMocks,
  injectCompanySession,
  expectDisputeDialogInViewport,
  openInstitutionInvoice,
  openExcludedBlock,
  treatDispute,
  scrollInvoice,
} = require('./helpers/institutionBillingG3');

test.describe.configure({ mode: 'serial' });

test.describe('G3 — dialogue contestation dans le viewport', () => {
  test.beforeEach(async ({ page }) => {
    await installApiMocks(page);
    await injectCompanySession(page);
  });

  test('Institution → période → exclu → dialogue visible dans le viewport', async ({
    page,
  }) => {
    await openInstitutionInvoice(page, expect);
    await expect(page.getByTestId('institution-summary-amount')).toContainText('320');
    await openExcludedBlock(page, expect);
    await expect(
      page.getByText(/Pourquoi ces courses ne sont pas facturées/i)
    ).toBeVisible();

    await scrollInvoice(page, 'top');
    await treatDispute(page, DUPONT_ID);
    await expectDisputeDialogInViewport(page, expect);
    await expect(page.getByText('Contestation — Marie DUPONT')).toBeVisible();

    await page.getByRole('button', { name: 'Fermer la contestation' }).click();
    await expect(page.getByTestId('dispute-resolution-panel')).toHaveCount(0);
    await expect(page.getByRole('heading', { name: 'Nouvelle facture' })).toBeVisible();

    await scrollInvoice(page, 'middle');
    await treatDispute(page, DUPONT_ID);
    await expectDisputeDialogInViewport(page, expect);

    await page.keyboard.press('Escape');
    await expect(page.getByTestId('dispute-resolution-panel')).toHaveCount(0);

    await scrollInvoice(page, 'bottom');
    await treatDispute(page, DUPONT_ID);
    await expectDisputeDialogInViewport(page, expect);
    await page.getByRole('button', { name: 'Fermer la contestation' }).click();

    await treatDispute(page, KLEIN_ID);
    await expectDisputeDialogInViewport(page, expect);
    await expect(page.getByText('Contestation — Arturo KLEIN')).toBeVisible();
    await expect(page.getByText('Contestation — Marie DUPONT')).toHaveCount(0);
  });

  test('ferme puis rouvre sans quitter la facture', async ({ page }) => {
    await openInstitutionInvoice(page, expect);
    await openExcludedBlock(page, expect);
    await treatDispute(page, DUPONT_ID);
    await expectDisputeDialogInViewport(page, expect);
    await page.getByRole('button', { name: 'Fermer la contestation' }).click();
    await expect(page.getByTestId('dispute-resolution-panel')).toHaveCount(0);
    await treatDispute(page, DUPONT_ID);
    await expectDisputeDialogInViewport(page, expect);
    await expect(page.getByTestId('institution-excluded-lines')).toBeVisible();
  });
});
