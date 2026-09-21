const { test, expect } = require('@playwright/test');
const { loginInstitution } = require('./helpers/auth');
const {
  attachCriticalGuards,
  waitForDashboardReady,
  waitForCreateFormReady,
} = require('./helpers/waitReady');
const {
  captureLocator,
  captureClip,
  unionBoxes,
} = require('./helpers/screenshot');
const { assertDocsTenantOnly } = require('./helpers/privacy');
const { inspectCapture } = require('./helpers/inspectCapture');
const {
  OFFICIAL_LOT4,
  captureLot4ReviewSheet,
  captureGuideReviewSheet,
} = require('./helpers/reviewSheet');
const { measureCreateModal } = require('./helpers/measurePanel');
const {
  FORM,
  openCreateRequest,
  closeCreateRequest,
  dismissOverlays,
  fillPatientRequest,
  fillDeliveryRequest,
  fillExternalRequest,
  scrollFormToTop,
} = require('./helpers/requestForm');

async function requireBox(locator, label) {
  const box = await locator.boundingBox();
  if (!box) {
    throw new Error(`Boîte introuvable pour le cadrage: ${label}`);
  }
  return box;
}

test.describe('Captures documentation Institution (Lot 4)', () => {
  test('02 patient · 03 livraison · 04 externe', async ({ page }) => {
    const guards = attachCriticalGuards(page);

    await test.step('login + horloge figée + dashboard prêt', async () => {
      await loginInstitution(page);
      await waitForDashboardReady(page);
    });

    await test.step('02-new-request-patient.png — formulaire Patient rempli', async () => {
      await page.getByRole('button', { name: 'Nouvelle demande' }).click();
      await waitForCreateFormReady(page);
      const form = page.locator(FORM);
      await fillPatientRequest(page, { date: '16.03.2026', time: '10:30' });
      await scrollFormToTop(page);
      await dismissOverlays(page);

      await expect(form.locator('.react-select__single-value')).toContainText(/TEST/i);
      await expect(form.getByRole('button', { name: /Transport de personne|Patient/ })).toBeVisible();
      await expect(form.locator('#mission_date')).toHaveValue('16.03.2026');
      await expect(form.locator('#pickup_time')).toHaveValue('10:30');
      await expect(form.locator('#pickup_location')).toHaveValue(/Route de Démonstration 10/);
      await expect(form.locator('#dropoff_location')).toHaveValue(/Rue Exemple Médical 20/);
      await expect(form.locator('#dropoff_service')).toHaveValue('Radiologie');
      await expect(form.getByRole('button', { name: /Fauteuil/ })).toHaveAttribute('aria-pressed', 'true');
      await expect(form.getByRole('button', { name: 'Assistance', exact: true }))
        .toHaveAttribute('aria-pressed', 'true');
      await expect(form.locator('#billing_intent')).toContainText('Institution');
      await assertDocsTenantOnly(page, form);

      const metrics = await measureCreateModal(page);
      // eslint-disable-next-line no-console
      console.log('[institution-docs] 02 modal', JSON.stringify(metrics, null, 2));
      if (metrics.scroller && metrics.scroller.scrollHeight > 2500) {
        // eslint-disable-next-line no-console
        console.log(
          '[institution-docs] 02 hauteur utile élevée — un seul fichier officiel conservé, '
          + `scrollHeight=${metrics.scroller.scrollHeight}`
        );
      }

      await captureLocator(page, form, '02-new-request-patient.png');
      await closeCreateRequest(page);
    });

    await test.step('03-new-request-delivery.png — Livraison haut / milieu', async () => {
      await openCreateRequest(page);
      await waitForCreateFormReady(page);
      const form = page.locator(FORM);
      await fillDeliveryRequest(page);
      await scrollFormToTop(page);
      await dismissOverlays(page);

      const delivery = page.locator('[data-testid="institution-create-delivery-block"]');
      await expect(delivery).toBeVisible();
      await expect(delivery.locator('#delivery_description')).toHaveValue('Livraison de documents');
      await expect(form.getByRole('button', { name: 'Livraison', exact: true })).toBeVisible();
      await expect(form.locator('#mission_date')).toHaveValue('16.03.2026');
      await expect(form.locator('#pickup_time')).toHaveValue('11:45');
      await expect(form.locator('#pickup_location')).toHaveValue(/Route de Démonstration 10/);
      await expect(form.locator('#dropoff_location')).toHaveValue(/Rue Exemple Médical 20/);
      await assertDocsTenantOnly(page, form);

      const metrics = await measureCreateModal(page);
      // eslint-disable-next-line no-console
      console.log('[institution-docs] 03 modal', JSON.stringify(metrics, null, 2));

      const patient = form.locator('[data-tour-id="institution-request-patient"]');
      const livraison = form.getByRole('button', { name: 'Livraison', exact: true });
      const datetime = form.locator('[data-tour-id="institution-request-datetime"]');
      const destination = form.locator('[data-tour-id="institution-request-destination"]');
      const clip = unionBoxes(
        await requireBox(patient, 'patient'),
        await requireBox(livraison, 'Livraison'),
        await requireBox(delivery, 'bloc livraison'),
        await requireBox(datetime, 'date/heure'),
        await requireBox(destination, 'itinéraire')
      );
      await captureClip(page, {
        x: clip.x - 16,
        y: clip.y - 16,
        width: clip.width + 32,
        height: clip.height + 20,
      }, '03-new-request-delivery.png');
      await closeCreateRequest(page);
    });

    await test.step('04-new-request-external.png — Externe + transporteur', async () => {
      await page.reload({ waitUntil: 'domcontentloaded' });
      await waitForDashboardReady(page);
      await openCreateRequest(page);
      await waitForCreateFormReady(page);
      const form = page.locator(FORM);
      await fillExternalRequest(page);
      await dismissOverlays(page);

      const carrier = page.locator('#external-carrier');
      await expect(carrier).toBeVisible();
      await expect(page.locator('#create-external-carrier-name')).toHaveValue('Taxi Démo Genève');
      await expect(page.locator('#create-external-carrier-phone')).toHaveValue('+41 22 000 00 99');
      await expect(page.locator('#create-external-carrier-email'))
        .toHaveValue('externe@docs.lirie.local');
      await expect(page.locator('#create-external-carrier-reason')).toHaveValue('Transporteur habituel');
      await expect(form.locator('input[name="execution_mode"][value="external"]')).toBeChecked();
      await expect(form.locator('[data-tour-id="institution-request-submit"]')).toHaveText('Enregistrer');
      await assertDocsTenantOnly(page, form);

      const metrics = await measureCreateModal(page);
      // eslint-disable-next-line no-console
      console.log('[institution-docs] 04 modal', JSON.stringify(metrics, null, 2));

      const footer = form.locator('[class*="formFooter"]');
      const formBox = await requireBox(form, 'formulaire');
      const union = unionBoxes(
        await requireBox(carrier, 'transporteur externe'),
        await requireBox(footer, 'pied de formulaire')
      );
      await captureClip(page, {
        x: formBox.x,
        y: union.y - 12,
        width: formBox.width,
        height: (union.y + union.height + 16) - (union.y - 12),
      }, '04-new-request-external.png');
      await closeCreateRequest(page);
    });

    expect(guards.pageErrors, `Erreurs JS: ${guards.pageErrors.join(' | ')}`).toEqual([]);
    expect(guards.failedApi, `5xx API: ${guards.failedApi.join(' | ')}`).toEqual([]);

    await test.step('planches de revue Lot 4 + guide', async () => {
      await captureLot4ReviewSheet(page);
      await captureGuideReviewSheet(page);
      for (const name of OFFICIAL_LOT4) {
        const report = inspectCapture(name);
        expect(report.width, name).toBeGreaterThan(400);
        expect(report.height, name).toBeGreaterThan(200);
        // eslint-disable-next-line no-console
        console.log(
          `[institution-docs] ${report.filename} ${report.width}x${report.height} `
          + `${report.bytes}o sha256=${report.sha256}`
        );
      }
    });
  });
});
