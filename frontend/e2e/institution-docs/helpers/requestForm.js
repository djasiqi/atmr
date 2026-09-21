const { expect } = require('@playwright/test');

const FORM = '[data-tour-id="institution-request-create"]';
const CTA = '[data-tour-id="institution-create-request-cta"]';

const PICKUP_ADDRESS =
  'Établissement de démonstration LIRIE, Route de Démonstration 10, 1200 Genève';
const DROPOFF_ADDRESS =
  'Hôpital de démonstration, Rue Exemple Médical 20, 1200 Genève';
const DROPOFF_STREET = 'Rue Exemple Médical 20, 1200 Genève';

/**
 * Ouvre le formulaire par le CTA réel du dashboard.
 * @param {import('@playwright/test').Page} page
 */
async function openCreateRequest(page) {
  await page.locator(CTA).click();
  const form = page.locator(FORM);
  await expect(form).toBeVisible();
  await expect(form.getByRole('heading', { name: 'Nouvelle demande' })).toBeVisible();
  await expect(form.locator('#patient-select')).toBeVisible();
  await expect(form.getByRole('button', { name: 'Livraison', exact: true })).toBeVisible();
  return form;
}

/**
 * Ferme le modal sans soumettre (aucune demande persistée).
 * @param {import('@playwright/test').Page} page
 */
async function closeCreateRequest(page) {
  const form = page.locator(FORM);
  if (await form.count() === 0) return;
  await form.getByRole('button', { name: 'Fermer' }).click();
  await expect(form).toHaveCount(0);
}

/**
 * Ferme autocomplete / date / heure / menus — jamais Escape (fermerait le modal).
 * @param {import('@playwright/test').Page} page
 */
async function dismissOverlays(page) {
  await page.evaluate(() => {
    const active = document.activeElement;
    if (active && active !== document.body && typeof active.blur === 'function') {
      active.blur();
    }
  });
  const heading = page.locator(`${FORM} h1`);
  if (await heading.count()) {
    await heading.click({ force: true, position: { x: 10, y: 8 } });
  }
  await expect(page.locator('#pickup_location-ac-listbox')).toHaveCount(0);
  await expect(page.locator('#dropoff_location-ac-listbox')).toHaveCount(0);
  await expect(page.locator('.react-select__menu')).toHaveCount(0);
  await expect(page.locator('[role="dialog"][aria-label="Choisir une date"]')).toHaveCount(0);
}

/**
 * Saisit une adresse puis ferme le dropdown (sans dépendre d'un résultat Maps).
 * @param {import('@playwright/test').Page} page
 * @param {'pickup_location'|'dropoff_location'} inputId
 * @param {string} value
 */
async function fillAddressField(page, inputId, value) {
  const input = page.locator(`#${inputId}`);
  await expect(input).toBeVisible();
  const acWait = page.waitForResponse(
    (response) => response.url().includes('geocode/autocomplete'),
    { timeout: 4000 }
  ).catch(() => null);
  await input.click();
  await input.fill(value);
  await expect(input).toHaveValue(value);
  await dismissOverlays(page);
  await acWait;
  await expect(page.locator(`#${inputId}-ac-listbox`)).toHaveCount(0);
  await expect(input).toHaveValue(value);
}

/**
 * @param {import('@playwright/test').Page} page
 */
async function selectPatientTest(page) {
  const input = page.locator('#patient-select');
  await input.click();
  const existing = page
    .locator('.react-select__option')
    .filter({ hasText: /Test TEST/ })
    .filter({ hasNotText: /Nouveau patient/ });
  if (await existing.count() === 0) {
    await input.fill('TEST');
  }
  await expect(existing.first()).toBeAttached();
  const shown = existing.filter({ visible: true });
  if (await shown.count()) {
    await shown.first().click();
  } else {
    await input.press('ArrowDown');
    await input.press('Enter');
  }
  await expect(page.locator('.react-select__single-value')).toContainText(/TEST/i);
  await expect(page.locator('.react-select__menu')).toHaveCount(0);
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {string} display  ex. 16.03.2026
 */
async function fillMissionDate(page, display) {
  const input = page.locator('#mission_date');
  await input.click();
  await input.fill('');
  await input.fill(display);
  await input.blur();
  await dismissOverlays(page);
  await expect(input).toHaveValue(display);
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {string} value  ex. 10:30
 */
async function fillPickupTime(page, value) {
  const input = page.locator('#pickup_time');
  await input.click();
  await input.fill(value.replace(':', ''));
  await input.blur();
  await dismissOverlays(page);
  await expect(input).toHaveValue(value);
}

/**
 * Désactive A/R si le produit l'active par défaut (itinéraire pédagogique simple).
 * @param {import('@playwright/test').Page} page
 */
async function disableRoundTripIfOn(page) {
  const form = page.locator(FORM);
  const ar = form.getByRole('button', { name: /A\/R/ });
  await expect(ar).toBeVisible();
  if ((await ar.getAttribute('aria-pressed')) === 'true') {
    await ar.click();
    await expect(ar).toHaveAttribute('aria-pressed', 'false');
  }
}

/**
 * Facturé à = Institution (ChipSelect existant).
 * @param {import('@playwright/test').Page} page
 */
async function selectBillingInstitution(page) {
  const chip = page.locator('#billing_intent');
  await expect(chip).toBeVisible();
  if ((await chip.innerText()).includes('Institution')) {
    return;
  }
  await chip.click();
  await page.getByRole('option', { name: 'Institution', exact: true }).click();
  await expect(chip).toContainText('Institution');
  await expect(page.getByRole('option', { name: 'Institution', exact: true })).toHaveCount(0);
}

/**
 * Contact pédagogique si le téléphone n'est pas prérempli.
 * @param {import('@playwright/test').Page} page
 */
async function ensureDocsContact(page) {
  const name = page.locator('#requester_name');
  const phone = page.locator('#requester_phone');
  await expect(name).toBeVisible();
  const nameVal = (await name.inputValue()).trim();
  const phoneVal = (await phone.inputValue()).trim();
  if (nameVal.includes('Test Documentation') && phoneVal.includes('22 000 00 10')) {
    return;
  }
  if (!phoneVal) {
    await page.locator(FORM).getByRole('button', { name: /Modifier/ }).click();
    await page.locator('#onsite_name').fill('Test Documentation');
    await page.locator('#onsite_phone').fill('+41 22 000 00 10');
  }
}

/**
 * Itinéraire docs commun (départ institution + destination médicale).
 * @param {import('@playwright/test').Page} page
 */
async function fillDocsRoute(page) {
  await disableRoundTripIfOn(page);
  await fillAddressField(page, 'pickup_location', PICKUP_ADDRESS);
  await fillAddressField(page, 'dropoff_location', DROPOFF_ADDRESS);
  const establishment = page.locator('#dropoff_establishment');
  if (await establishment.count()) {
    await establishment.fill('Hôpital de démonstration');
  }
  const service = page.locator('#dropoff_service');
  if (await service.count()) {
    await service.fill('Radiologie');
  }
  const medical = page.locator(FORM).getByRole('button', { name: 'Médical', exact: true });
  if (await medical.count()) {
    await expect(medical).toHaveAttribute('aria-pressed', 'true');
  }
}

/**
 * Demande patient pédagogique — ne pas soumettre.
 * @param {import('@playwright/test').Page} page
 * @param {{ date: string, time: string }} values
 */
async function fillPatientRequest(page, values) {
  await expect(page.locator('#pickup_location')).toBeVisible();
  await disableRoundTripIfOn(page);
  await selectPatientTest(page);
  await fillDocsRoute(page);
  await fillMissionDate(page, values.date);
  await fillPickupTime(page, values.time);

  const fauteuil = page.locator(FORM).getByRole('button', { name: /Fauteuil/ });
  await expect(fauteuil).toBeVisible();
  if ((await fauteuil.getAttribute('aria-pressed')) !== 'true') {
    await fauteuil.click();
    await expect(fauteuil).toHaveAttribute('aria-pressed', 'true');
  }
  const assistance = page.locator(FORM).getByRole('button', { name: 'Assistance', exact: true });
  if ((await assistance.getAttribute('aria-pressed')) !== 'true') {
    await assistance.click();
    await expect(assistance).toHaveAttribute('aria-pressed', 'true');
  }
  await page.locator('#patient_notes').fill('Assistance à la marche');

  await selectBillingInstitution(page);
  await ensureDocsContact(page);
  await dismissOverlays(page);
}

/**
 * Mode Livraison — mission_type est la source de vérité.
 * @param {import('@playwright/test').Page} page
 */
async function fillDeliveryRequest(page) {
  const livraison = page.locator(FORM).getByRole('button', { name: 'Livraison', exact: true });
  await livraison.click();
  const delivery = page.locator('[data-testid="institution-create-delivery-block"]');
  await expect(delivery).toBeVisible();
  await expect(delivery.locator('#delivery_description')).toBeVisible();
  await page.locator('#delivery_description').fill('Livraison de documents');
  await expect(page.locator('#delivery_description')).toHaveValue('Livraison de documents');
  await fillDocsRoute(page);
  await fillMissionDate(page, '16.03.2026');
  await fillPickupTime(page, '11:45');
  await dismissOverlays(page);
}

/**
 * Demande patient + transporteur externe — ne pas enregistrer.
 * @param {import('@playwright/test').Page} page
 */
async function fillExternalRequest(page) {
  await fillPatientRequest(page, { date: '17.03.2026', time: '14:00' });
  await page.locator(FORM).locator('label').filter({ hasText: /^Externe$/ }).click();
  await expect(page.locator('input[name="execution_mode"][value="external"]')).toBeChecked();
  const carrier = page.locator('#external-carrier');
  await expect(carrier).toBeVisible();
  await page.locator('#create-external-carrier-name').fill('Taxi Démo Genève');
  await page.locator('#create-external-carrier-phone').fill('+41 22 000 00 99');
  await page.locator('#create-external-carrier-email').fill('externe@docs.lirie.local');
  await page.locator('#create-external-carrier-reason').fill('Transporteur habituel');
  await expect(page.locator('[data-tour-id="institution-request-submit"]')).toHaveText('Enregistrer');
  await dismissOverlays(page);
}

/**
 * Remonte le corps scrollable du modal (Patient visible).
 * @param {import('@playwright/test').Page} page
 */
async function scrollFormToTop(page) {
  await page.locator(FORM).evaluate((root) => {
    const scroller = root.querySelector('form > div');
    if (scroller) scroller.scrollTop = 0;
  });
}

module.exports = {
  FORM,
  CTA,
  PICKUP_ADDRESS,
  DROPOFF_ADDRESS,
  DROPOFF_STREET,
  openCreateRequest,
  closeCreateRequest,
  dismissOverlays,
  fillAddressField,
  fillPatientRequest,
  fillDeliveryRequest,
  fillExternalRequest,
  scrollFormToTop,
};
