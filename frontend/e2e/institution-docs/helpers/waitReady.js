const { expect } = require('@playwright/test');

const LOADING_TEXT = /Chargement([.\s…]|\.\.\.)*/i;

const CRITICAL_API_HOST = /127\.0\.0\.1|localhost|\/api\//;
const IGNORE_ASSET = /\.(map|png|jpe?g|gif|webp|svg|woff2?|css)(\?|$)/i;

/**
 * Collecte les erreurs JS non catchées et les 5xx API locales.
 * Ignore le bruit réseau (favicon, sourcemaps, tiers).
 * @param {import('@playwright/test').Page} page
 */
function attachCriticalGuards(page) {
  const state = { pageErrors: [], failedApi: [] };

  page.on('pageerror', (error) => {
    state.pageErrors.push(error.message);
  });

  page.on('response', (response) => {
    const status = response.status();
    if (status < 500) return;
    const url = response.url();
    if (!CRITICAL_API_HOST.test(url) || IGNORE_ASSET.test(url)) return;
    state.failedApi.push(`${status} ${url}`);
  });

  return state;
}

/**
 * Attend la disparition des textes de chargement dans un périmètre.
 * @param {import('@playwright/test').Page | import('@playwright/test').Locator} root
 */
async function waitForLoadingGone(root) {
  await expect(root.getByText(LOADING_TEXT)).toHaveCount(0);
}

/**
 * Attend que les toasts Sonner aient disparu (pas de sleep fixe).
 * @param {import('@playwright/test').Page} page
 */
async function waitForSonnerGone(page) {
  await expect(page.locator('[data-sonner-toast]')).toHaveCount(0);
}

/**
 * Dashboard Institution : header + KPI numériques + plus de loaders/toasts.
 * @param {import('@playwright/test').Page} page
 */
async function waitForDashboardReady(page) {
  const header = page.locator('[data-tour-id="institution-dashboard"]');
  const kpiGrid = page.locator('[data-tour-id="institution-kpi-grid"]');

  await expect(header).toBeVisible();
  await expect(kpiGrid).toBeVisible();
  await expect(kpiGrid).toContainText('Demandes totales');
  await expect(kpiGrid).not.toContainText('—');

  await waitForLoadingGone(page.locator('main'));
  await waitForSonnerGone(page);
}

/**
 * Prêt générique d'une vue Institution (liste, détail, patients…).
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator} [root]
 */
async function waitForInstitutionViewReady(page, root) {
  const scope = root || page.locator('main');
  await expect(scope).toBeVisible();
  await waitForLoadingGone(scope);
  await waitForSonnerGone(page);
}

/**
 * Liste Transports : 4 demandes docs + badge livraison (mission_type).
 * @param {import('@playwright/test').Page} page
 */
async function waitForRequestsListReady(page) {
  const list = page.locator('[data-tour-id="institution-history"]');
  await expect(list).toBeVisible();
  await waitForInstitutionViewReady(page, list);
  await expect(list.getByPlaceholder(/Rechercher un patient/i)).toBeVisible();
  await expect(list.getByRole('button', { name: /^Toutes/ })).toBeVisible();
  await expect(list.getByText('LIVRAISON', { exact: true })).toBeVisible();
  await expect(list.getByText('DOCS-REQ-002', { exact: true })).toBeVisible();
  await expect(list.getByText('Test TEST').first()).toBeVisible();
  await expect(list.getByText('Exemple Alice')).toBeVisible();
  await expect(list.getByText('10:30').first()).toBeVisible();
  await expect(list.getByText('15:30').first()).toBeVisible();
  await expect(list.getByText(/Diffusion expirée/)).toHaveCount(0);
  await expect(list.getByRole('button', { name: 'Relancer' })).toHaveCount(0);
}

/**
 * Détail DOCS-REQ-001 : trajet / détails / besoins, hors facturation.
 * @param {import('@playwright/test').Page} page
 */
async function waitForRequestDetailReady(page) {
  const panel = page.locator('[data-tour-id="institution-request-detail-panel"]').last();
  await expect(panel).toBeVisible();
  await waitForInstitutionViewReady(page, panel);
  await expect(panel.getByText(/^Demande #/)).toBeVisible();
  await expect(panel.getByRole('button', { name: 'Modifier' })).toBeVisible();
  await expect(panel.getByRole('heading', { name: 'Trajet' })).toBeVisible();
  await expect(panel.getByRole('heading', { name: 'Détails' })).toBeVisible();
  await expect(panel.getByRole('heading', { name: 'Besoins' })).toBeVisible();
  await expect(panel.getByText('Radiologie')).toBeVisible();
  await expect(panel.getByText('DOCS-REQ-001')).toBeVisible();
}

/**
 * Liste Patients : 3 fiches docs, sans panneau détail ouvert.
 * @param {import('@playwright/test').Page} page
 */
async function waitForPatientsListReady(page) {
  await waitForInstitutionViewReady(page);
  await expect(page.getByPlaceholder(/Rechercher par nom/i)).toBeVisible();
  await expect(page.getByRole('button', { name: /Nouveau patient/ })).toBeVisible();
  await expect(page.getByText('Test TEST').first()).toBeVisible();
  await expect(page.getByText('Exemple Alice').first()).toBeVisible();
  await expect(page.getByText('Démonstration Marc').first()).toBeVisible();
  await expect(page.getByText(/3 patients/)).toBeVisible();
}

/**
 * Modal Nouvelle demande prêt (parcours CTA dashboard).
 * @param {import('@playwright/test').Page} page
 */
async function waitForCreateFormReady(page) {
  const form = page.locator('[data-tour-id="institution-request-create"]');
  await expect(form).toBeVisible();
  await waitForLoadingGone(form);
  await waitForSonnerGone(page);
  await expect(form.getByRole('heading', { name: 'Nouvelle demande' })).toBeVisible();
  await expect(form.locator('#patient-select')).toBeVisible();
  await expect(form.getByRole('button', { name: 'Livraison', exact: true })).toBeVisible();
}

module.exports = {
  LOADING_TEXT,
  attachCriticalGuards,
  waitForLoadingGone,
  waitForSonnerGone,
  waitForDashboardReady,
  waitForInstitutionViewReady,
  waitForRequestsListReady,
  waitForRequestDetailReady,
  waitForPatientsListReady,
  waitForCreateFormReady,
};
