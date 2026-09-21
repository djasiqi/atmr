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

module.exports = {
  LOADING_TEXT,
  attachCriticalGuards,
  waitForLoadingGone,
  waitForSonnerGone,
  waitForDashboardReady,
  waitForInstitutionViewReady,
};
