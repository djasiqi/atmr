const { expect } = require('@playwright/test');
const { freezeDocsClock } = require('./clock');

const INSTITUTION_DASHBOARD_URL = /\/dashboard\/institution\/[^/?#]+/;

/**
 * Identifiants du tenant documentation.
 * Lot 1 : fournis par l'environnement (tenant docs créé au Lot 2).
 * Compatible avec le futur compte docs.institution@docs.lirie.local.
 */
function getInstitutionDocsCredentials() {
  const email = (process.env.INSTITUTION_DOCS_EMAIL || '').trim();
  const password = process.env.INSTITUTION_DOCS_PASSWORD || '';

  if (!email || !password) {
    throw new Error(
      'INSTITUTION_DOCS_EMAIL et INSTITUTION_DOCS_PASSWORD doivent être définis '
      + 'pour le socle documentation Institution. '
      + 'Lot 2 créera le compte docs.institution@docs.lirie.local. '
      + 'Ex: INSTITUTION_DOCS_EMAIL=docs.institution@docs.lirie.local '
      + 'INSTITUTION_DOCS_PASSWORD=*** npm run e2e:institution-docs'
    );
  }

  return { email, password };
}

/**
 * Ouvre une session Institution réelle (formulaire /login).
 * Fige l'horloge avant le chargement pour que le dashboard soit déterministe.
 * @param {import('@playwright/test').Page} page
 * @param {{ email?: string, password?: string }} [credentials]
 */
async function loginInstitution(page, credentials = getInstitutionDocsCredentials()) {
  await page.goto('/login');
  await expect(page.getByRole('heading', { name: 'Connexion' })).toBeVisible();
  await expect(page.locator('input#email')).toBeVisible();

  await page.locator('input#email').fill(credentials.email);
  await page.locator('input#password').fill(credentials.password);
  await page.getByRole('button', { name: 'Se connecter' }).click();

  try {
    await expect(page).toHaveURL(INSTITUTION_DASHBOARD_URL);
  } catch (err) {
    const alertText = (await page.getByRole('alert').textContent().catch(() => '')) || '';
    const hint = alertText.trim() ? ` Alerte: ${alertText.trim()}` : '';
    throw new Error(
      `Login Institution échoué (URL=${page.url()}).${hint} ${err.message}`
    );
  }

  // Horloge figée après l'auth : un Date client antérieur à l'émission JWT
  // ferait échouer le login. Reload pour que le dashboard lise l'epoch docs.
  await freezeDocsClock(page);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page).toHaveURL(INSTITUTION_DASHBOARD_URL);
  await expect(page).not.toHaveURL(/\/demo\//);
}

module.exports = {
  INSTITUTION_DASHBOARD_URL,
  getInstitutionDocsCredentials,
  loginInstitution,
};
