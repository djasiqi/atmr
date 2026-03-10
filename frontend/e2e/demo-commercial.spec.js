const { test, expect } = require('@playwright/test');

const DEMO_PASSWORD =
  process.env.DEMO_PASSWORD || process.env.DEMO_DEFAULT_PASSWORD;
if (!DEMO_PASSWORD) {
  throw new Error(
    'DEMO_PASSWORD ou DEMO_DEFAULT_PASSWORD doit être défini pour les tests e2e demo. ' +
      'Ex: DEMO_PASSWORD=xxx npx playwright test e2e/demo-commercial.spec.js'
  );
}
const COMPANY_EMAIL = 'company1@demo.lirie.ch';
const INSTITUTION_EMAIL = 'institution.user1@demo.lirie.ch';

async function login(page, email) {
  await page.goto('/login');
  await expect(page.getByRole('heading', { name: 'Connexion' })).toBeVisible();
  await page.getByLabel('Adresse email').fill(email);
  await page.locator('input#password').fill(DEMO_PASSWORD);
  await page.getByRole('button', { name: 'Se connecter' }).click();
}

test.describe('Parcours demo commercial', () => {
  test('parcours transporteur: home -> mission -> fin', async ({ page }) => {
    const analyticsEvents = [];
    page.on('request', (request) => {
      if (request.url().includes('/api/v1/demo_access/analytics') && request.method() === 'POST') {
        const body = request.postDataJSON();
        analyticsEvents.push(body.event);
      }
    });

    await login(page, COMPANY_EMAIL);
    await expect(page).toHaveURL(/\/dashboard\/company\//);

    await page.goto('/demo/home');
    await expect(page.locator('[data-tour-id="demo-home"]')).toBeVisible();

    await page.getByRole('button', { name: 'Commencer ce parcours' }).first().click();
    await expect(page).toHaveURL(/demo_mission=transporteur/);
    await expect(page.locator('[data-tour-id="demo-guide-transporteur"]')).toBeVisible();
    await expect(page.locator('[data-tour-id="dashboard-transports"]')).toBeVisible();
    await expect(page.locator('[data-tour-id="create-booking"]')).toBeVisible();

    await page.getByRole('button', { name: 'Étape faite' }).first().click();
    await page.getByRole('button', { name: 'Terminer et contacter LIRIE' }).click();
    await expect(page).toHaveURL(/\/contact\/demo$/);

    await expect.poll(() => analyticsEvents).toContain('demo_session_start');
    await expect.poll(() => analyticsEvents).toContain('demo_step_reached');
    await expect.poll(() => analyticsEvents).toContain('demo_completed');
  });

  test('parcours institution + exploration libre', async ({ page }) => {
    await login(page, INSTITUTION_EMAIL);
    await expect(page).toHaveURL(/\/dashboard\/institution\//);

    await page.goto('/demo/home');
    await expect(page.locator('[data-tour-id="demo-home"]')).toBeVisible();

    const startButtons = page.getByRole('button', { name: 'Commencer ce parcours' });
    await startButtons.nth(1).click();
    await expect(page).toHaveURL(/demo_mission=institution/);
    await expect(page.locator('[data-tour-id="demo-guide-institution"]')).toBeVisible();
    await expect(page.locator('[data-tour-id="institution-dashboard"]')).toBeVisible();
    await expect(page.locator('[data-tour-id="institution-create-request"]')).toBeVisible();

    await page.goto('/demo/home');
    await page.getByRole('button', { name: 'Explorer' }).click();
    await expect(page).toHaveURL(/\/dashboard\/institution\/[^?]+$/);
    await expect(page).not.toHaveURL(/demo_mission=/);
  });
});
