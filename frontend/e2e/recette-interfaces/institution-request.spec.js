/**
 * Institution : connexion, formulaire de demande (écrans React).
 */
const { test, expect } = require('@playwright/test');
const { loadAccounts, loginExpectUrl } = require('./helpers');

const accounts = loadAccounts();

async function selectByInputId(page, inputId, query, optionRe) {
  const input = page.locator(`#${inputId}`);
  await expect(input).toBeVisible({ timeout: 30000 });
  await input.click();
  await input.fill('');
  await input.fill(query);
  const option = page.getByRole('option', { name: optionRe }).first();
  await expect(option).toBeVisible({ timeout: 30000 });
  await option.click();
}

test.describe('LOT-RECETTE — Institution demande', () => {
  test('connexion + formulaire + envoi si possible', async ({ page }) => {
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
      accounts.institutionEmail,
      accounts.password,
      /\/dashboard\/institution\//
    );
    await expect(page).not.toHaveURL(/\/demo\//);

    const createBtn = page.getByRole('button', {
      name: /Nouvelle demande|Créer une demande|Nouvelle/i,
    });
    if (await createBtn.count()) {
      await createBtn.first().click();
    } else {
      await page.goto(
        `/dashboard/institution/${accounts.institutionPublicId}/requests/new`
      );
    }

    await expect(page.locator('#pickup_location, #dropoff_location').first()).toBeVisible({
      timeout: 45000,
    });

    // Patient / bénéficiaire si requis
    const patient = page.locator('#patient-select, [id*="patient"]').first();
    if (await patient.count()) {
      await patient.click();
      const opt = page.getByRole('option').first();
      if (await opt.count()) await opt.click();
    }

    if (await page.locator('#pickup_location').count()) {
      await selectByInputId(page, 'pickup_location', 'Cornavin', /Cornavin|Genève/i);
    }
    if (await page.locator('#dropoff_location').count()) {
      await selectByInputId(page, 'dropoff_location', 'Mont-Blanc', /Mont-Blanc|Genève/i);
    }

    const dateInput = page.locator('#mission_date');
    if (await dateInput.count()) {
      await dateInput.fill('13.11.2026');
    }

    const submit = page.getByRole('button', {
      name: /Envoyer|Transmettre|Créer|Valider|Soumettre/i,
    });
    if (await submit.count()) {
      await submit.first().click();
      await expect
        .poll(
          () =>
            apiLog.some(
              (e) =>
                e.method === 'POST' &&
                /institution|request|booking/i.test(e.path)
            ) || page.url().includes('/requests'),
          { timeout: 20000 }
        )
        .toBeTruthy();
    }

    expect(
      apiLog.some((e) => /institution|request|geocode|companies/i.test(e.path))
    ).toBeTruthy();
  });
});
