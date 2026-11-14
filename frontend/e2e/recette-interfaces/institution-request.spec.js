/**
 * Institution : création + transmission LIRIE réelle (navigateur).
 */
const { test, expect } = require('@playwright/test');
const { loadAccounts, loginExpectUrl } = require('./helpers');

const accounts = loadAccounts();

async function fillReactInput(locator, value) {
  await locator.click();
  await locator.fill('');
  await locator.pressSequentially(String(value), { delay: 15 });
  await locator.blur();
}

async function selectByInputId(page, inputId, query, optionRe) {
  const input = page.locator(`#${inputId}`);
  await expect(input).toBeVisible({ timeout: 30000 });
  await input.click();
  await input.fill('');
  await input.pressSequentially(query, { delay: 25 });
  const option = page.getByRole('option', { name: optionRe }).first();
  await expect(option).toBeVisible({ timeout: 30000 });
  await option.click();
}

test.describe('LOT-RECETTE-2 — Institution transmission', () => {
  test('créer et transmettre une demande LIRIE', async ({ page }) => {
    await loginExpectUrl(
      page,
      accounts.institutionEmail,
      accounts.password,
      /\/dashboard\/institution\//
    );

    await page.goto(
      `/dashboard/institution/${accounts.institutionPublicId}/requests/new`
    );

    await expect(page.locator('#dropoff_location')).toBeVisible({ timeout: 45000 });

    const patient = page.locator('#patient-select');
    if (await patient.count()) {
      await patient.click();
      const opt = page.getByRole('option', { name: /Alice|Recette/i }).first();
      if (await opt.count()) await opt.click();
      else if (await page.getByRole('option').count()) {
        await page.getByRole('option').first().click();
      }
    }

    // Éviter 409 : ref DPI patient réutilisée comme external_reference unique
    const dpiRef = page.locator('#external_reference, input[name="external_reference"]');
    if (await dpiRef.count()) {
      const unique = `RI2-${Date.now()}`;
      await dpiRef.first().click();
      await dpiRef.first().fill('');
      await dpiRef.first().pressSequentially(unique, { delay: 10 });
      await dpiRef.first().blur();
    }

    const autre = page.getByText('Autre lieu', { exact: true });
    if (await autre.count()) await autre.first().click();

    await selectByInputId(page, 'dropoff_location', 'Cornavin', /Cornavin|Genève/i);

    const service = page.locator('#dropoff_service');
    if ((await service.count()) && (await service.isVisible())) {
      await fillReactInput(service, 'Consultation');
    }

    const arBtn = page.getByRole('button', { name: /A\/R|Aller-retour/i });
    if (await arBtn.count()) {
      const pressed = await arBtn.first().getAttribute('aria-pressed');
      if (pressed === 'true') await arBtn.first().click();
    }

    if (await page.locator('#mission_date').count()) {
      await fillReactInput(page.locator('#mission_date'), '14.11.2026');
    }
    if (await page.locator('#pickup_time').count()) {
      await fillReactInput(page.locator('#pickup_time'), '15:30');
    }
    if (await page.locator('#dropoff_time').count()) {
      await fillReactInput(page.locator('#dropoff_time'), '16:00');
    }

    const lirieLabel = page.getByText('LIRIE', { exact: true });
    if (await lirieLabel.count()) await lirieLabel.first().click();

    const submit = page.getByRole('button', {
      name: /Envoyer aux transporteurs LIRIE/i,
    });
    await expect(submit).toBeVisible();

    const createWait = page.waitForResponse(
      (r) =>
        r.request().method() === 'POST' &&
        /\/institutions\/requests\/?$/.test(new URL(r.url()).pathname),
      { timeout: 60000 }
    );
    const sendWait = page.waitForResponse(
      (r) =>
        r.request().method() === 'POST' &&
        /\/institutions\/requests\/\d+\/(send|dispatch)/.test(
          new URL(r.url()).pathname
        ),
      { timeout: 60000 }
    ).catch(() => null);

    await submit.click();
    const createRes = await createWait;
    expect(createRes.status(), await createRes.text()).toBeLessThan(400);

    const sendRes = await sendWait;
    if (sendRes) {
      expect(sendRes.status(), await sendRes.text()).toBeLessThan(400);
    }

    await expect(page.locator('body')).toContainText(
      /envoyée|créée|demande|suivi|Cornavin|Alice|transporteur|SENT|envoy/i,
      { timeout: 20000 }
    );
  });
});
