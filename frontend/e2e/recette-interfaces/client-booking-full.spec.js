/**
 * Création réelle d'une réservation Client via les écrans React.
 * Adresses : domicile (coords) + suggestion Photon (Cornavin).
 */
const { test, expect } = require('@playwright/test');
const { loadAccounts, loginExpectUrl } = require('./helpers');

const accounts = loadAccounts();

async function selectAutocomplete(page, combobox, query, optionRe) {
  await combobox.click();
  await combobox.fill('');
  await combobox.fill(query);
  const option = page.getByRole('option', { name: optionRe }).first();
  await expect(option).toBeVisible({ timeout: 30000 });
  await option.click();
}

test.describe('LOT-RECETTE — Client booking UI complet', () => {
  test('créer aller-retour, consulter, annuler', async ({ page }) => {
    const mutations = [];
    const apiLog = [];
    page.on('request', (req) => {
      if (
        req.url().includes('/api/') &&
        ['POST', 'PUT', 'DELETE', 'PATCH'].includes(req.method())
      ) {
        mutations.push(`${req.method()} ${new URL(req.url()).pathname}`);
      }
    });
    page.on('response', async (res) => {
      const url = res.url();
      if (!url.includes('/api/') || !['POST', 'PUT', 'DELETE', 'PATCH'].includes(res.request().method())) {
        return;
      }
      let body = '';
      try {
        body = (await res.text()).slice(0, 400);
      } catch (_) {
        /* ignore */
      }
      apiLog.push({
        status: res.status(),
        path: new URL(url).pathname,
        method: res.request().method(),
        body,
      });
    });

    await loginExpectUrl(
      page,
      accounts.clientEmail,
      accounts.password,
      /\/dashboard\/client\//
    );

    const homeBtn = page.getByRole('button', {
      name: /Utiliser mon adresse de domicile/i,
    });
    await expect(homeBtn).toBeVisible({ timeout: 30000 });
    await homeBtn.click();
    await expect(page.getByRole('combobox', { name: /Départ/i })).not.toHaveValue('');

    const dest = page.getByRole('combobox', { name: /Destination/i });
    await selectAutocomplete(page, dest, 'Cornavin', /Cornavin|Genève/i);

    const pickupTime = page
      .locator('#client-booking-departure-time')
      .or(page.getByRole('textbox', { name: 'Heure de départ', exact: true }))
      .or(page.getByRole('textbox', { name: /Heure de prise en charge/i }));
    await pickupTime.first().fill('16:30');
    const apptTime = page
      .locator('#client-booking-appointment-time')
      .or(page.getByRole('textbox', { name: /Heure du rendez-vous|Rendez-vous/i }));
    await apptTime.first().fill('17:00');
    const returnTime = page.locator('#client-booking-return-time');
    if (await returnTime.count()) {
      await returnTime.fill('18:00');
    }

    const roundTrip = page.getByRole('switch', { name: /Aller-retour/i });
    if (await roundTrip.count()) {
      const checked = await roundTrip.getAttribute('aria-checked');
      if (checked !== 'true') await roundTrip.click();
    }

    const validate = page.getByRole('button', {
      name: /Valider la demande de transport|Vérifier la demande/i,
    });
    await expect(validate).toBeVisible();
    await validate.click();

    // Étape revue privée éventuelle
    const confirmReview = page.getByRole('button', {
      name: /Confirmer et envoyer|Confirmer la demande|Envoyer la demande/i,
    });
    if (await confirmReview.count()) {
      await confirmReview.first().click();
    }

    // Attendre la création réelle (hors preview)
    await expect
      .poll(
        () =>
          apiLog.some(
            (e) =>
              e.method === 'POST' &&
              /\/clients\/[^/]+\/bookings$/.test(e.path) &&
              !e.path.includes('preview')
          ),
        { timeout: 60000 }
      )
      .toBeTruthy();

    const create = apiLog.find(
      (e) =>
        e.method === 'POST' &&
        /\/clients\/[^/]+\/bookings$/.test(e.path) &&
        !e.path.includes('preview')
    );
    expect(
      create?.status,
      `Création échouée: ${create?.status} ${create?.body}`
    ).toBeLessThan(400);

    // Feedback UI
    await expect(page.locator('body')).toContainText(
      /Demande enregistrée|en attente|référence|#\d+/i,
      { timeout: 15000 }
    );

    await page.getByRole('link', { name: /Mes courses/i }).click();
    await expect(page).toHaveURL(/\/reservations\//);
    await expect(page.locator('body')).toContainText(
      /Cornavin|Mont-Blanc|16:30|en attente|pending|confirm|course/i
    );

    const cancelBtn = page.getByRole('button', { name: /Annuler/i }).first();
    if (await cancelBtn.count()) {
      await cancelBtn.click();
      const yes = page.getByRole('button', { name: /Confirmer|Oui|Annuler la/i });
      if (await yes.count()) await yes.first().click();
      await expect
        .poll(
          () =>
            apiLog.some(
              (e) =>
                e.method === 'DELETE' ||
                (e.method === 'POST' && /cancel/i.test(e.path))
            ),
          { timeout: 20000 }
        )
        .toBeTruthy();
    }

    expect(mutations.some((m) => /\/clients\/[^/]+\/bookings$/.test(m))).toBeTruthy();
  });
});
