const { test, expect } = require('@playwright/test');
const { loadAccounts, loginExpectUrl } = require('./helpers');

const accounts = loadAccounts();

test.describe('LOT-RECETTE-INTERFACES-1 — navigateur', () => {
  test('CLIENT — connexion + dashboard + ouverture formulaire réservation', async ({
    page,
  }) => {
    const apiHits = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/') && req.method() !== 'OPTIONS') {
        apiHits.push(`${req.method()} ${req.url()}`);
      }
    });

    await loginExpectUrl(
      page,
      accounts.clientEmail,
      accounts.password,
      /\/dashboard\/client\//
    );
    await expect(page).not.toHaveURL(/\/demo\//);

    // Formulaire de réservation sur le dashboard client
    const pickup = page.getByLabel(/Départ|Adresse de départ|Lieu de prise/i).first();
    const dest = page.getByLabel(/Destination|Arrivée|Lieu d.arrivée/i).first();
    // Fallbacks textuels si labels dynamiques
    if (!(await pickup.count())) {
      await expect(
        page.getByText(/Réserver|Course|Transport/i).first()
      ).toBeVisible({ timeout: 30000 });
    } else {
      await pickup.fill('Rue du Rhône 1, Genève');
      await dest.fill('Hôpital, Genève');
    }

    // Navigation liste réservations
    await page.goto(`/reservations/${accounts.clientPublicId}`);
    await expect(page).toHaveURL(new RegExp(`/reservations/${accounts.clientPublicId}`));
    await expect(
      page.getByText(/course|réservation|Aucune|venir/i).first()
    ).toBeVisible({ timeout: 30000 });

    const bookingApis = apiHits.filter((u) => /bookings|clients/i.test(u));
    expect(bookingApis.length).toBeGreaterThan(0);
  });

  test('ENTREPRISE — connexion + réservations + mode manuel visible', async ({
    page,
  }) => {
    const apiHits = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/') && req.method() !== 'OPTIONS') {
        apiHits.push(`${req.method()} ${req.url()}`);
      }
    });

    await loginExpectUrl(
      page,
      accounts.companyEmail,
      accounts.password,
      /\/dashboard\/company\//
    );
    await expect(page).not.toHaveURL(/\/demo\//);

    // Espace réservations / transports
    const base = `/dashboard/company/${accounts.companyPublicId}`;
    await page.goto(`${base}/reservations`);
    await expect(page).toHaveURL(new RegExp(`${base}/reservations`));
    await expect(
      page.getByText(/Réservation|Transport|Mission|Offre|Aucun/i).first()
    ).toBeVisible({ timeout: 45000 });

    // Settings opérations : mode auto ne doit pas être proposé comme actif
    await page.goto(`${base}/settings`);
    // Si onglet opérations présent
    const ops = page.getByRole('tab', { name: /Opérations|Dispatch/i });
    if (await ops.count()) {
      await ops.click();
      const bodyText = await page.locator('body').innerText();
      expect(bodyText.toLowerCase()).not.toMatch(/fully.?auto.*activ/);
    }

    expect(apiHits.some((u) => /companies|reservations|bookings/i.test(u))).toBeTruthy();
  });

  test('INSTITUTION — connexion + création demande (écran)', async ({ page }) => {
    const apiHits = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/') && req.method() !== 'OPTIONS') {
        apiHits.push(`${req.method()} ${req.url()}`);
      }
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

    await expect(
      page.getByText(/demande|patient|transport|départ|destination/i).first()
    ).toBeVisible({ timeout: 45000 });

    expect(apiHits.some((u) => /institution|request/i.test(u))).toBeTruthy();
  });

  test('MULTI — Client liste + Entreprise voit le périmètre (sync écrans)', async ({
    page,
  }) => {
    // Client
    await loginExpectUrl(
      page,
      accounts.clientEmail,
      accounts.password,
      /\/dashboard\/client\//
    );
    await page.goto(`/reservations/${accounts.clientPublicId}`);
    await expect(page.locator('body')).toContainText(/course|réservation|Aucune/i);

    // Logout via navigation login (évite dépendre du menu)
    await page.goto('/login');
    await loginExpectUrl(
      page,
      accounts.companyEmail,
      accounts.password,
      /\/dashboard\/company\//
    );
    await page.goto(
      `/dashboard/company/${accounts.companyPublicId}/reservations`
    );
    await expect(page.locator('body')).toContainText(
      /Réservation|Transport|Mission|Offre|Aucun/i
    );
  });
});
