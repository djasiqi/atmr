/** Fixtures et assertions G3 — viewport réel, pas de Jest. */

const COMPANY_ID = 1;
const COMPANY_PUBLIC_ID = 'g3-company';
const CLINIC_ID = 10;
const CLINIC_COMPANY_ID = 1;
const DUPONT_ID = 45705;
const KLEIN_ID = 45690;

const COMPANY_USER = {
  id: 1,
  public_id: COMPANY_PUBLIC_ID,
  role: 'company',
  email: 'g3.company@lirie.test',
  first_name: 'Gate',
  last_name: 'Trois',
  force_password_change: false,
};

function fakeCompanyJwt() {
  const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString(
    'base64url'
  );
  const body = Buffer.from(
    JSON.stringify({
      exp: Math.floor(Date.now() / 1000) + 86400,
      role: 'company',
      sub: '1',
      public_id: COMPANY_PUBLIC_ID,
    })
  ).toString('base64url');
  return `${header}.${body}.g3`;
}

function pendingRow(bookingId, day) {
  return {
    booking_id: bookingId,
    origin: 'LIRIE_MARKETPLACE',
    validation_status: 'pending',
    payer: 'clinic',
    eligible: false,
    invoice_bucket: 'pending_blocked',
    amount_ht: 40,
    exclusion_reason: 'market_pending_before_deadline',
    dispute_id: null,
    dispute_status: null,
    dispute_treatable: false,
    patient_name: `Patient ATTENTE ${bookingId}`,
    scheduled_at: `2026-08-${String(day).padStart(2, '0')}T10:00:00`,
    pickup_location: "Chemin des Courbes 9, Anières",
    dropoff_location: 'HUG Genève',
  };
}

function disputedRow(bookingId, patientName, day) {
  return {
    booking_id: bookingId,
    origin: 'LIRIE_MARKETPLACE',
    validation_status: 'disputed',
    payer: 'clinic',
    eligible: false,
    invoice_bucket: 'disputed_blocked',
    amount_ht: 40,
    exclusion_reason: 'disputed',
    dispute_id: bookingId,
    dispute_status: 'disputed',
    dispute_treatable: true,
    patient_name: patientName,
    scheduled_at: `2026-08-${String(day).padStart(2, '0')}T09:00:00`,
    pickup_location: "Chemin des Courbes 9, Anières",
    dropoff_location: 'HUG Genève',
  };
}

function institutionPlan() {
  const pendingIds = [401, 402, 403, 404, 405, 406];
  const clinicIds = [101, 102, 103, 104, 105, 106, 107, 108];
  const disputed = [
    disputedRow(DUPONT_ID, 'Marie DUPONT', 16),
    disputedRow(KLEIN_ID, 'Arturo KLEIN', 2),
  ];
  const pending = pendingIds.map((id, i) => pendingRow(id, 3 + i));
  return {
    clinic: {
      display_name: "Clinique les Hauts d'Anières",
      transports_count: 8,
      estimated_total: 320,
      booking_ids: clinicIds,
    },
    reconciliation: {
      buckets: {
        clinic_billable: { count: 8, amount_ht: 320, booking_ids: clinicIds },
        disputed_blocked: {
          count: 2,
          amount_ht: 80,
          booking_ids: [DUPONT_ID, KLEIN_ID],
        },
        pending_blocked: { count: 6, amount_ht: 240, booking_ids: pendingIds },
      },
      bookings: [
        ...clinicIds.map((id) => ({
          booking_id: id,
          origin: 'OWN_PORTFOLIO',
          validation_status: 'not_required',
          payer: 'clinic',
          eligible: true,
          invoice_bucket: 'clinic_billable',
          amount_ht: 40,
          patient_name: `Patient ${id}`,
          scheduled_at: '2026-08-05T10:00:00',
        })),
        ...pending,
        ...disputed,
      ],
    },
  };
}

function disputePayload(bookingId) {
  const patient = bookingId === KLEIN_ID ? 'Arturo KLEIN' : 'Marie DUPONT';
  return {
    id: bookingId,
    booking_id: bookingId,
    status: 'disputed',
    treatable: true,
    patient_name: patient,
    scheduled_at:
      bookingId === KLEIN_ID ? '2026-08-02T08:00:00' : '2026-08-16T09:00:00',
    amount_ht: 40,
    institution_reason_code: 'OTHER',
    institution_reason_text: 'OTHER: Pas de retour suite hospitalisation',
    evidence: [],
    system_facts: { driver_name: 'Chauffeur G3', completed: true },
  };
}

function json(route, body, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });
}

async function installApiMocks(page) {
  await page.route('http://127.0.0.1:5100/**', async (route) => {
    const req = route.request();
    const url = req.url();
    const path = url.replace(/^https?:\/\/127\.0\.0\.1:5100/, '');

    if (path.includes('/auth/me')) {
      return json(route, { user: COMPANY_USER });
    }
    if (path.match(/\/companies\/me\/?(\?|$)/) && req.method() === 'GET') {
      return json(route, {
        data: {
          id: COMPANY_ID,
          public_id: COMPANY_PUBLIC_ID,
          name: 'Transports G3',
          email: COMPANY_USER.email,
        },
      });
    }
    if (path.includes('/invoices/institution-invoice-plan')) {
      return json(route, { data: institutionPlan() });
    }
    if (path.includes('/clients/institutions')) {
      return json(route, {
        institutions: [
          {
            id: CLINIC_ID,
            institution_name: "Clinique les Hauts d'Anières",
            clinic_company_id: CLINIC_COMPANY_ID,
          },
        ],
      });
    }
    if (/\/bookings\/(\d+)\/dispute/.test(path) && req.method() === 'GET') {
      const bookingId = Number(path.match(/\/bookings\/(\d+)\/dispute/)[1]);
      return json(route, { data: disputePayload(bookingId) });
    }
    if (path.includes('/invoices/companies/') && path.includes('/invoices')) {
      return json(route, {
        data: [],
        invoices: [],
        pagination: { page: 1, pages: 1, total: 0 },
        stats: {
          total_issued: 0,
          total_paid: 0,
          total_balance: 0,
          overdue_count: 0,
        },
      });
    }
    if (path.includes('billing-opportunities') || path.includes('billable-partners')) {
      return json(route, { data: [] });
    }
    return json(route, { data: {} });
  });
}

async function injectCompanySession(page) {
  const token = fakeCompanyJwt();
  await page.addInitScript(
    ({ user, token: accessToken, publicId }) => {
      const serialized = JSON.stringify(user);
      localStorage.setItem('lirie_auth_env', 'app');
      localStorage.setItem('app_user', serialized);
      localStorage.setItem('user', serialized);
      localStorage.setItem('company_user', serialized);
      localStorage.setItem('app_public_id', publicId);
      localStorage.setItem('public_id', publicId);
      localStorage.setItem('company_public_id', publicId);
      localStorage.setItem('app_access_token', accessToken);
      localStorage.setItem('authToken', accessToken);
      localStorage.setItem('company_access_token', accessToken);
    },
    { user: COMPANY_USER, token, publicId: COMPANY_PUBLIC_ID }
  );
}

/**
 * G3 : visible ET dans le viewport. toBeAttached / toBeInTheDocument = FAIL.
 */
async function expectDisputeDialogInViewport(page, expect) {
  const overlay = page.getByTestId('dispute-resolution-overlay');
  const panel = page.getByTestId('dispute-resolution-panel');
  await expect(overlay).toBeVisible();
  await expect(panel).toBeVisible();
  await expect(panel).toBeInViewport({ ratio: 0.6 });
  await expect(overlay).toHaveAttribute('data-placement', 'viewport-fixed');

  const box = await panel.boundingBox();
  const viewport = page.viewportSize();
  if (!box || !viewport) {
    throw new Error('G3 FAIL : boundingBox ou viewport introuvable (toBeInTheDocument insuffisant).');
  }
  if (box.width <= 0 || box.height <= 0) {
    throw new Error('G3 FAIL : dialogue de taille nulle.');
  }
  const intersectsX = box.x < viewport.width && box.x + box.width > 0;
  const intersectsY = box.y < viewport.height && box.y + box.height > 0;
  if (!intersectsX || !intersectsY) {
    throw new Error(
      `G3 FAIL : dialogue hors viewport (box=${JSON.stringify(box)} vp=${JSON.stringify(viewport)}).`
    );
  }

  const geometry = await panel.evaluate((el) => {
    const scroll = document.querySelector('[data-testid="bill-period-form-scroll"]');
    const style = window.getComputedStyle(el.parentElement || el);
    return {
      insideInvoiceScroll: Boolean(scroll && scroll.contains(el)),
      overlayFixed: style.position === 'fixed',
    };
  });
  if (geometry.insideInvoiceScroll) {
    throw new Error('G3 FAIL : le dialogue est clipé dans le scroll facture.');
  }
  if (!geometry.overlayFixed) {
    throw new Error('G3 FAIL : overlay contestation n’est pas position:fixed.');
  }
}

async function openInstitutionInvoice(page, expect) {
  await page.goto(`/dashboard/company/${COMPANY_PUBLIC_ID}/invoices/clients`);
  await expect(page.getByRole('button', { name: 'Nouvelle facture' })).toBeVisible();
  await page.getByRole('button', { name: 'Nouvelle facture' }).click();
  await expect(page.getByRole('heading', { name: 'Nouvelle facture' })).toBeVisible();
  await page.getByTitle('Facture à une institution').click();
  await page.locator('#bill-period-ym').click();
  const monthDialog = page.getByRole('dialog', { name: /Choisir le mois/i });
  await monthDialog.getByRole('button', { name: 'Août' }).click();
  const clinicCombo = page.getByRole('combobox', { name: 'Institution' });
  await expect(clinicCombo).toBeEnabled();
  await clinicCombo.click();
  await page
    .getByRole('option', { name: /Clinique les Hauts d'Anières/i })
    .click();
  await expect(page.getByTestId('institution-summary-amount')).toBeVisible();
  await expect(page.getByTestId('institution-excluded-warning')).toBeVisible();
}

async function openExcludedBlock(page, expect) {
  await page.getByTestId('institution-excluded-toggle').click();
  await expect(page.getByTestId('institution-excluded-lines')).toBeVisible();
}

async function treatDispute(page, bookingId) {
  const button = page.getByTestId(`dispute-treat-${bookingId}`);
  await button.scrollIntoViewIfNeeded();
  await button.click();
}

async function scrollInvoice(page, where) {
  const scroller = page.getByTestId('bill-period-form-scroll');
  await scroller.evaluate((el, position) => {
    const max = Math.max(0, el.scrollHeight - el.clientHeight);
    if (position === 'top') el.scrollTop = 0;
    else if (position === 'middle') el.scrollTop = Math.round(max / 2);
    else el.scrollTop = max;
  }, where);
}

module.exports = {
  COMPANY_PUBLIC_ID,
  DUPONT_ID,
  KLEIN_ID,
  installApiMocks,
  injectCompanySession,
  expectDisputeDialogInViewport,
  openInstitutionInvoice,
  openExcludedBlock,
  treatDispute,
  scrollInvoice,
};
