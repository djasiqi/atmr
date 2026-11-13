const fs = require('fs');
const path = require('path');
const { expect } = require('@playwright/test');

const ACCOUNTS_PATH =
  process.env.RECETTE_ACCOUNTS_JSON ||
  path.join(
    process.env.TEMP || process.env.TMP || 'C:/Users/jasiq/AppData/Local/Temp',
    'lirie-recette-interfaces-e2e',
    'accounts.json'
  );

function loadAccounts() {
  const raw = fs.readFileSync(ACCOUNTS_PATH, 'utf8');
  return JSON.parse(raw);
}

async function login(page, email, password) {
  await page.goto('/login');
  await page.evaluate(() => {
    try {
      localStorage.clear();
      sessionStorage.clear();
    } catch (_) {
      /* ignore */
    }
  });
  await page.context().clearCookies();
  await page.goto('/login', { waitUntil: 'domcontentloaded' });
  const emailInput = page.locator('input#email');
  await expect(emailInput).toBeVisible({ timeout: 60000 });
  await emailInput.fill(email);
  await page.locator('input#password').fill(password);
  await page.getByRole('button', { name: /Se connecter/i }).click();
}

async function loginExpectUrl(page, email, password, urlRe) {
  await login(page, email, password);
  try {
    await expect(page).toHaveURL(urlRe, { timeout: 60000 });
  } catch (err) {
    const alertText =
      (await page.getByRole('alert').textContent().catch(() => '')) || '';
    throw new Error(
      `Login échoué pour ${email} (URL=${page.url()}). ${alertText} ${err.message}`
    );
  }
}

module.exports = {
  loadAccounts,
  login,
  loginExpectUrl,
};
