const fs = require('fs');
const path = require('path');
const { expect } = require('@playwright/test');

const CAPTURES_DIR = path.resolve(
  __dirname,
  '../../../../docs/guides/institution/captures'
);

/** Uniquement les éléments non déterministes documentés — jamais l'UI fonctionnelle. */
const HIDE_SELECTORS = [
  '[data-sonner-toaster]',
  '[data-sonner-toast]',
];

function resolveCapturePath(filename) {
  if (!filename || typeof filename !== 'string') {
    throw new Error('Nom de fichier de capture manquant.');
  }
  const base = path.basename(filename);
  if (base !== filename || filename.includes('..')) {
    throw new Error(`Nom de capture invalide: ${filename}`);
  }
  if (!filename.toLowerCase().endsWith('.png')) {
    throw new Error(`La capture doit être un PNG: ${filename}`);
  }
  fs.mkdirSync(CAPTURES_DIR, { recursive: true });
  return path.join(CAPTURES_DIR, filename);
}

/**
 * Masque toaster / toasts Sonner sans toucher aux contrôles LIRIE.
 * @param {import('@playwright/test').Page} page
 */
async function hideNonDeterministic(page) {
  await page.addStyleTag({
    content: `${HIDE_SELECTORS.join(', ')} { visibility: hidden !important; }`,
  });
}

const SCREENSHOT_OPTIONS = {
  animations: 'disabled',
  caret: 'hide',
  type: 'png',
};

/**
 * Capture un composant visible (préféré pour le guide).
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator} locator
 * @param {string} filename
 */
async function captureLocator(page, locator, filename) {
  await expect(locator, `Cible de capture invisible: ${filename}`).toBeVisible();
  await hideNonDeterministic(page);
  const dest = resolveCapturePath(filename);
  await locator.screenshot({
    ...SCREENSHOT_OPTIONS,
    path: dest,
  });
  return dest;
}

/**
 * Capture le viewport page (sans chrome navigateur).
 * @param {import('@playwright/test').Page} page
 * @param {string} filename
 */
async function capturePage(page, filename) {
  await hideNonDeterministic(page);
  const dest = resolveCapturePath(filename);
  await page.screenshot({
    ...SCREENSHOT_OPTIONS,
    path: dest,
    fullPage: false,
  });
  return dest;
}

module.exports = {
  CAPTURES_DIR,
  HIDE_SELECTORS,
  captureLocator,
  capturePage,
};
