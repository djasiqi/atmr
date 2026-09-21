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
async function hideNonDeterministic(page, extraSelectors = []) {
  const selectors = [...HIDE_SELECTORS, ...extraSelectors];
  await page.addStyleTag({
    content: `${selectors.join(', ')} { visibility: hidden !important; }`,
  });
}

const SCREENSHOT_OPTIONS = {
  animations: 'disabled',
  caret: 'hide',
  type: 'png',
};

function clampClip(clip, viewport) {
  const x = Math.max(0, Math.floor(clip.x));
  const y = Math.max(0, Math.floor(clip.y));
  const maxW = viewport.width - x;
  const maxH = viewport.height - y;
  const width = Math.max(1, Math.min(maxW, Math.ceil(clip.width)));
  const height = Math.max(1, Math.min(maxH, Math.ceil(clip.height)));
  if (width < 24 || height < 24) {
    throw new Error(
      `Clip trop petit (${width}x${height}) — cadrage documentaire invalide.`
    );
  }
  return { x, y, width, height };
}

/**
 * Union de boîtes Playwright (CSS px).
 * @param  {...{x:number,y:number,width:number,height:number}} boxes
 */
function unionBoxes(...boxes) {
  const valid = boxes.filter(Boolean);
  if (!valid.length) {
    throw new Error('Aucune boîte pour le cadrage.');
  }
  const x = Math.min(...valid.map((b) => b.x));
  const y = Math.min(...valid.map((b) => b.y));
  const right = Math.max(...valid.map((b) => b.x + b.width));
  const bottom = Math.max(...valid.map((b) => b.y + b.height));
  return { x, y, width: right - x, height: bottom - y };
}

async function requireBox(locator, label) {
  const box = await locator.boundingBox();
  if (!box) {
    throw new Error(`Boîte introuvable pour le cadrage: ${label}`);
  }
  return box;
}

/**
 * Retire le focus / curseur pour éviter les anneaux et tooltips.
 * @param {import('@playwright/test').Page} page
 */
async function prepareCaptureChrome(page) {
  await page.evaluate(() => {
    const active = document.activeElement;
    if (active && typeof active.blur === 'function') {
      active.blur();
    }
  });
  await page.mouse.move(0, 0);
}

/**
 * Capture une zone viewport (sans chrome navigateur).
 * @param {import('@playwright/test').Page} page
 * @param {{x:number,y:number,width:number,height:number}} clip
 * @param {string} filename
 */
async function captureClip(page, clip, filename) {
  await hideNonDeterministic(page);
  await prepareCaptureChrome(page);
  const dest = resolveCapturePath(filename);
  const viewport = page.viewportSize() || { width: 1440, height: 1000 };
  await page.screenshot({
    ...SCREENSHOT_OPTIONS,
    path: dest,
    fullPage: false,
    clip: clampClip(clip, viewport),
  });
  return dest;
}

/**
 * Portail compact : sidebar + header + contenu jusqu'au bas du dernier locator.
 * Coupe le vide sous les blocs (la sidebar fixed ferait sinon 100vh).
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator[]} contentLocators
 * @param {string} filename
 * @param {{ pad?: number }} [options]
 */
async function capturePortalUntil(page, contentLocators, filename, options = {}) {
  const pad = options.pad ?? 16;
  const sidebar = page.locator('[data-tour-id="institution-sidebar"]');
  const header = page.locator('main header').first();
  await expect(sidebar).toBeVisible();
  await expect(header).toBeVisible();

  const sidebarBox = await requireBox(sidebar, 'sidebar');
  const headerBox = await requireBox(header, 'header');
  const contentBoxes = [];
  for (const locator of contentLocators) {
    contentBoxes.push(await requireBox(locator, await locator.evaluate((el) => el.getAttribute('data-tour-id') || el.tagName)));
  }

  const contentUnion = unionBoxes(headerBox, ...contentBoxes);
  const clip = {
    x: 0,
    y: Math.min(sidebarBox.y, headerBox.y),
    width: Math.max(sidebarBox.x + sidebarBox.width, contentUnion.x + contentUnion.width),
    height: contentUnion.y + contentUnion.height + pad - Math.min(sidebarBox.y, headerBox.y),
  };
  return captureClip(page, clip, filename);
}

/**
 * Contenu principal Institution (sans sidebar) jusqu'au bas des blocs.
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator[]} contentLocators
 * @param {string} filename
 * @param {{ pad?: number }} [options]
 */
async function captureMainUntil(page, contentLocators, filename, options = {}) {
  const pad = options.pad ?? 16;
  const header = page.locator('main header').first();
  await expect(header).toBeVisible();
  const headerBox = await requireBox(header, 'header');
  const contentBoxes = [];
  for (const locator of contentLocators) {
    contentBoxes.push(await requireBox(locator, await locator.evaluate((el) => el.getAttribute('data-tour-id') || el.tagName)));
  }
  const contentUnion = unionBoxes(headerBox, ...contentBoxes);
  const clip = {
    x: headerBox.x,
    y: headerBox.y,
    width: (contentUnion.x + contentUnion.width) - headerBox.x,
    height: contentUnion.y + contentUnion.height + pad - headerBox.y,
  };
  return captureClip(page, clip, filename);
}

/**
 * Screenshot de l'élément complet (offsetWidth, même hors viewport) puis coupe verticale.
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator} element
 * @param {import('@playwright/test').Locator} lastContent
 * @param {string} filename
 * @param {{ pad?: number }} [options]
 */
async function captureElementUntil(page, element, lastContent, filename, options = {}) {
  const pad = options.pad ?? 16;
  await expect(element).toBeVisible();
  await expect(lastContent).toBeVisible();
  // Header sticky du layout : recouvre le titre du panneau si Playwright
  // fait défiler l'élément (hauteur > viewport). Masquage chrome uniquement.
  await hideNonDeterministic(page, ['main > header']);
  await prepareCaptureChrome(page);

  const layout = await element.evaluate((el) => {
    const rect = el.getBoundingClientRect();
    return {
      x: rect.x,
      y: rect.y,
      offsetWidth: el.offsetWidth,
      offsetHeight: el.offsetHeight,
      clientWidth: el.clientWidth,
    };
  });
  const lastBox = await requireBox(lastContent, 'dernier contenu');
  const cropCss = lastBox.y + lastBox.height + pad - layout.y;

  const buffer = await element.screenshot({
    ...SCREENSHOT_OPTIONS,
  });
  const pngWidth = buffer.readUInt32BE(16);
  const pngHeight = buffer.readUInt32BE(20);
  const scale = pngWidth / layout.offsetWidth;
  const cropPx = Math.max(1, Math.min(pngHeight, Math.round(cropCss * scale)));

  const dest = resolveCapturePath(filename);
  if (cropPx >= pngHeight) {
    fs.writeFileSync(dest, buffer);
    return dest;
  }

  const croppedB64 = await page.evaluate(async ({ b64, height }) => {
    const res = await fetch(`data:image/png;base64,${b64}`);
    const blob = await res.blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Canvas 2D indisponible pour le cadrage documentaire.');
    }
    ctx.drawImage(bitmap, 0, 0);
    return canvas.toDataURL('image/png').replace(/^data:image\/png;base64,/, '');
  }, { b64: buffer.toString('base64'), height: cropPx });

  fs.writeFileSync(dest, Buffer.from(croppedB64, 'base64'));
  return dest;
}

/**
 * Colonne liste : haut du locator jusqu'au bas d'un contenu (évite le min-height 100vh).
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator} column
 * @param {import('@playwright/test').Locator} lastContent
 * @param {string} filename
 * @param {{ pad?: number }} [options]
 */
async function captureColumnUntil(page, column, lastContent, filename, options = {}) {
  const lastBox = await requireBox(lastContent, 'dernier contenu');
  return captureColumnUntilBox(page, column, lastBox, filename, options);
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator} column
 * @param {{x:number,y:number,width:number,height:number}} lastBox
 * @param {string} filename
 * @param {{ pad?: number }} [options]
 */
async function captureColumnUntilBox(page, column, lastBox, filename, options = {}) {
  const pad = options.pad ?? 16;
  const columnBox = await requireBox(column, 'colonne');
  const clip = {
    x: columnBox.x,
    y: columnBox.y,
    width: columnBox.width,
    height: lastBox.y + lastBox.height + pad - columnBox.y,
  };
  return captureClip(page, clip, filename);
}

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
  captureClip,
  capturePortalUntil,
  captureMainUntil,
  captureElementUntil,
  captureColumnUntil,
  captureColumnUntilBox,
  unionBoxes,
  clampClip,
};
