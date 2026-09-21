const fs = require('fs');
const path = require('path');
const { CAPTURES_DIR } = require('./screenshot');

const OFFICIAL_LOT3 = [
  '01-dashboard.png',
  '07-transport-list.png',
  '08-transport-detail.png',
  '10-patients-list.png',
];

const OFFICIAL_LOT4 = [
  '02-new-request-patient.png',
  '03-new-request-delivery.png',
  '04-new-request-external.png',
];

const OFFICIAL_GUIDE = [
  '01-dashboard.png',
  '02-new-request-patient.png',
  '03-new-request-delivery.png',
  '04-new-request-external.png',
  '07-transport-list.png',
  '08-transport-detail.png',
  '10-patients-list.png',
];

/**
 * Planche temporaire — non officielle, gitignorée.
 * @param {import('@playwright/test').Page} page
 * @param {string[]} filenames
 * @param {string} destName
 * @param {{ columns?: number, imageHeight?: number, viewport?: {width:number,height:number} }} [options]
 */
async function captureReviewSheetFrom(
  page,
  filenames,
  destName,
  options = {}
) {
  const columns = options.columns ?? 2;
  const imageHeight = options.imageHeight ?? 420;
  const viewport = options.viewport ?? { width: 1600, height: 980 };

  const cells = filenames.map((name) => {
    const filePath = path.join(CAPTURES_DIR, name);
    if (!fs.existsSync(filePath)) {
      throw new Error(`Capture officielle manquante pour la planche: ${name}`);
    }
    const b64 = fs.readFileSync(filePath).toString('base64');
    return { name, b64 };
  });

  const html = `<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <style>
    html, body { margin: 0; padding: 0; background: #0f172a; font-family: system-ui, sans-serif; }
    .grid { display: grid; grid-template-columns: repeat(${columns}, 1fr); gap: 16px; padding: 20px; }
    figure { margin: 0; background: #1e293b; border-radius: 8px; overflow: hidden; }
    figcaption { color: #e2e8f0; font-size: 13px; font-weight: 600; padding: 8px 12px; }
    img { display: block; width: 100%; height: ${imageHeight}px; object-fit: contain; background: #f8fafc; }
  </style>
</head>
<body>
  <div class="grid">
    ${cells.map((cell) => `
      <figure>
        <figcaption>${cell.name}</figcaption>
        <img src="data:image/png;base64,${cell.b64}" alt="${cell.name}" />
      </figure>
    `).join('')}
  </div>
</body>
</html>`;

  const previous = page.viewportSize();
  await page.setViewportSize(viewport);
  await page.setContent(html, { waitUntil: 'domcontentloaded' });
  const dest = path.join(CAPTURES_DIR, destName);
  await page.screenshot({
    path: dest,
    type: 'png',
    animations: 'disabled',
    fullPage: false,
  });
  if (previous) {
    await page.setViewportSize(previous);
  }
  return dest;
}

/**
 * Planche temporaire 2x2 — non officielle, gitignorée.
 * @param {import('@playwright/test').Page} page
 * @param {string} [filename]
 */
async function captureReviewSheet(page, filename = '_artifact-review-lot3.png') {
  return captureReviewSheetFrom(page, OFFICIAL_LOT3, filename);
}

/**
 * @param {import('@playwright/test').Page} page
 */
async function captureLot4ReviewSheet(page) {
  return captureReviewSheetFrom(
    page,
    OFFICIAL_LOT4,
    '_artifact-review-lot4.png',
    { columns: 3, imageHeight: 520, viewport: { width: 1800, height: 720 } }
  );
}

/**
 * @param {import('@playwright/test').Page} page
 */
async function captureGuideReviewSheet(page) {
  return captureReviewSheetFrom(
    page,
    OFFICIAL_GUIDE,
    '_artifact-review-guide.png',
    { columns: 4, imageHeight: 280, viewport: { width: 1800, height: 820 } }
  );
}

module.exports = {
  OFFICIAL_LOT3,
  OFFICIAL_LOT4,
  OFFICIAL_GUIDE,
  captureReviewSheet,
  captureReviewSheetFrom,
  captureLot4ReviewSheet,
  captureGuideReviewSheet,
};
