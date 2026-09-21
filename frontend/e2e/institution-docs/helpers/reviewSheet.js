const fs = require('fs');
const path = require('path');
const { CAPTURES_DIR } = require('./screenshot');

const OFFICIAL_LOT3 = [
  '01-dashboard.png',
  '07-transport-list.png',
  '08-transport-detail.png',
  '10-patients-list.png',
];

/**
 * Planche temporaire 2x2 — non officielle, gitignorée.
 * @param {import('@playwright/test').Page} page
 * @param {string} [filename]
 */
async function captureReviewSheet(page, filename = '_artifact-review-lot3.png') {
  const cells = OFFICIAL_LOT3.map((name) => {
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
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px; }
    figure { margin: 0; background: #1e293b; border-radius: 8px; overflow: hidden; }
    figcaption { color: #e2e8f0; font-size: 13px; font-weight: 600; padding: 8px 12px; }
    img { display: block; width: 100%; height: 420px; object-fit: contain; background: #f8fafc; }
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
  await page.setViewportSize({ width: 1600, height: 980 });
  await page.setContent(html, { waitUntil: 'domcontentloaded' });
  const dest = path.join(CAPTURES_DIR, filename);
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

module.exports = {
  OFFICIAL_LOT3,
  captureReviewSheet,
};
