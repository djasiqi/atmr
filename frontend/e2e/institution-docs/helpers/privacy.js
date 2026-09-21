const { expect } = require('@playwright/test');

const DOCS_EMAIL_DOMAIN = 'docs.lirie.local';
const FORBIDDEN_EMAIL = /[A-Z0-9._%+-]+@(?!docs\.lirie\.local\b)[A-Z0-9.-]+\.[A-Z]{2,}/i;

/**
 * Vérifie que la vue visible ne contient que le tenant documentation.
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').Locator} [root]
 */
async function assertDocsTenantOnly(page, root) {
  const scope = root || page.locator('body');
  const text = (await scope.innerText()).replace(/\s+/g, ' ');

  const leaked = text.match(FORBIDDEN_EMAIL);
  expect(
    leaked,
    `E-mail hors @${DOCS_EMAIL_DOMAIN} visible: ${leaked && leaked[0]}`
  ).toBeNull();

  expect(text, 'Donnée d’un autre tenant local suspecte').not.toMatch(
    /lirie\.ch|gmail\.com|hotmail\.com|bluewin\.ch/i
  );
}

module.exports = {
  DOCS_EMAIL_DOMAIN,
  FORBIDDEN_EMAIL,
  assertDocsTenantOnly,
};
