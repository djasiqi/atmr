/**
 * Mesure les locators du panneau détail (Lot 3.1) sans modifier le produit.
 * @param {import('@playwright/test').Page} page
 */
async function measureDetailPanel(page) {
  const nodes = page.locator('[data-tour-id="institution-request-detail-panel"]');
  const count = await nodes.count();
  const items = [];
  for (let i = 0; i < count; i += 1) {
    const locator = nodes.nth(i);
    const box = await locator.boundingBox();
    const layout = await locator.evaluate((el) => ({
      tag: el.tagName,
      className: el.className,
      offsetWidth: el.offsetWidth,
      clientWidth: el.clientWidth,
      parentClass: el.parentElement ? el.parentElement.className : null,
      parentOffsetWidth: el.parentElement ? el.parentElement.offsetWidth : null,
    }));
    items.push({
      index: i,
      boundingBox: box,
      ...layout,
    });
  }
  return items;
}

/**
 * Overflow réel d'un texte dans le panneau (scrollWidth > clientWidth).
 * @param {import('@playwright/test').Locator} scope
 * @param {string} text
 */
async function measureTextOverflow(scope, text) {
  const el = scope.getByText(text, { exact: false }).first();
  if (await el.count() === 0) {
    return { text, found: false };
  }
  return el.evaluate((node, expected) => {
    const target = node.textContent && node.textContent.includes(expected)
      ? node
      : node;
    return {
      text: expected,
      found: true,
      content: (target.textContent || '').trim(),
      clientWidth: target.clientWidth,
      scrollWidth: target.scrollWidth,
      overflows: target.scrollWidth > target.clientWidth + 1,
    };
  }, text);
}

module.exports = {
  measureDetailPanel,
  measureTextOverflow,
};
