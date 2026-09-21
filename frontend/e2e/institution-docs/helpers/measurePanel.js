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

/**
 * Dimensions CSS du modal Nouvelle demande (Lot 4).
 * @param {import('@playwright/test').Page} page
 */
async function measureCreateModal(page) {
  return page.evaluate(() => {
    const form = document.querySelector('[data-tour-id="institution-request-create"]');
    const dialog = form && form.closest('.modal-content');
    const scroller = form && form.querySelector('form > div');
    if (!form || !dialog) {
      return { found: false };
    }
    const formRect = form.getBoundingClientRect();
    const dialogRect = dialog.getBoundingClientRect();
    return {
      found: true,
      dialog: {
        cssWidth: Math.round(dialogRect.width),
        cssHeight: Math.round(dialogRect.height),
        offsetWidth: dialog.offsetWidth,
        offsetHeight: dialog.offsetHeight,
      },
      form: {
        cssWidth: Math.round(formRect.width),
        cssHeight: Math.round(formRect.height),
        offsetWidth: form.offsetWidth,
        offsetHeight: form.offsetHeight,
      },
      scroller: scroller
        ? {
            clientHeight: scroller.clientHeight,
            scrollHeight: scroller.scrollHeight,
            scrollTop: scroller.scrollTop,
          }
        : null,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
    };
  });
}

module.exports = {
  measureDetailPanel,
  measureTextOverflow,
  measureCreateModal,
};
