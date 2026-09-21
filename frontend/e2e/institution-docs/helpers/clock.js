/**
 * Horloge documentation Institution — epoch figée.
 * À appeler avant toute navigation vers un écran daté (dashboard, notifs).
 */

const DOCS_EPOCH_ISO = '2026-03-16T10:00:00+01:00';
const DOCS_TIMEZONE = 'Europe/Zurich';
const DOCS_LOCALE = 'fr-CH';

/**
 * Fige Date.now() / new Date() à l'epoch docs, sans installer de fake timers
 * (les timers React / React Query doivent continuer à tourner).
 * @param {import('@playwright/test').Page} page
 */
async function freezeDocsClock(page) {
  const epochMs = new Date(DOCS_EPOCH_ISO).getTime();

  await page.addInitScript(({ frozen }) => {
    const RealDate = Date;
    class DocsDate extends RealDate {
      constructor(...args) {
        if (args.length === 0) {
          super(frozen);
        } else {
          super(...args);
        }
      }

      static now() {
        return frozen;
      }
    }
    DocsDate.parse = RealDate.parse.bind(RealDate);
    DocsDate.UTC = RealDate.UTC.bind(RealDate);
    window.Date = DocsDate;
  }, { frozen: epochMs });

  if (page.clock && typeof page.clock.setFixedTime === 'function') {
    await page.clock.setFixedTime(new Date(DOCS_EPOCH_ISO));
  }
}

module.exports = {
  DOCS_EPOCH_ISO,
  DOCS_TIMEZONE,
  DOCS_LOCALE,
  freezeDocsClock,
};
