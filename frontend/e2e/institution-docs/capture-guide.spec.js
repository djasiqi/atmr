const { test, expect } = require('@playwright/test');
const { inspectCapture } = require('./helpers/inspectCapture');

/**
 * Hashes figés Lot 3.1 — ne plus régénérer ces PNG dans le Lot 4.
 */
const FROZEN_LOT3 = {
  '01-dashboard.png':
    'd343018c2be8551af2f2df09f498572e896f2b93d854ad9626d1780710c99a75',
  '07-transport-list.png':
    '55b1959a269d06ab7030506199b944ebae8237e348479048af791e2d89c9de22',
  '08-transport-detail.png':
    '70db785caeae7683b09d7f10fc68eccf54a57e3329a815bc84245944514cf449',
  '10-patients-list.png':
    'b5978fa7336c44c36412ba02feac6c6342d826008acdf51babce998a801fdbad',
};

test.describe('Captures documentation Institution (Lot 3.1 figé)', () => {
  test('01 / 07 / 08 / 10 restent inchangées', () => {
    for (const [filename, sha256] of Object.entries(FROZEN_LOT3)) {
      const report = inspectCapture(filename);
      expect(report.sha256, filename).toBe(sha256);
    }
  });
});
