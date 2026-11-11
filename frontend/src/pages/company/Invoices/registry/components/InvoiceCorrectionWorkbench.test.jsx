/**
 * @jest-environment jsdom
 */

describe('InvoiceCorrectionWorkbench (déprécié)', () => {
  it('réexporte la boîte de confirmation ponctuelle', async () => {
    const wb = await import('./InvoiceCorrectionWorkbench');
    const dialog = await import('./InvoiceCorrectionConfirmDialog');
    expect(wb.default).toBe(dialog.default);
  });
});
