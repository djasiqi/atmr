/**
 * @deprecated Le parcours UI utilise InvoiceCorrectionWorkbench dans l’éditeur unique.
 * Les tests métier sont dans InvoiceCorrectionWorkbench.test.jsx.
 */
describe('InvoiceCorrectionModal (pont déprécié)', () => {
  it('réexporte le workbench intégré', async () => {
    const mod = await import('./InvoiceCorrectionModal');
    const workbench = await import('./InvoiceCorrectionWorkbench');
    expect(mod.default).toBe(workbench.default);
  });
});
