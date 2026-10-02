import {
  presentPartnerInvoiceSummary,
  presentPatientInvoiceSummary,
} from '../payerInvoiceSummaryUi';

describe('payerInvoiceSummaryUi', () => {
  it('résumé Patient = uniquement ce que ce patient doit payer', () => {
    const summary = presentPatientInvoiceSummary({
      display_name: 'Charlotte CAVADINI',
      transports_count: 1,
      unbilled_total_amount: 40,
      can_generate: true,
    });
    expect(summary).toMatchObject({
      visible: true,
      hasBillable: true,
      displayName: 'Charlotte CAVADINI',
      transportsCount: 1,
      totalHt: 40,
      blocked: false,
    });
  });

  it('patient à compléter : visible, non facturable', () => {
    const summary = presentPatientInvoiceSummary({
      display_name: 'Alice MARTIN',
      segments_count: 2,
      unbilled_total_amount: 80,
      can_generate: false,
    });
    expect(summary.hasBillable).toBe(false);
    expect(summary.blocked).toBe(true);
    expect(summary.gateHeld).toBe(false);
    expect(summary.transportsCount).toBe(2);
    expect(summary.emptyNote).toBe('Identité ou destinataire à compléter avant facturation.');
  });

  it('course institution en attente de validation : retenue expliquée, pas « à compléter »', () => {
    // Cas réel : #40543 SBORDONE Anna, 30.09.2026, payeur patient, origine CLHDA (Market LIRIE).
    const summary = presentPatientInvoiceSummary({
      display_name: 'SBORDONE Anna',
      segments_count: 0,
      unbilled_total_amount: 0,
      can_generate: false,
      blocked_reason: 'pending_institution_validation',
      pending_validation_count: 1,
      pending_validation_amount: 45,
      // 01.10.2026 00:00 Europe/Zurich
      pending_validation_release_at: '2026-10-01T00:00:00+02:00',
    });
    expect(summary.hasBillable).toBe(false);
    expect(summary.blocked).toBe(false);
    expect(summary.gateHeld).toBe(true);
    expect(summary.blockedReason).toBe('pending_institution_validation');
    expect(summary.pendingValidation).toMatchObject({
      visible: true,
      count: 1,
      amountHt: 45,
      releaseAtLabel: '01.10.2026',
    });
    expect(summary.emptyNote).toBe(
      "1 prestation en attente de validation par l'institution — facturable après validation ou automatiquement dès le 01.10.2026."
    );
    expect(summary.pendingNote).toBe('');
  });

  it('cas mixte : facturable maintenant + retenue institution signalée hors total', () => {
    const summary = presentPatientInvoiceSummary({
      display_name: 'Jean DUPONT',
      segments_count: 1,
      unbilled_total_amount: 40,
      can_generate: true,
      pending_validation_count: 2,
      pending_validation_amount: 90,
      pending_validation_release_at: '2026-10-01T00:00:00+02:00',
    });
    expect(summary.hasBillable).toBe(true);
    expect(summary.transportsCount).toBe(1);
    expect(summary.totalHt).toBe(40);
    expect(summary.emptyNote).toBe('');
    expect(summary.pendingNote).toBe(
      "2 prestations en attente de validation par l'institution — facturables après validation ou automatiquement dès le 01.10.2026 — non incluses dans cette facture."
    );
  });

  it('course contestée par l’institution : message dédié', () => {
    const summary = presentPatientInvoiceSummary({
      display_name: 'Marie CURIE',
      segments_count: 0,
      unbilled_total_amount: 0,
      can_generate: false,
      blocked_reason: 'disputed',
      disputed_count: 1,
    });
    expect(summary.hasBillable).toBe(false);
    expect(summary.gateHeld).toBe(true);
    expect(summary.blocked).toBe(false);
    expect(summary.emptyNote).toBe(
      "1 prestation contestée par l'institution — à traiter avant facturation."
    );
  });

  it('résumé Partenaire = uniquement les transferts validés de ce partenaire', () => {
    const summary = presentPartnerInvoiceSummary({
      partner_company_name: 'Partenaire Test',
      validated_unbilled_transfers_count: 4,
      unbilled_transfers_count: 6,
      estimated_subtotal_ht: 160,
    });
    expect(summary).toMatchObject({
      visible: true,
      hasBillable: true,
      transportsCount: 4,
      totalHt: 160,
    });
    expect(summary.excluded).toMatchObject({ visible: true, count: 2 });
  });
});
