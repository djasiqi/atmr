import {
  canTreatDispute,
  hasUploadedEvidence,
  institutionReasonComment,
  institutionReasonLabel,
  presentInstitutionDisputeReason,
  unwrapDisputePayload,
} from '../bookingDisputeUi';
import { exclusionWhyText } from '../institutionInvoicePlanUi';

describe('bookingDisputeUi', () => {
  it('autorise Traiter tant que la contestation est ouverte', () => {
    expect(
      canTreatDispute({
        invoiceBucket: 'disputed_blocked',
        validationStatus: 'disputed',
        disputeTreatable: true,
      })
    ).toBe(true);
    expect(
      canTreatDispute({
        invoiceBucket: 'other_excluded',
        validationStatus: 'not_billable',
        disputeStatus: 'disputed',
        disputeTreatable: true,
      })
    ).toBe(true);
    expect(
      canTreatDispute({
        invoiceBucket: 'other_excluded',
        disputeStatus: 'resolved_institution',
      })
    ).toBe(false);
  });

  it('exige une preuve uploadée, pas le snapshot système', () => {
    expect(
      hasUploadedEvidence({
        evidence: [{ source: 'system', kind: 'system_snapshot' }],
      })
    ).toBe(false);
    expect(
      hasUploadedEvidence({
        evidence: [{ source: 'uploaded', kind: 'signed_transport_sheet' }],
      })
    ).toBe(true);
  });

  it('explique le motif institution sans jargon ni liste de catégories', () => {
    expect(institutionReasonLabel('TRANSPORT_DISPUTED')).toBe('Course non reconnue');
    expect(institutionReasonLabel('OTHER')).toBe('Autre');
    expect(institutionReasonLabel('')).toBe('Autre');
    expect(institutionReasonLabel('')).not.toMatch(/mauvais payeur/);
    expect(
      institutionReasonComment('OTHER', 'OTHER: Pas de retour suite hospitalisation')
    ).toBe('Pas de retour suite hospitalisation');
    expect(
      presentInstitutionDisputeReason(
        'OTHER',
        'OTHER: Pas de retour suite hospitalisation'
      )
    ).toEqual({
      category: 'Autre',
      comment: 'Pas de retour suite hospitalisation',
    });
  });

  it('unwrappe la payload API success_response', () => {
    expect(unwrapDisputePayload({ data: { id: 3 } })).toEqual({ id: 3 });
  });

  it('adapte le texte d’exclusion selon l’étape', () => {
    expect(
      exclusionWhyText({
        invoiceBucket: 'other_excluded',
        validationStatus: 'not_billable',
        disputeStatus: 'disputed',
        disputeTreatable: true,
      })
    ).toMatch(/n'est pas résolue/);
    expect(
      exclusionWhyText({
        invoiceBucket: 'disputed_blocked',
        disputeStatus: 'evidence_submitted',
      })
    ).toMatch(/justificatif/i);
    expect(
      exclusionWhyText({
        exclusionReason: 'resolved_institution_not_billable',
        validationStatus: 'not_billable',
      })
    ).toMatch(/exclue définitivement/i);
  });
});
