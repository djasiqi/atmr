import {
  INSTITUTION_PLAN_POLL_MS,
  bindInstitutionPlanLiveRefresh,
  clinicMonthlyPreparePayload,
  draftInvoiceFromPrepareError,
  draftPdfActionsAvailability,
  institutionPlanScopeKey,
  isInstitutionPlanCurrent,
  isInstitutionPlanFetchCanceled,
  nextInstitutionPlanRequestId,
  presentBillPeriodComposerUi,
  shouldApplyInstitutionPlanResponse,
  shouldKeepPreviousInstitutionPlan,
  shouldPollInstitutionPlan,
  shouldShowDraftInvoiceToolbar,
  shouldShowInstitutionPlanSkeleton,
  shouldShowSimpleInvoiceLinesPreview,
  unwrapPreparedDraftInvoice,
} from '../institutionInvoicePlanLiveSync';
import {
  draftInvoicePlanParity,
  exclusionWhyText,
  flattenInstitutionPreviewBookingIds,
  institutionSurfaceParity,
  institutionSurfacesFromPlan,
} from '../institutionInvoicePlanUi';
import { presentInvoiceLinesPreview } from '../invoiceLinesPreviewUi';

const clinicBooking = (partial) => ({
  origin: 'LIRIE_MARKETPLACE',
  validation_status: 'validated',
  payer: 'clinic',
  eligible: true,
  invoice_bucket: 'clinic_billable',
  group_id: null,
  amount_ht: 40,
  ...partial,
});

const livePlan = {
  clinic: {
    display_name: 'Clinique les Hauts d’Anières',
    transports_count: 7,
    estimated_total: 280,
    booking_ids: [45697, 45702, 45703, 45704, 45705, 45706, 45707],
  },
  patients: [],
  partners: [],
  reconciliation: {
    buckets: {
      clinic_billable: {
        count: 7,
        amount_ht: 280,
        booking_ids: [45697, 45702, 45703, 45704, 45705, 45706, 45707],
      },
      patient_billable: { count: 0, amount_ht: 0, booking_ids: [] },
      pending_blocked: { count: 0, amount_ht: 0, booking_ids: [] },
      disputed_blocked: { count: 1, amount_ht: 40, booking_ids: [45700] },
      already_invoiced: { count: 0, amount_ht: 0, booking_ids: [] },
    },
    bookings: [
      clinicBooking({
        booking_id: 45697,
        patient_name: 'Alice MARTIN',
        scheduled_at: '2026-08-02T08:00:00',
      }),
      clinicBooking({
        booking_id: 45702,
        patient_name: 'Arturo KLEIN',
        scheduled_at: '2026-08-15T08:00:00',
        group_id: 'request_id:2271',
      }),
      clinicBooking({
        booking_id: 45703,
        patient_name: 'Arturo KLEIN',
        scheduled_at: '2026-08-15T10:00:00',
        group_id: 'parent_booking_id:45702',
        grouping_relation: 'parent_booking_id',
      }),
      clinicBooking({
        booking_id: 45704,
        patient_name: 'Marie DUPONT',
        scheduled_at: '2026-08-16T08:00:00',
        group_id: 'request_id:2272',
      }),
      clinicBooking({
        booking_id: 45705,
        patient_name: 'Marie DUPONT',
        scheduled_at: '2026-08-16T16:00:00',
        group_id: 'parent_booking_id:45704',
        grouping_relation: 'parent_booking_id',
      }),
      clinicBooking({
        booking_id: 45706,
        patient_name: 'Jacques BARBEY',
        scheduled_at: '2026-08-20T08:00:00',
      }),
      clinicBooking({
        booking_id: 45707,
        patient_name: 'Jacques BARBEY',
        scheduled_at: '2026-08-22T08:00:00',
      }),
      {
        booking_id: 45700,
        origin: 'LIRIE_MARKETPLACE',
        validation_status: 'disputed',
        payer: 'clinic',
        eligible: false,
        invoice_bucket: 'disputed_blocked',
        amount_ht: 40,
        patient_name: 'Alice MARTIN',
        scheduled_at: '2026-08-12T08:00:00',
      },
    ],
  },
};

const moveClinicBookingToPatient = (plan, bookingId) => {
  const id = Number(bookingId);
  const row = (plan.reconciliation.bookings || []).find(
    (item) => Number(item.booking_id) === id
  );
  const amount = Number(row?.amount_ht) || 0;
  const clinicIds = plan.reconciliation.buckets.clinic_billable.booking_ids.filter(
    (item) => Number(item) !== id
  );
  return {
    ...plan,
    clinic: {
      ...plan.clinic,
      transports_count: clinicIds.length,
      estimated_total: Number(plan.clinic.estimated_total) - amount,
      booking_ids: clinicIds,
    },
    reconciliation: {
      ...plan.reconciliation,
      buckets: {
        ...plan.reconciliation.buckets,
        clinic_billable: {
          count: clinicIds.length,
          amount_ht: Number(plan.reconciliation.buckets.clinic_billable.amount_ht) - amount,
          booking_ids: clinicIds,
        },
        patient_billable: {
          count: (plan.reconciliation.buckets.patient_billable.count || 0) + 1,
          amount_ht:
            (Number(plan.reconciliation.buckets.patient_billable.amount_ht) || 0) + amount,
          booking_ids: [
            ...(plan.reconciliation.buckets.patient_billable.booking_ids || []),
            id,
          ],
        },
      },
      bookings: (plan.reconciliation.bookings || []).map((item) =>
        Number(item.booking_id) === id
          ? {
              ...item,
              payer: 'patient',
              invoice_bucket: 'patient_billable',
            }
          : item
      ),
    },
  };
};

describe('institutionInvoicePlanLiveSync — gate live payeur', () => {
  it('UI-1 : résumé et lignes ont les mêmes booking_ids source', () => {
    const { summary, lineBookingIds } = institutionSurfacesFromPlan(livePlan);
    expect([...summary.bookingIds].sort((a, b) => a - b)).toEqual(
      [...lineBookingIds].sort((a, b) => a - b)
    );
    expect(institutionSurfaceParity(livePlan).bookingIdsMatch).toBe(true);
  });

  it('UI-2 : Clinique → Patient retire la course du résumé et des lignes', () => {
    const before = institutionSurfacesFromPlan(livePlan);
    const next = moveClinicBookingToPatient(livePlan, 45697);
    const after = institutionSurfacesFromPlan(next);
    expect(after.summary.transportsCount).toBe(before.summary.transportsCount - 1);
    expect(after.summary.totalHt).toBe(before.summary.totalHt - 40);
    expect(after.lineBookingIds).not.toContain(45697);
    expect(after.summary.bookingIds).not.toContain(45697);
  });

  it('UI-3 : Patient → Clinique réintroduit la course dans les deux surfaces', () => {
    const patientPlan = moveClinicBookingToPatient(livePlan, 45697);
    const restored = institutionSurfacesFromPlan(livePlan);
    const afterPatient = institutionSurfacesFromPlan(patientPlan);
    expect(afterPatient.lineBookingIds).not.toContain(45697);
    expect(restored.lineBookingIds).toContain(45697);
    expect(restored.summary.transportsCount).toBe(afterPatient.summary.transportsCount + 1);
    expect(restored.summary.totalHt).toBe(afterPatient.summary.totalHt + 40);
  });

  it('UI-4 : preview ouverte pendant refresh — pas de ligne stale', () => {
    const showLinesPreview = true;
    const first = institutionSurfacesFromPlan(livePlan);
    const nextPlan = moveClinicBookingToPatient(livePlan, 45697);
    const second = institutionSurfacesFromPlan(nextPlan);
    expect(showLinesPreview).toBe(true);
    expect(first.lineBookingIds).toContain(45697);
    expect(second.lineBookingIds).not.toContain(45697);
    expect(second.lines.some((line) => line.booking_ids.includes(45697))).toBe(false);
  });

  it('UI-5 : contestée hors résumé, lignes et draft', () => {
    const { summary, lines, excluded, lineBookingIds } =
      institutionSurfacesFromPlan(livePlan);
    expect(summary.bookingIds).not.toContain(45700);
    expect(lineBookingIds).not.toContain(45700);
    expect(excluded.map((row) => row.bookingId)).toEqual([45700]);
    expect(exclusionWhyText(excluded[0])).toContain('contesté');
    expect(lines.every((line) => !line.booking_ids.includes(45700))).toBe(true);
  });

  it('UI-6 : événement socket déclenche un refresh du plan', () => {
    const reasons = [];
    const socket = {
      on: jest.fn(),
      off: jest.fn(),
    };
    const unbind = bindInstitutionPlanLiveRefresh({
      socket,
      refresh: (meta) => reasons.push(meta.reason),
      documentRef: { visibilityState: 'hidden', addEventListener: jest.fn(), removeEventListener: jest.fn() },
      setIntervalFn: () => 1,
      clearIntervalFn: jest.fn(),
    });
    expect(socket.on).toHaveBeenCalledWith('booking_updated', expect.any(Function));
    socket.on.mock.calls[0][1]();
    expect(reasons).toEqual(['socket']);
    unbind();
  });

  it('UI-7 : visibilitychange visible relance le plan', () => {
    const reasons = [];
    const listeners = {};
    const documentRef = {
      visibilityState: 'hidden',
      addEventListener: (event, fn) => {
        listeners[event] = fn;
      },
      removeEventListener: jest.fn(),
    };
    const unbind = bindInstitutionPlanLiveRefresh({
      refresh: (meta) => reasons.push(meta.reason),
      documentRef,
      setIntervalFn: () => 1,
      clearIntervalFn: jest.fn(),
    });
    documentRef.visibilityState = 'visible';
    listeners.visibilitychange();
    expect(reasons).toContain('visibility');
    unbind();
  });

  it('UI-8 : fallback polling met à jour sans socket', () => {
    const reasons = [];
    const timers = [];
    const unbind = bindInstitutionPlanLiveRefresh({
      refresh: (meta) => reasons.push(meta.reason),
      pollMs: 25,
      documentRef: {
        visibilityState: 'visible',
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
      },
      setIntervalFn: (fn) => {
        timers.push(fn);
        return 7;
      },
      clearIntervalFn: jest.fn(),
    });
    expect(shouldPollInstitutionPlan({
      open: true,
      payerType: 'clinic',
      clinicCompanyId: 1,
      visibilityState: 'visible',
    })).toBe(true);
    expect(shouldPollInstitutionPlan({
      open: true,
      payerType: 'clinic',
      clinicCompanyId: 1,
      visibilityState: 'hidden',
    })).toBe(false);
    timers[0]();
    expect(reasons).toEqual(['poll']);
    unbind();
  });

  it('UI-9 : fermeture / réouverture n’empile pas les écouteurs', () => {
    const socket = { on: jest.fn(), off: jest.fn() };
    const documentRef = {
      visibilityState: 'visible',
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    };
    const clearIntervalFn = jest.fn();
    const first = bindInstitutionPlanLiveRefresh({
      socket,
      refresh: () => {},
      documentRef,
      setIntervalFn: () => 11,
      clearIntervalFn,
    });
    first();
    const second = bindInstitutionPlanLiveRefresh({
      socket,
      refresh: () => {},
      documentRef,
      setIntervalFn: () => 12,
      clearIntervalFn,
    });
    second();
    expect(socket.on).toHaveBeenCalledTimes(2);
    expect(socket.off).toHaveBeenCalledTimes(2);
    expect(documentRef.removeEventListener).toHaveBeenCalledTimes(2);
    expect(clearIntervalFn).toHaveBeenCalledTimes(2);
  });

  it('UI-10 : une réponse ancienne n’écrase pas la plus récente', () => {
    let latest = 0;
    let current = null;
    const apply = (reqId, plan) => {
      latest = Math.max(latest, reqId);
      if (!shouldApplyInstitutionPlanResponse(reqId, latest)) return;
      current = plan;
    };
    const firstId = nextInstitutionPlanRequestId(0);
    const secondId = nextInstitutionPlanRequestId(firstId);
    apply(firstId, livePlan);
    apply(secondId, moveClinicBookingToPatient(livePlan, 45697));
    apply(firstId, livePlan);
    expect(current.clinic.transports_count).toBe(6);
    expect(shouldApplyInstitutionPlanResponse(firstId, secondId)).toBe(false);
  });

  it('UI-11 : A/R affiche M lignes ≠ N prestations', () => {
    const { summary, lines } = institutionSurfacesFromPlan(livePlan);
    const preview = presentInvoiceLinesPreview(lines, {
      prestationCount: summary.transportsCount,
      totalHt: summary.totalHt,
    });
    expect(preview.visualLineCount).toBe(5);
    expect(preview.prestationCount).toBe(7);
    expect(preview.visualLineCount).toBeLessThan(preview.prestationCount);
    expect(lines.filter((line) => line.unit_type === 'round_trip')).toHaveLength(2);
  });

  it('UI-12 : Prepare sans Preview n’envoie pas de booking_ids', () => {
    const payload = clinicMonthlyPreparePayload({
      clinicCompanyId: 1,
      periodYear: 2026,
      periodMonth: 8,
    });
    expect(payload).toEqual({
      mode: 'clinic_monthly',
      clinic_company_id: 1,
      period_year: 2026,
      period_month: 8,
    });
    expect(payload.reservation_ids).toBeUndefined();
    expect(payload.booking_ids).toBeUndefined();
  });

  it('UI-13 : Prepare après changement de payeur suit le dernier plan, pas une sélection périmée', () => {
    const latest = moveClinicBookingToPatient(livePlan, 45697);
    const { summary } = institutionSurfacesFromPlan(latest);
    const payload = clinicMonthlyPreparePayload({
      clinicCompanyId: 1,
      periodYear: 2026,
      periodMonth: 8,
    });
    expect(payload.reservation_ids).toBeUndefined();
    expect(summary.bookingIds).not.toContain(45697);
    expect(summary.bookingIds).toHaveLength(6);
  });

  it('UI-14 : somme HT des lignes = montant résumé = bucket Institution', () => {
    const parity = institutionSurfaceParity(livePlan);
    expect(parity.totalHtMatch).toBe(true);
    expect(parity.prestationCountMatch).toBe(true);
    expect(parity.lineHt).toBe(280);
    expect(parity.summaryHt).toBe(280);
    const preview = presentInvoiceLinesPreview(
      institutionSurfacesFromPlan(livePlan).lines,
      {
        prestationCount: 7,
        totalHt: 280,
      }
    );
    expect(preview.totalHt).toBe(280);
  });

  it('UI-DRAFT-1 : toolbar absente avant Prepare', () => {
    const ui = presentBillPeriodComposerUi({
      hasPreparedDraft: false,
      showLinesPreview: false,
    });
    expect(shouldShowDraftInvoiceToolbar({ hasPreparedDraft: false })).toBe(false);
    expect(ui.showDraftToolbar).toBe(false);
    expect(ui.showPrepareFooter).toBe(true);
  });

  it('UI-DRAFT-2 : toolbar absente quand seul invoiceLinesPreview est ouvert', () => {
    const ui = presentBillPeriodComposerUi({
      hasPreparedDraft: false,
      showLinesPreview: true,
    });
    expect(ui.showDraftToolbar).toBe(false);
    expect(ui.showSimpleLinesPreview).toBe(true);
    expect(shouldShowSimpleInvoiceLinesPreview({
      showLinesPreview: true,
      hasPreparedDraft: false,
    })).toBe(true);
  });

  it('UI-DRAFT-3 : Prepare direct → toolbar présente', () => {
    const ui = presentBillPeriodComposerUi({
      hasPreparedDraft: true,
      showLinesPreview: false,
    });
    expect(ui.showDraftToolbar).toBe(true);
    expect(ui.showSimpleLinesPreview).toBe(false);
    expect(ui.showPrepareFooter).toBe(false);
  });

  it('UI-DRAFT-4 : Preview lignes → Prepare → toolbar présente et liste simple repliée', () => {
    const afterPrepare = presentBillPeriodComposerUi({
      hasPreparedDraft: true,
      showLinesPreview: true,
    });
    expect(afterPrepare.showDraftToolbar).toBe(true);
    expect(afterPrepare.showSimpleLinesPreview).toBe(false);
  });

  it('UI-DRAFT-5 : toolbar et InvoiceLivePreview partagent le même brouillon', () => {
    const prepared = unwrapPreparedDraftInvoice({
      data: { id: 88, invoice_number: 'EM-2026-08-0002', status: 'draft', total_ht: 280 },
    });
    expect(prepared.id).toBe(88);
    expect(prepared.status).toBe('draft');
    expect(prepared.total_ht).toBe(280);
    expect(shouldShowDraftInvoiceToolbar({ hasPreparedDraft: Boolean(prepared) })).toBe(true);
  });

  it('UI-DRAFT-6 / UI-DRAFT-7 : montant et booking_ids du draft = dernier plan', () => {
    const { summary } = institutionSurfacesFromPlan(livePlan);
    const draft = {
      id: 88,
      status: 'draft',
      total_ht: summary.totalHt,
      booking_ids: summary.bookingIds,
    };
    const parity = draftInvoicePlanParity(draft, livePlan);
    expect(parity.totalMatch).toBe(true);
    expect(parity.bookingIdsMatch).toBe(true);
    expect(parity.planTotal).toBe(280);
    expect(parity.draftTotal).toBe(280);
    expect(parity.planBookingIds).toEqual(parity.draftBookingIds);
  });

  it('UI-DRAFT-6b : un 409 réhydrate le brouillon existant sans rester sur le formulaire', () => {
    const existing = draftInvoiceFromPrepareError({
      response: {
        data: {
          existing_invoice_id: 88,
          existing_invoice_number: 'EM-2026-08-0002',
        },
      },
    });
    expect(existing).toEqual({
      id: 88,
      invoice_number: 'EM-2026-08-0002',
      status: 'draft',
    });
    expect(shouldShowDraftInvoiceToolbar({ hasPreparedDraft: Boolean(existing) })).toBe(true);
  });

  it('UI-DRAFT-12 : actions PDF suivent le lifecycle du brouillon', () => {
    const afterPrepareNoPdf = draftPdfActionsAvailability({
      invoiceStatus: 'draft',
      hasStoredPdf: false,
    });
    expect(afterPrepareNoPdf.download).toBe(true);
    expect(afterPrepareNoPdf.print).toBe(true);
    expect(afterPrepareNoPdf.open).toBe(false);

    const withPdf = draftPdfActionsAvailability({
      invoiceStatus: 'draft',
      hasStoredPdf: true,
    });
    expect(withPdf.download).toBe(true);
    expect(withPdf.open).toBe(true);
  });

  it('garde le dernier plan et évite le skeleton pendant un refresh de même scope', () => {
    const scope = institutionPlanScopeKey({
      clinicCompanyId: 1,
      periodYear: 2026,
      periodMonth: 8,
    });
    expect(
      isInstitutionPlanCurrent(livePlan, scope, scope)
    ).toBe(true);
    expect(
      shouldShowInstitutionPlanSkeleton({
        loading: true,
        planIsCurrent: true,
      })
    ).toBe(false);
    expect(
      shouldKeepPreviousInstitutionPlan({ silent: true, hasPreviousPlan: true })
    ).toBe(true);
    expect(
      shouldKeepPreviousInstitutionPlan({ silent: false, hasPreviousPlan: true })
    ).toBe(true);
    expect(isInstitutionPlanFetchCanceled({ name: 'CanceledError' })).toBe(true);
    expect(INSTITUTION_PLAN_POLL_MS).toBe(10000);
    expect(
      flattenInstitutionPreviewBookingIds([{ booking_ids: [1, 2] }])
    ).toEqual([1, 2]);
  });
});
