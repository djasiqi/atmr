import React, { useState, useEffect, useCallback, useMemo, useRef, useId } from 'react';
import { createPortal } from 'react-dom';
import {
  FiX,
  FiEye,
  FiFileText,
  FiLoader,
  FiUser,
  FiHome,
  FiUsers,
  FiChevronsDown,
  FiChevronsUp,
  FiPercent,
  FiPlus,
  FiList,
  FiRefreshCw,
  FiMaximize2,
  FiMinimize2,
  FiDownload,
  FiPrinter,
  FiExternalLink,
  FiCheck,
  FiTrash2,
  FiSearch,
  FiChevronLeft,
  FiChevronRight,
} from 'react-icons/fi';
import { invoiceService, formatCurrencyCHF, generateInvoice } from '../../../../../services/invoiceService';
import { getApiErrorMessage } from '../../../../../utils/apiErrorMessage';
import {
  alreadyInvoicedInvoiceLabel,
  excludedBlockTitle,
  excludedRowWhoLabel,
  exclusionWhyText,
  presentInstitutionAlreadyInvoicedRows,
  presentInstitutionBillablePreviewLines,
  presentInstitutionExcludedRows,
  presentInstitutionInvoiceSummary,
} from '../../../../../utils/institutionInvoicePlanUi';
import {
  bindInstitutionPlanLiveRefresh,
  clinicMonthlyPreparePayload,
  draftInvoiceFromPrepareError,
  institutionPlanScopeKey,
  shouldShowDraftInvoiceToolbar,
  shouldShowSimpleInvoiceLinesPreview,
  unwrapPreparedDraftInvoice,
  isInstitutionPlanCurrent,
  isInstitutionPlanFetchCanceled,
  nextInstitutionPlanRequestId,
  shouldApplyInstitutionPlanResponse,
  shouldKeepPreviousInstitutionPlan,
  shouldShowInstitutionPlanSkeleton,
} from '../../../../../utils/institutionInvoicePlanLiveSync';
import {
  formatPreviewDayMonth,
  presentInvoiceLinesPreview,
} from '../../../../../utils/invoiceLinesPreviewUi';
import {
  presentPartnerInvoiceSummary,
  presentPatientInvoiceSummary,
} from '../../../../../utils/payerInvoiceSummaryUi';
import { canTreatDispute } from '../../../../../utils/bookingDisputeUi';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import DisputeResolutionPanel from './DisputeResolutionPanel';
import InvoiceLivePreview from './InvoiceLivePreview';
import InvoiceLineEditorContext from './InvoiceLineEditorContext';
import draftEditorStyles from './InvoiceDraftEditModal.module.css';
import styles from './BillPeriodModal.module.css';
import InlineMonthYearPicker from '../../../../../components/ui/InlineMonthYearPicker';
import InlineDatePicker from '../../../../../components/ui/InlineDatePicker';
import ChipSelect from '../../../../../components/ui/ChipSelect';
import useCompanySocket from '../../../../../hooks/useCompanySocket';
import { normalizeServiceDateToIsoForApi } from '../../../../../utils/invoiceServiceDate';
import { filterInvoiceLines } from '../../../../../utils/invoiceLineFilter';
import {
  getInvoiceLineMeta,
  invertTrajetLineDescription,
  isRoundTripPreviewHiddenLine,
  canShowRoundTripLegExcludeActions,
  sortInvoiceLinesForEditor,
} from '../../../../../utils/invoiceLineRoundTrip';

const LINE_PAGE_SIZE = 25;
const HEAVY_LINES_THRESHOLD = 12;

const unwrapApi = (res) => {
  if (res && typeof res === 'object' && res.data && typeof res.data === 'object' && 'transports_count' in res.data) {
    return res.data;
  }
  if (res && typeof res === 'object' && 'transports_count' in res) {
    return res;
  }
  return res?.data ?? res;
};

const now = new Date();
const defaultYear = now.getFullYear();
const defaultMonth = now.getMonth() + 1;

/** Libellés mois (liste déroulante « Période facturée »). */
const MONTHS_FR = [
  'janvier',
  'février',
  'mars',
  'avril',
  'mai',
  'juin',
  'juillet',
  'août',
  'septembre',
  'octobre',
  'novembre',
  'décembre',
];

const periodLabelFr = (year, month) => {
  const raw = String(MONTHS_FR[month - 1] || '');
  return `${raw.charAt(0).toUpperCase()}${raw.slice(1)} ${year}`;
};

/** PDF / fichier : toujours après préparation d’un vrai brouillon. */
const PERIOD_PREVIEW_DISABLED_HINT = 'Disponible après « Préparer la facture »';

/** Aligné sur DraftInvoiceEditorPanel — remise globale vs par ligne. */
const PERIOD_REMISE_PCT_MODE = {
  global: 'global',
  perLine: 'perLine',
};

/** Ligne suppl. HT — même modes que DraftInvoiceEditorPanel. */
const EXTRA_LINE_MODE = {
  time: 'time',
  quantity: 'quantity',
};

const TIME_UNITS = [
  { value: 'min', label: 'min' },
  { value: 'h', label: 'h' },
  { value: 'd', label: 'j' },
  { value: 'mois', label: 'mois' },
];

function tauxSuffixForUnit(u) {
  if (u === 'min') return 'CHF/min';
  if (u === 'h') return 'CHF/h';
  if (u === 'd') return 'CHF/j';
  if (u === 'mois') return 'CHF/mois';
  return 'CHF';
}

/** Lignes d’aperçu période : id `pv-{booking_id}`. */
function bookingIdFromPeriodLineId(lineId) {
  const m = String(lineId ?? '').match(/^pv-(.+)$/);
  return m ? m[1] : null;
}

function periodLineTypeLabel(type) {
  const t = String(type || '').toLowerCase();
  if (t === 'ride') return 'Transport';
  if (t === 'material_delivery') return 'Service';
  return 'Ligne';
}

function cloneInvoiceMeta(baseMeta) {
  if (baseMeta == null) return {};
  if (typeof baseMeta === 'string') {
    try {
      const p = JSON.parse(baseMeta);
      return typeof p === 'object' && p !== null ? { ...p } : {};
    } catch {
      return {};
    }
  }
  if (typeof baseMeta === 'object') return { ...baseMeta };
  return {};
}

function applyPeriodRoundTripLegMeta(prevLm, legKept) {
  const nextLm = { ...prevLm };
  delete nextLm.round_trip_merge_partner_reservation_id;
  delete nextLm.is_round_trip_leg;
  delete nextLm.transport_type;
  delete nextLm.round_trip_primary_amount_ht;
  delete nextLm.round_trip_partner_amount_ht;
  delete nextLm.round_trip_partner_description;
  delete nextLm.round_trip_primary_description;
  delete nextLm.service_date_end;
  delete nextLm.round_trip_partner_scheduled_at;
  delete nextLm.round_trip_primary_scheduled_at;
  nextLm.period_preview_single_leg = legKept;
  if (legKept === 'return') {
    const partnerDate = prevLm.round_trip_partner_scheduled_at ?? prevLm.service_date_end;
    if (partnerDate != null && String(partnerDate).trim() !== '') {
      nextLm.service_date = partnerDate;
    }
  } else if (legKept === 'outbound') {
    const primaryDate = prevLm.round_trip_primary_scheduled_at ?? prevLm.service_date;
    if (primaryDate != null && String(primaryDate).trim() !== '') {
      nextLm.service_date = primaryDate;
    }
  }
  return nextLm;
}

/**
 * Applique remises (globale / par ligne), surcharges de lignes, lignes custom et déductions HT à l’aperçu période.
 * Remise globale : `line_meta.original_line_total` + `meta.global_discount` pour le même rendu que le brouillon.
 */
function mergePeriodPreviewInvoice(baseInvoice, linePatch, extraLinesRaw, remiseOpts) {
  if (!baseInvoice) return null;
  const vr =
    Number(baseInvoice.subtotal_amount) > 0.001
      ? Number(baseInvoice.vat_total_amount) / Number(baseInvoice.subtotal_amount)
      : 0;

  const mode = remiseOpts?.mode ?? PERIOD_REMISE_PCT_MODE.global;
  const globalPct = parseFloat(String(remiseOpts?.globalPctStr ?? '').replace(',', '.'));
  const factorGlobal =
    mode === PERIOD_REMISE_PCT_MODE.global && Number.isFinite(globalPct)
      ? 1 - Math.min(100, Math.max(0, globalPct)) / 100
      : 1;
  const perLineMap =
    remiseOpts?.perLineMap && typeof remiseOpts.perLineMap === 'object' ? remiseOpts.perLineMap : {};

  const lines = [];
  for (const ln of baseInvoice.lines) {
    const p = linePatch[ln.id] || {};
    const desc = p.description != null && String(p.description).trim() !== '' ? p.description : ln.description;
    const adjustment_note =
      p.adjustment_note !== undefined ? p.adjustment_note : (ln.adjustment_note ?? '');
    const oht = Number(ln.line_total);
    let ht =
      p.line_total !== undefined && p.line_total !== ''
        ? Number(String(p.line_total).replace(',', '.'))
        : oht;
    if (!Number.isFinite(ht)) ht = oht;
    let vat = Number(ln.vat_amount);
    let ttc = Number(ln.total_with_vat);
    if (Number.isFinite(ht) && oht > 0 && Math.abs(ht - oht) > 0.0001) {
      const r = ht / oht;
      vat = Math.round(vat * r * 100) / 100;
      ttc = Math.round((ht + vat) * 100) / 100;
    }
    const htBeforeRemise = ht;
    let factor = 1;
    if (mode === PERIOD_REMISE_PCT_MODE.global) {
      factor = factorGlobal;
    } else {
      const rp = parseFloat(String(perLineMap[ln.id] ?? '').replace(',', '.'));
      if (Number.isFinite(rp) && rp > 0) {
        factor = 1 - Math.min(100, Math.max(0, rp)) / 100;
      }
    }
    ht = Math.round(ht * factor * 100) / 100;
    vat = Math.round(vat * factor * 100) / 100;
    ttc = Math.round((ht + vat) * 100) / 100;
    const prevLm = ln.line_meta && typeof ln.line_meta === 'object' ? ln.line_meta : {};
    let nextLm =
      p.round_trip_leg_kept === 'outbound' || p.round_trip_leg_kept === 'return'
        ? applyPeriodRoundTripLegMeta(prevLm, p.round_trip_leg_kept)
        : { ...prevLm };
    if (factor < 1 - 1e-9) {
      nextLm.original_line_total = htBeforeRemise;
    } else {
      delete nextLm.original_line_total;
    }
    lines.push({
      ...ln,
      description: desc,
      adjustment_note,
      line_total: ht,
      vat_amount: vat,
      total_with_vat: ttc,
      line_meta: nextLm,
    });
  }

  for (const ex of extraLinesRaw) {
    const ht0 = Number(ex.line_total);
    if (!Number.isFinite(ht0) || ht0 <= 0) continue;
    const fac = mode === PERIOD_REMISE_PCT_MODE.global ? factorGlobal : 1;
    const ht = Math.round(ht0 * fac * 100) / 100;
    const vat = Math.round(ht * vr * 100) / 100;
    const ttc = Math.round((ht + vat) * 100) / 100;
    const lm = {};
    if (fac < 1 - 1e-9) {
      lm.original_line_total = ht0;
    }
    const sd =
      ex.service_date_iso != null ? String(ex.service_date_iso).trim() : '';
    if (sd) {
      lm.service_date_iso = sd;
      lm.service_date = sd;
    }
    const cm = ex.custom_mode === 'quantity' ? 'quantity' : 'time';
    if (cm === 'quantity') {
      lm.custom_prestation = { mode: 'quantity' };
    } else {
      const tu =
        ex.time_unit && ['min', 'h', 'd', 'mois'].includes(String(ex.time_unit))
          ? String(ex.time_unit)
          : 'h';
      lm.custom_prestation = { mode: 'time', time_unit: tu };
    }
    const qtyRaw = Number(ex.qty);
    const qtyLine = Number.isFinite(qtyRaw) && qtyRaw > 0 ? qtyRaw : 1;
    const upRaw = Number(ex.unit_price);
    const unitLine = Number.isFinite(upRaw) && upRaw > 0 ? upRaw : ht0 / qtyLine;
    lines.push({
      id: ex.id,
      type: 'custom',
      line_type: 'custom',
      description: ex.description || '—',
      line_total: ht,
      qty: qtyLine,
      unit_price: unitLine,
      vat_amount: vat,
      total_with_vat: ttc,
      line_meta: lm,
    });
  }

  const deductions = Array.isArray(remiseOpts?.deductionLines) ? remiseOpts.deductionLines : [];
  for (const ded of deductions) {
    const ht = Number(ded.htNegative);
    if (!Number.isFinite(ht) || ht >= 0) continue;
    const vat = Math.round(ht * vr * 100) / 100;
    const ttc = Math.round((ht + vat) * 100) / 100;
    lines.push({
      id: ded.id,
      type: 'custom',
      line_type: 'custom',
      description: ded.description || 'Déduction',
      line_total: ht,
      vat_amount: vat,
      total_with_vat: ttc,
      line_meta: {},
    });
  }

  const subtotal_amount =
    Math.round(lines.reduce((s, l) => s + Number(l.line_total || 0), 0) * 100) / 100;
  const vat_total_amount =
    Math.round(lines.reduce((s, l) => s + Number(l.vat_amount || 0), 0) * 100) / 100;
  const total_amount =
    Math.round(lines.reduce((s, l) => s + Number(l.total_with_vat || 0), 0) * 100) / 100;

  let metaOut = cloneInvoiceMeta(baseInvoice.meta);
  if (mode === PERIOD_REMISE_PCT_MODE.global && factorGlobal < 1 - 1e-9) {
    const pctDisplay = Number.isFinite(globalPct)
      ? Math.min(100, Math.max(0, globalPct))
      : Math.round((1 - factorGlobal) * 10000) / 100;
    const noteTrim = String(remiseOpts?.globalNote ?? '').trim();
    metaOut = {
      ...metaOut,
      global_discount: {
        percent: pctDisplay,
        ...(noteTrim ? { note: noteTrim } : {}),
      },
    };
  } else {
    const { global_discount: _gd, ...rest } = metaOut;
    metaOut = rest;
  }

  return {
    ...baseInvoice,
    lines,
    subtotal_amount,
    vat_total_amount,
    total_amount,
    meta: metaOut,
  };
}

/**
 * Facture factice pour InvoiceLivePreview : lignes = sélection courante, totaux proportionnels au sous-total HT.
 */
function periodPreviewRowKey(row) {
  if (row == null || typeof row !== 'object') return null;
  // Clé UI = booking primaire (l'unité A/R est développée à l'envoi via booking_ids).
  const bid = row.booking_id ?? row.primary_booking_id;
  if (bid != null && Number.isFinite(Number(bid))) return Number(bid);
  return null;
}

function buildSyntheticInvoiceForPeriodAssembly({
  preview,
  selectedBookingIds,
  payerType,
  periodYear,
  periodMonth,
  clientId,
  clinicKey,
  partnershipId,
  clients,
  institutions,
  billablePartners,
}) {
  const lines = Array.isArray(preview?.preview_lines) ? preview.preview_lines : [];
  const subHt = Number(preview.estimated_subtotal_ht ?? 0);
  const vatFull = Number(preview.estimated_vat_total ?? 0);
  const ttcFull = Number(preview.estimated_total_with_vat ?? preview.estimated_total ?? 0);

  const selectedRows = lines.filter((row) => {
    if (row.is_locked) return false;
    const k = periodPreviewRowKey(row);
    return k != null && selectedBookingIds.has(k);
  });
  let htSum = 0;
  for (const row of selectedRows) {
    htSum += Number(row.amount_ht ?? 0);
  }

  const selectable = lines.filter((l) => !l.is_locked);
  const allIdsSelected =
    selectable.length > 0 &&
    selectable.every((l) => {
      const k = periodPreviewRowKey(l);
      return k != null && selectedBookingIds.has(k);
    });

  let scale = 1;
  if (subHt > 0 && htSum >= 0 && !allIdsSelected) {
    scale = htSum / subHt;
  }

  const totalVat = Math.round(vatFull * scale * 100) / 100;
  const totalTtc = Math.round(ttcFull * scale * 100) / 100;
  const subtotal_amount = Math.round(htSum * 100) / 100;

  const invoiceLines = [];
  let remainingVat = totalVat;
  selectedRows.forEach((row, idx) => {
    const lineHt = Number(row.amount_ht ?? 0);
    const st = String(row.source_type || 'ride').toLowerCase();
    const lineType = st === 'service' ? 'material_delivery' : 'ride';
    let lineVat = 0;
    if (selectedRows.length > 0 && Math.abs(totalVat) > 0.0005) {
      if (idx === selectedRows.length - 1) {
        lineVat = Math.round(remainingVat * 100) / 100;
      } else if (htSum > 0) {
        lineVat = Math.round(((totalVat * lineHt) / htSum) * 100) / 100;
        remainingVat -= lineVat;
      }
    }
    const lineTtc = Math.round((lineHt + lineVat) * 100) / 100;
    const arMeta =
      row.is_round_trip_leg === true
        ? {
            is_round_trip_leg: true,
            transport_type: 'A/R',
            billing_unit: 'round_trip',
            primary_booking_id: Number(row.booking_id),
            ...(Array.isArray(row.booking_ids) && row.booking_ids.length
              ? {
                  booking_ids: row.booking_ids.map((id) => Number(id)),
                }
              : row.round_trip_partner_booking_id != null
                ? {
                    booking_ids: [
                      Number(row.booking_id),
                      Number(row.round_trip_partner_booking_id),
                    ],
                  }
                : {}),
            ...(row.round_trip_partner_booking_id != null
              ? {
                  round_trip_merge_partner_reservation_id: Number(row.round_trip_partner_booking_id),
                }
              : {}),
            ...(row.round_trip_primary_amount_ht != null
              ? { round_trip_primary_amount_ht: Number(row.round_trip_primary_amount_ht) }
              : {}),
            ...(row.round_trip_partner_amount_ht != null
              ? { round_trip_partner_amount_ht: Number(row.round_trip_partner_amount_ht) }
              : {}),
            ...(row.round_trip_partner_description != null &&
            String(row.round_trip_partner_description).trim() !== ''
              ? {
                  round_trip_partner_description: String(row.round_trip_partner_description).trim(),
                }
              : {}),
            ...(row.round_trip_partner_scheduled_at != null &&
            String(row.round_trip_partner_scheduled_at).trim() !== ''
              ? {
                  round_trip_partner_scheduled_at: row.round_trip_partner_scheduled_at,
                }
              : {}),
            ...(row.round_trip_primary_scheduled_at != null &&
            String(row.round_trip_primary_scheduled_at).trim() !== ''
              ? {
                  round_trip_primary_scheduled_at: row.round_trip_primary_scheduled_at,
                }
              : {}),
            ...(row.description != null && String(row.description).trim() !== ''
              ? { round_trip_primary_description: String(row.description).trim() }
              : {}),
          }
        : {};
    const patientMeta =
      row.patient_name != null && String(row.patient_name).trim() !== ''
        ? { patient_name: String(row.patient_name).trim() }
        : {};
    const endDateMeta =
      row.scheduled_at_end != null && String(row.scheduled_at_end).trim() !== ''
        ? { service_date_end: row.scheduled_at_end }
        : {};
    const rowKey = periodPreviewRowKey(row);
    invoiceLines.push({
      id: rowKey != null ? `pv-${rowKey}` : `pv-${row.booking_id}`,
      type: lineType,
      line_type: lineType,
      description: row.description || '—',
      line_total: lineHt,
      vat_amount: lineVat,
      total_with_vat: lineTtc,
      reservation_id: row.booking_id,
      line_meta: {
        service_date: row.scheduled_at,
        ...endDateMeta,
        ...arMeta,
        ...patientMeta,
      },
    });
  });

  const billing_strategy =
    preview.mode === 'clinic_monthly'
      ? 's2_clinic_monthly'
      : preview.mode === 'partner_monthly'
        ? 'partner_monthly'
        : undefined;

  let client = { first_name: '', last_name: '—' };
  if (payerType === 'patient' && clientId) {
    const c = clients.find((x) => String(x.id) === String(clientId));
    if (c) {
      client = {
        first_name: c.first_name || '',
        last_name: c.last_name || '',
      };
    }
  } else if (payerType === 'clinic' && clinicKey) {
    const inst = institutions.find((i) => String(i.id) === String(clinicKey));
    if (inst) {
      client = {
        first_name: '',
        last_name: '',
        institution_name: inst.institution_name,
      };
    }
  } else if (payerType === 'partner' && partnershipId && Array.isArray(billablePartners)) {
    const prow = billablePartners.find((p) => String(p.partnership_id) === String(partnershipId));
    if (prow?.partner_company_name) {
      client = {
        first_name: '',
        last_name: '',
        institution_name: String(prow.partner_company_name),
      };
    }
  }

  const vatApplicable = vatFull > 0.005;
  const meta = {
    vat: {
      applicable: vatApplicable,
      label: 'TVA',
    },
  };

  const now = new Date();
  const due = new Date(now);
  due.setDate(due.getDate() + 30);

  return {
    invoice_number: 'Aperçu',
    issued_at: now.toISOString(),
    due_date: due.toISOString(),
    period_year: periodYear,
    period_month: periodMonth,
    billing_strategy,
    client,
    lines: invoiceLines,
    subtotal_amount,
    vat_total_amount: totalVat,
    total_amount: selectedRows.length === 0 ? 0 : totalTtc,
    meta,
  };
}

const BillPeriodModal = ({
  open,
  onClose,
  companyId,
  /** Rafraîchir la liste après génération ou mise à jour (sans ouvrir un second modal). */
  onInvoiceGenerated,
  /** @deprecated Rétrocompatibilité ; non utilisé depuis suppression du lien « Assistant complet ». */
  onOpenLegacy: _unusedOnOpenLegacy,
  onOpenSendEmail,
  onMarkAsSent,
  /** Aligné facturation entreprise : colonnes TVA/TTC dans l’aperçu HTML. */
  companyVatApplicable = true,
}) => {
  /** Conteneur portal dédié sur body — au-dessus de la sidebar, hors stacking context layout. */
  const [portalTarget, setPortalTarget] = useState(null);
  useEffect(() => {
    if (typeof document === 'undefined' || !document.body) return undefined;
    const el = document.createElement('div');
    el.setAttribute('data-portal', 'bill-period-modal');
    document.body.appendChild(el);
    setPortalTarget(el);
    return () => {
      setPortalTarget(null);
      if (el.parentNode) el.parentNode.removeChild(el);
    };
  }, []);

  /** form = sélection payeur/lignes ; draft = préparation dans la même modale. */
  const [composerPhase, setComposerPhase] = useState('form');
  const [draftInvoiceStub, setDraftInvoiceStub] = useState(null);
  const [showLinesPreview, setShowLinesPreview] = useState(false);
  const [showAlreadyInvoicedLines, setShowAlreadyInvoicedLines] = useState(false);
  const [showExcludedLines, setShowExcludedLines] = useState(false);
  const [treatingExcludedRow, setTreatingExcludedRow] = useState(null);
  const [payerType, setPayerType] = useState('patient');
  const [periodYear, setPeriodYear] = useState(defaultYear);
  const [periodMonth, setPeriodMonth] = useState(defaultMonth);
  const [clients, setClients] = useState([]);
  const [institutions, setInstitutions] = useState([]);
  const [clientId, setClientId] = useState('');
  const [clinicKey, setClinicKey] = useState(''); // institution id as string
  const companySocket = useCompanySocket();
  const institutionPlanRequestId = useRef(0);
  const institutionPlanAbortRef = useRef(null);
  const fetchInstitutionPlanRef = useRef(null);
  const [partnershipId, setPartnershipId] = useState('');
  const [billablePartners, setBillablePartners] = useState([]);
  const [loadingLists, setLoadingLists] = useState(false);
  const [preview, setPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [generateLoading, setGenerateLoading] = useState(false);
  const [error, setError] = useState('');
  const [assemblyPreviewExpanded, setAssemblyPreviewExpanded] = useState(false);
  /** Plein écran navigateur sur le bloc aperçu période (aligné brouillon). */
  const [periodPdfFullscreen, setPeriodPdfFullscreen] = useState(false);
  const periodPdfWrapRef = useRef(null);
  /** Panneaux sous l’aperçu (remise / ligne sup. / lignes) — aligné DraftInvoiceEditorPanel. */
  const [periodEditSheet, setPeriodEditSheet] = useState(null);
  const [periodLinePatch, setPeriodLinePatch] = useState(() => ({}));
  const [periodRemisePercentMode, setPeriodRemisePercentMode] = useState(
    () => PERIOD_REMISE_PCT_MODE.global
  );
  const [periodGlobalRemiseInput, setPeriodGlobalRemiseInput] = useState('');
  const [periodRemiseNote, setPeriodRemiseNote] = useState('');
  const [periodPerLineRemisePct, setPeriodPerLineRemisePct] = useState(() => ({}));
  const [periodFreeDeductionDesc, setPeriodFreeDeductionDesc] = useState('');
  const [periodFreeDeductionAmt, setPeriodFreeDeductionAmt] = useState('');
  const [periodDeductionLines, setPeriodDeductionLines] = useState(() => []);
  const [periodExtraLines, setPeriodExtraLines] = useState(() => []);
  const [addLineDesc, setAddLineDesc] = useState('');
  const [addLineMode, setAddLineMode] = useState(() => EXTRA_LINE_MODE.time);
  const [addLineTaux, setAddLineTaux] = useState('');
  const [addLineTimeValue, setAddLineTimeValue] = useState('1');
  const [addLineTimeUnit, setAddLineTimeUnit] = useState('h');
  const [addLineUnitPrice, setAddLineUnitPrice] = useState('');
  const [addLineQty, setAddLineQty] = useState('1');
  const [addLineServiceDate, setAddLineServiceDate] = useState('');
  const addLineServiceDateRef = useRef('');
  /** IDs réservations cochées pour la génération (patient / clinique). */
  const [selectedBookingIds, setSelectedBookingIds] = useState(() => new Set());
  const [periodLineFilter, setPeriodLineFilter] = useState('');
  const [periodLinePage, setPeriodLinePage] = useState(1);
  const [institutionPlan, setInstitutionPlan] = useState(null);
  const [institutionPlanScope, setInstitutionPlanScope] = useState('');
  const [institutionPlanLoading, setInstitutionPlanLoading] = useState(false);
  const [institutionPlanRefreshing, setInstitutionPlanRefreshing] = useState(false);
  const [institutionPlanLiveError, setInstitutionPlanLiveError] = useState('');
  const periodLinesHeadingId = useId();
  const periodRemiseHeadingId = useId();
  const periodAddLineHeadingId = useId();

  /** Unité « mois » : date de prestation = mois+année (stockage premier jour du mois). */
  const addLineServiceDateMonthYearOnly =
    addLineMode === EXTRA_LINE_MODE.time && addLineTimeUnit === 'mois';

  useEffect(() => {
    if (!addLineServiceDateMonthYearOnly) return;
    setAddLineServiceDate((prev) => {
      if (!prev || String(prev).trim().length < 7) return prev;
      const head = String(prev).trim().slice(0, 7);
      return `${head}-01`;
    });
  }, [addLineServiceDateMonthYearOnly]);

  useEffect(() => {
    addLineServiceDateRef.current = addLineServiceDate;
  }, [addLineServiceDate]);

  const addLinePreview = useMemo(() => {
    if (addLineMode === EXTRA_LINE_MODE.quantity) {
      const u = parseFloat(String(addLineUnitPrice).replace(',', '.'));
      const q = parseFloat(String(addLineQty).replace(',', '.'));
      if (!Number.isFinite(u) || u <= 0 || !Number.isFinite(q) || q <= 0) return null;
      return u * q;
    }
    const t = parseFloat(String(addLineTaux).replace(',', '.'));
    const v = parseFloat(String(addLineTimeValue).replace(',', '.'));
    if (!Number.isFinite(t) || t <= 0 || !Number.isFinite(v) || v <= 0) return null;
    return t * v;
  }, [addLineMode, addLineUnitPrice, addLineQty, addLineTaux, addLineTimeValue]);

  const loadLists = useCallback(async () => {
    if (!companyId || !open) return;
    setLoadingLists(true);
    setError('');
    try {
      /** Opportunités patient (sujet + payeur) — remplace le regroupement par client_id seul. */
      const oppPromise =
        payerType === 'patient'
          ? invoiceService.fetchBillingOpportunities(companyId, periodYear, periodMonth)
          : Promise.resolve(null);

      const [oppRaw, inst, bpRaw] = await Promise.all([
        oppPromise,
        invoiceService.fetchInstitutions(companyId),
        invoiceService.fetchBillablePartners(companyId, {
          year: periodYear,
          month: periodMonth,
        }),
      ]);
      const oppData = oppRaw?.data ?? oppRaw;
      const payers = Array.isArray(oppData?.patient_payers) ? oppData.patient_payers : [];
      const ec =
        payerType === 'patient'
          ? payers.map((p) => ({
              id: p.opportunity_key || `client:${p.client_id}`,
              opportunity_key: p.opportunity_key,
              client_id: p.carrier_client_id ?? p.client_id,
              carrier_client_id: p.carrier_client_id ?? p.client_id,
              institution_patient_id:
                p.subject_type === 'institution_patient' ? p.subject_id : null,
              billing_party_id: p.billing_party_id,
              first_name: '',
              last_name: p.display_name || '',
              display_name: p.display_name,
              payer_display_name: p.payer_display_name,
              unbilled_total_amount: p.unbilled_total_amount ?? p.estimated_total,
              can_generate: p.can_generate !== false,
              identity_status: p.identity_status,
              recipient_status: p.recipient_status,
              segments_count: p.segments_count,
              units_count: p.units_count,
              transports_count: p.transports_count ?? p.segments_count,
            }))
          : [];
      setClients(Array.isArray(ec) ? ec : []);
      setInstitutions(inst?.institutions || inst?.data?.institutions || []);
      const bpList = bpRaw?.data ?? bpRaw;
      setBillablePartners(Array.isArray(bpList) ? bpList : []);
    } catch (e) {
      setError("Impossible de charger les listes. Réessayez.");
    } finally {
      setLoadingLists(false);
    }
  }, [companyId, open, periodYear, periodMonth, payerType]);

  useEffect(() => {
    if (open) {
      void loadLists();
    }
  }, [open, loadLists]);

  /** Évite un client_id obsolète (autre entreprise / autre période / autre mode payeur). */
  useEffect(() => {
    if (!open) return;
    setClientId('');
    setClinicKey('');
    setPartnershipId('');
    setPreview(null);
    setShowLinesPreview(false);
    setSelectedBookingIds(new Set());
    setInstitutionPlan(null);
    setComposerPhase('form');
    setDraftInvoiceStub(null);
  }, [companyId, open, periodYear, periodMonth, payerType]);

  useEffect(() => {
    if (!open) {
      setPartnershipId('');
      setClientId('');
      setClinicKey('');
      setPreview(null);
      setShowLinesPreview(false);
      setSelectedBookingIds(new Set());
      setComposerPhase('form');
      setDraftInvoiceStub(null);
      setShowAlreadyInvoicedLines(false);
      setShowExcludedLines(false);
      setTreatingExcludedRow(null);
      setPayerType('patient');
      setAssemblyPreviewExpanded(false);
      const exit =
        document.exitFullscreen ||
        document.webkitExitFullscreen ||
        document.msExitFullscreen;
      if (
        exit &&
        (document.fullscreenElement ||
          document.webkitFullscreenElement ||
          document.msFullscreenElement)
      ) {
        try {
          void exit.call(document);
        } catch {
          /* ignore */
        }
      }
    }
  }, [open]);

  useEffect(() => {
    if (!preview) setAssemblyPreviewExpanded(false);
  }, [preview]);

  useEffect(() => {
    setPeriodEditSheet(null);
    setPeriodLinePatch({});
    setPeriodRemisePercentMode(PERIOD_REMISE_PCT_MODE.global);
    setPeriodGlobalRemiseInput('');
    setPeriodRemiseNote('');
    setPeriodPerLineRemisePct({});
    setPeriodFreeDeductionDesc('');
    setPeriodFreeDeductionAmt('');
    setPeriodDeductionLines([]);
    setPeriodExtraLines([]);
    setAddLineDesc('');
    setAddLineMode(EXTRA_LINE_MODE.time);
    setAddLineTaux('');
    setAddLineTimeValue('1');
    setAddLineTimeUnit('h');
    setAddLineUnitPrice('');
    setAddLineQty('1');
    addLineServiceDateRef.current = '';
    setAddLineServiceDate('');
    setPeriodLineFilter('');
    setPeriodLinePage(1);
  }, [preview]);

  useEffect(() => {
    const syncFs = () => {
      const fs =
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.msFullscreenElement;
      setPeriodPdfFullscreen(!!fs);
    };
    document.addEventListener('fullscreenchange', syncFs);
    document.addEventListener('webkitfullscreenchange', syncFs);
    document.addEventListener('MSFullscreenChange', syncFs);
    return () => {
      document.removeEventListener('fullscreenchange', syncFs);
      document.removeEventListener('webkitfullscreenchange', syncFs);
      document.removeEventListener('MSFullscreenChange', syncFs);
    };
  }, []);

  const handlePeriodBrowserFullscreen = useCallback(() => {
    const el = periodPdfWrapRef.current;
    if (!el) return;
    const fsEl =
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.msFullscreenElement;
    if (!fsEl) {
      const req =
        el.requestFullscreen ||
        el.webkitRequestFullscreen ||
        el.msRequestFullscreen;
      if (req) void req.call(el);
    } else {
      const exit =
        document.exitFullscreen ||
        document.webkitExitFullscreen ||
        document.msExitFullscreen;
      if (exit) void exit.call(document);
    }
  }, []);

  useEffect(() => {
    if (!preview) {
      setSelectedBookingIds(new Set());
      return;
    }
    const lines = preview.preview_lines;
    if (!Array.isArray(lines) || lines.length === 0) {
      setSelectedBookingIds(new Set());
      return;
    }
    const next = new Set(
      lines
        .filter((l) => !l.is_locked)
        .map((l) => periodPreviewRowKey(l))
        .filter((k) => k != null)
    );
    setSelectedBookingIds(next);
  }, [preview, payerType]);

  const hasAssemblyLines =
    (payerType === 'patient' ||
      payerType === 'clinic' ||
      payerType === 'partner') &&
    Array.isArray(preview?.preview_lines) &&
    preview.preview_lines.length > 0;

  const periodPreviewBarSubtitle = useMemo(() => {
    const mi = Math.max(1, Math.min(12, Number(periodMonth) || 1)) - 1;
    const mLabel = MONTHS_FR[mi] || '';
    const periodPart = `${mLabel.charAt(0).toUpperCase()}${mLabel.slice(1)} ${periodYear}`;
    if (payerType === 'patient' && clientId) {
      const c = clients.find((x) => String(x.id) === String(clientId));
      if (c) {
        const n = (c.display_name || `${c.first_name || ''} ${c.last_name || ''}`).trim();
        if (n) return `${periodPart} · ${n}`;
      }
    }
    if (payerType === 'clinic' && clinicKey) {
      const inst = institutions.find((i) => String(i.id) === String(clinicKey));
      if (inst?.institution_name) return `${periodPart} · ${inst.institution_name}`;
    }
    if (payerType === 'partner' && partnershipId) {
      const pr = billablePartners.find((p) => String(p.partnership_id) === String(partnershipId));
      if (pr?.partner_company_name) return `${periodPart} · ${pr.partner_company_name}`;
    }
    return periodPart;
  }, [
    periodMonth,
    periodYear,
    payerType,
    clientId,
    clinicKey,
    partnershipId,
    clients,
    institutions,
    billablePartners,
  ]);

  const syntheticPeriodInvoice = useMemo(() => {
    if (!hasAssemblyLines || !preview) return null;
    return buildSyntheticInvoiceForPeriodAssembly({
      preview,
      selectedBookingIds,
      payerType,
      periodYear,
      periodMonth,
      clientId,
      clinicKey,
      partnershipId,
      clients,
      institutions,
      billablePartners,
    });
  }, [
    hasAssemblyLines,
    preview,
    selectedBookingIds,
    payerType,
    periodYear,
    periodMonth,
    clientId,
    clinicKey,
    partnershipId,
    clients,
    institutions,
    billablePartners,
  ]);

  const mergedPeriodInvoice = useMemo(
    () =>
      mergePeriodPreviewInvoice(syntheticPeriodInvoice, periodLinePatch, periodExtraLines, {
        mode: periodRemisePercentMode,
        globalPctStr: periodGlobalRemiseInput,
        globalNote: periodRemiseNote,
        perLineMap: periodPerLineRemisePct,
        deductionLines: periodDeductionLines,
      }),
    [
      syntheticPeriodInvoice,
      periodLinePatch,
      periodExtraLines,
      periodRemisePercentMode,
      periodGlobalRemiseInput,
      periodRemiseNote,
      periodPerLineRemisePct,
      periodDeductionLines,
    ]
  );

  const billPatientChipOptions = useMemo(() => {
    const rows = clients.map((c) => {
      const name = c.display_name || `${c.first_name || ''} ${c.last_name || ''}`.trim();
      const amt = c.unbilled_total_amount;
      const payer =
        c.payer_display_name && c.payer_display_name !== name
          ? ` — facturé à ${c.payer_display_name}`
          : '';
      const blocked = c.can_generate === false ? ' [à compléter]' : '';
      const segments = Number(c.segments_count) || 0;
      const trips = segments > 0 ? ` — ${segments} transport${segments > 1 ? 's' : ''}` : '';
      return {
        value: String(c.id),
        label: `${name}${payer}${trips}${amt != null ? ` — ${amt} CHF` : ''}${blocked}`,
        disabled: c.can_generate === false,
      };
    });
    return rows;
  }, [clients]);

  const billClinicChipOptions = useMemo(
    () =>
      institutions.map((i) => ({
        value: String(i.id),
        label: `${i.institution_name}${i.clinic_company_id ? ` (S2 #${i.clinic_company_id})` : ''}`,
      })),
    [institutions]
  );

  const billPartnerChipOptions = useMemo(
    () =>
      billablePartners.map((p) => ({
        value: String(p.partnership_id),
        label: `${p.partner_company_name}${
          typeof p.validated_unbilled_transfers_count === 'number'
            ? ` · ${p.validated_unbilled_transfers_count} validé(s)`
            : ''
        }`,
      })),
    [billablePartners]
  );

  const hasActivePeriodRemise = useMemo(() => {
    const g = parseFloat(String(periodGlobalRemiseInput ?? '').replace(',', '.'));
    if (Number.isFinite(g) && g > 0.004) return true;
    for (const v of Object.values(periodPerLineRemisePct)) {
      const p = parseFloat(String(v ?? '').replace(',', '.'));
      if (Number.isFinite(p) && p > 0.004) return true;
    }
    if (periodDeductionLines.length > 0) return true;
    return false;
  }, [periodGlobalRemiseInput, periodPerLineRemisePct, periodDeductionLines]);

  const openPeriodSheet = useCallback((key) => {
    setPeriodEditSheet((prev) => (prev === key ? null : key));
  }, []);

  const handlePeriodYmPickerChange = useCallback((ym) => {
    if (!ym) {
      setPeriodYear(defaultYear);
      setPeriodMonth(defaultMonth);
    } else {
      const [ys, ms] = ym.split('-');
      const y = parseInt(ys, 10);
      const m = parseInt(ms, 10);
      if (!Number.isFinite(y) || !Number.isFinite(m)) return;
      setPeriodYear(Math.min(2100, Math.max(2000, y)));
      setPeriodMonth(Math.min(12, Math.max(1, m)));
    }
    setClientId('');
    setClinicKey('');
    setPartnershipId('');
    setPreview(null);
    setSelectedBookingIds(new Set());
  }, []);

  const resetPeriodLinePatch = useCallback((lineId) => {
    setPeriodLinePatch((prev) => {
      if (!prev[lineId]) return prev;
      const next = { ...prev };
      delete next[lineId];
      return next;
    });
  }, []);

  const periodEditorLines = useMemo(
    () => sortInvoiceLinesForEditor(Array.isArray(mergedPeriodInvoice?.lines) ? mergedPeriodInvoice.lines : []),
    [mergedPeriodInvoice?.lines]
  );

  const periodFilteredLines = useMemo(
    () => filterInvoiceLines(periodEditorLines, periodLineFilter),
    [periodEditorLines, periodLineFilter]
  );

  const periodTotalLinePages = Math.max(1, Math.ceil(periodFilteredLines.length / LINE_PAGE_SIZE));
  const periodEffectiveLinePage = Math.min(Math.max(1, periodLinePage), periodTotalLinePages);

  const periodPaginatedLines = useMemo(() => {
    const start = (periodEffectiveLinePage - 1) * LINE_PAGE_SIZE;
    return periodFilteredLines.slice(start, start + LINE_PAGE_SIZE);
  }, [periodFilteredLines, periodEffectiveLinePage]);

  useEffect(() => {
    setPeriodLinePage((p) => Math.min(p, periodTotalLinePages));
  }, [periodTotalLinePages]);

  useEffect(() => {
    setPeriodLinePage(1);
  }, [periodLineFilter]);

  const periodLineRangeStart =
    periodFilteredLines.length === 0 ? 0 : (periodEffectiveLinePage - 1) * LINE_PAGE_SIZE + 1;
  const periodLineRangeEnd =
    periodFilteredLines.length === 0
      ? 0
      : Math.min(periodEffectiveLinePage * LINE_PAGE_SIZE, periodFilteredLines.length);

  const showPeriodLinesToolbar = periodEditorLines.length >= HEAVY_LINES_THRESHOLD;

  /** Retire la réservation de la sélection : la ligne disparaît de l’aperçu et des totaux. */
  const excludePeriodLineFromPreview = useCallback((lineId) => {
    const rawBid = bookingIdFromPeriodLineId(lineId);
    if (rawBid == null) return;
    setSelectedBookingIds((prev) => {
      const next = new Set(prev);
      for (const id of prev) {
        if (String(id) === String(rawBid)) {
          next.delete(id);
          break;
        }
      }
      return next;
    });
    setPeriodLinePatch((prev) => {
      if (!prev[lineId]) return prev;
      const n = { ...prev };
      delete n[lineId];
      return n;
    });
  }, []);

  /** A/R fusionné en aperçu période : une seule jambe (description, date, HT, sans tag [A/R]). */
  const excludePeriodRoundTripLeg = useCallback(
    (lineId, leg) => {
      const baseLn = syntheticPeriodInvoice?.lines?.find((l) => l.id === lineId);
      if (!baseLn) return;
      const meta = getInvoiceLineMeta(baseLn);
      const primaryHt = meta?.round_trip_primary_amount_ht;
      const partnerHt = meta?.round_trip_partner_amount_ht;
      const primaryDesc =
        meta?.round_trip_primary_description != null &&
        String(meta.round_trip_primary_description).trim() !== ''
          ? String(meta.round_trip_primary_description).trim()
          : (baseLn.description ?? '');
      const partnerDescRaw =
        meta?.round_trip_partner_description != null &&
        String(meta.round_trip_partner_description).trim() !== ''
          ? String(meta.round_trip_partner_description).trim()
          : null;

      if (leg === 'return' && primaryHt != null && Number.isFinite(Number(primaryHt))) {
        setPeriodLinePatch((prev) => ({
          ...prev,
          [lineId]: {
            ...prev[lineId],
            line_total: String(Number(primaryHt)),
            description: primaryDesc,
            round_trip_leg_kept: 'outbound',
          },
        }));
        return;
      }
      if (leg === 'outbound' && partnerHt != null && Number.isFinite(Number(partnerHt))) {
        const returnDesc =
          partnerDescRaw && partnerDescRaw !== primaryDesc
            ? partnerDescRaw
            : invertTrajetLineDescription(primaryDesc);
        setPeriodLinePatch((prev) => ({
          ...prev,
          [lineId]: {
            ...prev[lineId],
            line_total: String(Number(partnerHt)),
            description: returnDesc,
            round_trip_leg_kept: 'return',
          },
        }));
      }
    },
    [syntheticPeriodInvoice]
  );

  const handlePeriodRemoveRemise = useCallback(() => {
    setPeriodGlobalRemiseInput('');
    setPeriodRemiseNote('');
    setPeriodPerLineRemisePct({});
    setPeriodDeductionLines([]);
    setPeriodRemisePercentMode(PERIOD_REMISE_PCT_MODE.global);
    setError('');
  }, []);

  const handlePeriodApplyGlobalRemise = useCallback(() => {
    const raw = String(periodGlobalRemiseInput ?? '').trim();
    if (raw === '') {
      setError('');
      return;
    }
    const p = parseFloat(raw.replace(',', '.'));
    if (!Number.isFinite(p) || p < 0 || p > 100) {
      setError('Indiquez un pourcentage entre 0 et 100, ou laissez vide.');
      return;
    }
    setError('');
  }, [periodGlobalRemiseInput]);

  const handlePeriodApplyPerLineRemise = useCallback(() => {
    for (const lid of Object.keys(periodPerLineRemisePct)) {
      const v = periodPerLineRemisePct[lid];
      if (v === undefined || String(v).trim() === '') continue;
      const p = parseFloat(String(v).replace(',', '.'));
      if (!Number.isFinite(p) || p <= 0 || p > 100) {
        setError('Chaque pourcentage doit être entre 0 et 100 (champs vides autorisés).');
        return;
      }
    }
    setError('');
  }, [periodPerLineRemisePct]);

  const handlePeriodAddDeductionLine = useCallback(() => {
    const amt = parseFloat(String(periodFreeDeductionAmt).replace(',', '.'));
    if (!Number.isFinite(amt) || amt <= 0) {
      setError('Indiquez un montant HT positif pour la déduction.');
      return;
    }
    setPeriodDeductionLines((prev) => [
      ...prev,
      {
        id: `ded-${Date.now()}`,
        description: periodFreeDeductionDesc.trim() || 'Déduction',
        htNegative: -Math.abs(amt),
      },
    ]);
    setPeriodFreeDeductionDesc('');
    setPeriodFreeDeductionAmt('');
    setError('');
  }, [periodFreeDeductionAmt, periodFreeDeductionDesc]);

  const handlePeriodAddExtraLine = useCallback(async () => {
    if (!addLineDesc.trim()) {
      setError('Indiquez un libellé.');
      return;
    }
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
    const rawSvc = (addLineServiceDateRef.current || addLineServiceDate || '').trim();
    let serviceDateIso;
    if (rawSvc) {
      serviceDateIso = normalizeServiceDateToIsoForApi(rawSvc);
      if (!serviceDateIso) {
        setError('Date de prestation invalide (utilisez JJ.MM.AAAA ou AAAA-MM-JJ).');
        return;
      }
    }
    let lineTotal;
    let qtyVal;
    let unitPriceVal;
    let customMode;
    let timeUnit;
    if (addLineMode === EXTRA_LINE_MODE.quantity) {
      const u = parseFloat(String(addLineUnitPrice).replace(',', '.'));
      const q = parseFloat(String(addLineQty).replace(',', '.'));
      if (!Number.isFinite(u) || u <= 0) {
        setError('Prix unitaire HT invalide.');
        return;
      }
      if (!Number.isFinite(q) || q <= 0) {
        setError('Quantité invalide.');
        return;
      }
      lineTotal = u * q;
      qtyVal = q;
      unitPriceVal = u;
      customMode = 'quantity';
    } else {
      const t = parseFloat(String(addLineTaux).replace(',', '.'));
      const v = parseFloat(String(addLineTimeValue).replace(',', '.'));
      if (!Number.isFinite(t) || t <= 0) {
        setError('Prix (CHF) par unité de temps invalide.');
        return;
      }
      if (!Number.isFinite(v) || v <= 0) {
        setError('Valeur temps invalide.');
        return;
      }
      lineTotal = t * v;
      qtyVal = v;
      unitPriceVal = t;
      customMode = 'time';
      timeUnit = addLineTimeUnit;
    }
    setPeriodExtraLines((prev) => [
      ...prev,
      {
        id: `extra-${Date.now()}`,
        description: addLineDesc.trim() || 'Prestation',
        line_total: lineTotal,
        qty: qtyVal,
        unit_price: unitPriceVal,
        custom_mode: customMode,
        time_unit: customMode === 'time' ? timeUnit : undefined,
        service_date_iso: serviceDateIso,
      },
    ]);
    setAddLineDesc('');
    setAddLineTaux('');
    setAddLineTimeValue('1');
    setAddLineTimeUnit('h');
    setAddLineUnitPrice('');
    setAddLineQty('1');
    addLineServiceDateRef.current = '';
    setAddLineServiceDate('');
    setAddLineMode(EXTRA_LINE_MODE.time);
    setError('');
  }, [
    addLineDesc,
    addLineMode,
    addLineTaux,
    addLineTimeValue,
    addLineUnitPrice,
    addLineQty,
    addLineServiceDate,
    addLineTimeUnit,
  ]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape' || !open || generateLoading || previewLoading) return;
      if (treatingExcludedRow) return;
      if (periodEditSheet) {
        setPeriodEditSheet(null);
        return;
      }
      onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose, generateLoading, previewLoading, periodEditSheet, treatingExcludedRow]);

  const selectedClinic = institutions.find((i) => String(i.id) === clinicKey);
  const clinicCompanyId = selectedClinic?.clinic_company_id ?? null;

  const fetchInstitutionPlan = useCallback(
    async ({ silent = false } = {}) => {
      if (!open || !companyId || !clinicCompanyId || payerType !== 'clinic') {
        if (!silent && payerType !== 'clinic') {
          setInstitutionPlan(null);
          setInstitutionPlanScope('');
          setInstitutionPlanLoading(false);
          setInstitutionPlanRefreshing(false);
          setInstitutionPlanLiveError('');
        }
        return;
      }
      const fetchScope = institutionPlanScopeKey({
        clinicCompanyId,
        periodYear,
        periodMonth,
      });
      const reqId = nextInstitutionPlanRequestId(institutionPlanRequestId.current);
      institutionPlanRequestId.current = reqId;
      institutionPlanAbortRef.current?.abort();
      const controller = new AbortController();
      institutionPlanAbortRef.current = controller;
      if (silent) {
        setInstitutionPlanRefreshing(true);
      } else {
        setInstitutionPlanLoading(true);
      }
      try {
        const res = await invoiceService.fetchInstitutionInvoicePlan(companyId, {
          year: periodYear,
          month: periodMonth,
          clinicCompanyId,
          clinicClientId: clinicKey || undefined,
          signal: controller.signal,
        });
        if (!shouldApplyInstitutionPlanResponse(reqId, institutionPlanRequestId.current)) {
          return;
        }
        setInstitutionPlan(unwrapApi(res));
        setInstitutionPlanScope(fetchScope);
        setInstitutionPlanLiveError('');
        if (!silent) {
          setShowAlreadyInvoicedLines(false);
          setShowExcludedLines(false);
        }
      } catch (err) {
        if (isInstitutionPlanFetchCanceled(err)) return;
        if (!shouldApplyInstitutionPlanResponse(reqId, institutionPlanRequestId.current)) {
          return;
        }
        setInstitutionPlanLiveError('Mise à jour du plan impossible.');
        setInstitutionPlan((prev) =>
          shouldKeepPreviousInstitutionPlan({
            silent,
            hasPreviousPlan: Boolean(prev),
          })
            ? prev
            : null
        );
      } finally {
        if (shouldApplyInstitutionPlanResponse(reqId, institutionPlanRequestId.current)) {
          setInstitutionPlanRefreshing(false);
          if (!silent) {
            setInstitutionPlanLoading(false);
          }
        }
      }
    },
    [
      open,
      companyId,
      clinicCompanyId,
      clinicKey,
      periodYear,
      periodMonth,
      payerType,
    ]
  );

  fetchInstitutionPlanRef.current = fetchInstitutionPlan;

  useEffect(() => {
    if (!open || !companyId || !clinicCompanyId || payerType !== 'clinic') {
      institutionPlanRequestId.current = nextInstitutionPlanRequestId(
        institutionPlanRequestId.current
      );
      institutionPlanAbortRef.current?.abort();
      if (payerType !== 'clinic') {
        setInstitutionPlan(null);
        setInstitutionPlanScope('');
        setInstitutionPlanLoading(false);
        setInstitutionPlanRefreshing(false);
        setInstitutionPlanLiveError('');
      }
      return undefined;
    }
    void fetchInstitutionPlan({ silent: false });
    return undefined;
  }, [open, companyId, clinicCompanyId, clinicKey, periodYear, periodMonth, payerType, fetchInstitutionPlan]);

  useEffect(() => {
    if (!open || payerType !== 'clinic' || !clinicCompanyId) return undefined;
    return bindInstitutionPlanLiveRefresh({
      socket: companySocket,
      refresh: () => {
        void fetchInstitutionPlanRef.current?.({ silent: true });
      },
    });
  }, [open, payerType, clinicCompanyId, companySocket]);

  const currentInstitutionPlanScope = institutionPlanScopeKey({
    clinicCompanyId,
    periodYear,
    periodMonth,
  });
  const institutionPlanIsCurrent = isInstitutionPlanCurrent(
    institutionPlan,
    institutionPlanScope,
    currentInstitutionPlanScope
  );
  const showInstitutionPlanSkeleton = shouldShowInstitutionPlanSkeleton({
    loading: institutionPlanLoading,
    planIsCurrent: institutionPlanIsCurrent,
  });

  const canPreview = useCallback(() => {
    if (payerType === 'patient') return Boolean(clientId);
    if (payerType === 'clinic') return Boolean(clinicKey && clinicCompanyId);
    if (payerType === 'partner') return Boolean(partnershipId);
    return false;
  }, [payerType, clientId, clinicKey, clinicCompanyId, partnershipId]);

  const acceptPreparedDraft = (inv) => {
    const stub = unwrapPreparedDraftInvoice(inv);
    if (!stub?.id) return false;
    setDraftInvoiceStub(stub);
    setComposerPhase('draft');
    setShowLinesPreview(false);
    onInvoiceGenerated?.(stub);
    return true;
  };

  const acceptPreparedDraftOrError = (err) => {
    const existing = draftInvoiceFromPrepareError(err);
    if (existing) {
      acceptPreparedDraft(existing);
      return true;
    }
    return false;
  };

  const prepareClinicFromPlan = async () => {
    if (!clinicCompanyId) {
      setError('Sélectionnez une institution pour préparer la facture.');
      return;
    }
    setPayerType('clinic');
    setError('');
    setGenerateLoading(true);
    try {
      const result = await generateInvoice(
        companyId,
        clinicMonthlyPreparePayload({
          clinicCompanyId,
          periodYear,
          periodMonth,
        })
      );
      if (!acceptPreparedDraft(result)) {
        setError('Réponse inattendue du serveur.');
      }
    } catch (err) {
      if (!acceptPreparedDraftOrError(err)) {
        setError(getApiErrorMessage(err, 'Échec de génération'));
      }
    } finally {
      setGenerateLoading(false);
    }
  };

  const preparePatientFromOpportunity = async () => {
    const opp = clients.find((c) => String(c.id) === String(clientId));
    if (!opp) {
      setError('Sélectionnez un patient pour préparer la facture.');
      return;
    }
    if (opp.can_generate === false) {
      setError('Ce patient n’est pas encore facturable — identité ou destinataire à compléter.');
      return;
    }
    const carrierId = Number(opp.carrier_client_id ?? opp.client_id);
    setError('');
    setGenerateLoading(true);
    try {
      const result = await generateInvoice(companyId, {
        ...(Number.isFinite(carrierId) && carrierId > 0 ? { client_id: carrierId } : {}),
        period_year: periodYear,
        period_month: periodMonth,
        billing_opportunity_key: String(opp.opportunity_key || clientId),
      });
      if (!acceptPreparedDraft(result)) {
        setError('Réponse inattendue du serveur.');
      }
    } catch (err) {
      if (!acceptPreparedDraftOrError(err)) {
        setError(getApiErrorMessage(err, 'Échec de génération'));
      }
    } finally {
      setGenerateLoading(false);
    }
  };

  const preparePartnerFromRow = async () => {
    if (!partnershipId) {
      setError('Sélectionnez un partenaire pour préparer la facture.');
      return;
    }
    setError('');
    setGenerateLoading(true);
    try {
      const result = await invoiceService.generatePartnerInvoice(companyId, {
        partnership_id: parseInt(partnershipId, 10),
        period_year: periodYear,
        period_month: periodMonth,
      });
      if (!acceptPreparedDraft(result)) {
        setError('Réponse inattendue du serveur.');
      }
    } catch (err) {
      if (!acceptPreparedDraftOrError(err)) {
        setError(getApiErrorMessage(err, 'Échec de génération'));
      }
    } finally {
      setGenerateLoading(false);
    }
  };

  const selectedPatient = useMemo(
    () => clients.find((c) => String(c.id) === String(clientId)) || null,
    [clients, clientId]
  );
  const selectedPartner = useMemo(
    () =>
      billablePartners.find((p) => String(p.partnership_id) === String(partnershipId)) ||
      null,
    [billablePartners, partnershipId]
  );
  const patientSummary = useMemo(
    () => presentPatientInvoiceSummary(selectedPatient),
    [selectedPatient]
  );
  const partnerSummary = useMemo(
    () => presentPartnerInvoiceSummary(selectedPartner),
    [selectedPartner]
  );
  const institutionSummary = useMemo(
    () => presentInstitutionInvoiceSummary(institutionPlan),
    [institutionPlan]
  );
  const institutionAlreadyInvoicedRows = useMemo(
    () => presentInstitutionAlreadyInvoicedRows(institutionPlan),
    [institutionPlan]
  );
  const institutionExcludedRows = useMemo(
    () => presentInstitutionExcludedRows(institutionPlan),
    [institutionPlan]
  );
  const institutionPreviewLines = useMemo(
    () => presentInstitutionBillablePreviewLines(institutionPlan),
    [institutionPlan]
  );
  const linesPreview = useMemo(() => {
    const prestationCount =
      payerType === 'clinic'
        ? institutionPlanIsCurrent
          ? institutionSummary.transportsCount
          : 0
        : payerType === 'patient'
          ? patientSummary.transportsCount
          : partnerSummary.transportsCount;
    const fallbackTotal =
      payerType === 'clinic'
        ? institutionPlanIsCurrent
          ? institutionSummary.totalHt
          : 0
        : payerType === 'patient'
          ? patientSummary.totalHt
          : partnerSummary.totalHt;
    const sourceLines =
      payerType === 'clinic'
        ? institutionPlanIsCurrent
          ? institutionPreviewLines
          : []
        : preview?.preview_lines;
    const sourceTotal =
      payerType === 'clinic'
        ? fallbackTotal
        : preview?.estimated_total != null
          ? preview.estimated_total
          : fallbackTotal;
    return presentInvoiceLinesPreview(sourceLines, {
      prestationCount,
      totalHt: sourceTotal,
    });
  }, [
    preview,
    payerType,
    institutionPreviewLines,
    institutionPlanIsCurrent,
    institutionSummary.transportsCount,
    institutionSummary.totalHt,
    patientSummary.transportsCount,
    patientSummary.totalHt,
    partnerSummary.transportsCount,
    partnerSummary.totalHt,
  ]);

  const footerHint = useMemo(() => {
    if (loadingLists) return null;
    if (payerType === 'patient' && !clientId) {
      return 'Sélectionnez un patient pour préparer la facture.';
    }
    if (payerType === 'patient' && clientId && !patientSummary.hasBillable) {
      return patientSummary.blocked
        ? 'Ce patient n’est pas encore facturable — identité ou destinataire à compléter.'
        : 'Aucune prestation à charge de ce patient pour cette période.';
    }
    if (payerType === 'clinic') {
      if (!clinicKey) return 'Sélectionnez une institution pour préparer la facture.';
      if (!clinicCompanyId) {
        return 'Cette institution n’a pas d’entreprise S2 associée — impossible de facturer.';
      }
      if (institutionPlanIsCurrent && !institutionSummary.hasBillable) {
        return 'Aucune prestation à charge de cette institution pour cette période.';
      }
    }
    if (payerType === 'partner' && !partnershipId) {
      return 'Sélectionnez un partenaire pour préparer la facture.';
    }
    if (payerType === 'partner' && partnershipId && !partnerSummary.hasBillable) {
      return 'Aucune prestation à charge de ce partenaire pour cette période.';
    }
    if (preview && preview.transports_count === 0) {
      return payerType === 'partner'
        ? 'Aucun transfert à facturer sur cette période pour ce partenaire.'
        : 'Aucun transport à facturer sur cette période pour ce payeur.';
    }
    return null;
  }, [
    loadingLists,
    payerType,
    clientId,
    clinicKey,
    clinicCompanyId,
    preview,
    partnershipId,
    institutionPlanIsCurrent,
    institutionSummary.hasBillable,
    patientSummary.hasBillable,
    patientSummary.blocked,
    partnerSummary.hasBillable,
  ]);

  const runPreview = async () => {
    if (!canPreview()) {
      setError('Sélectionnez un payeur.');
      return;
    }
    setError('');
    setPreview(null);

    if (payerType === 'partner') {
      const row = billablePartners.find((p) => String(p.partnership_id) === partnershipId);
      if (!row) {
        setError('Partenariat introuvable pour cette période.');
        return;
      }
      const validated = Number(row.validated_unbilled_transfers_count ?? 0);
      const total = Number(row.total_amount ?? 0);
      const unbilled = Number(row.unbilled_transfers_count ?? 0);
      const warnings = [];
      if (validated === 0 && unbilled > 0) {
        warnings.push(
          'Certains transferts ne sont pas encore validés — le montant estimé ne les inclut pas.'
        );
      }
      if (validated === 0 && unbilled === 0) {
        warnings.push('Aucun transfert facturable sur cette période pour ce partenaire.');
      }
      const previewLines = Array.isArray(row.preview_lines) ? row.preview_lines : [];
      const estHt = Number(row.estimated_subtotal_ht ?? total ?? 0);
      const estVat = Number(row.estimated_vat_total ?? 0);
      const estTtc = Number(row.estimated_total_with_vat ?? total ?? estHt + estVat);
      setPreview({
        mode: 'partner_monthly',
        transports_count: validated,
        estimated_total: total,
        preview_lines: previewLines,
        estimated_subtotal_ht: estHt,
        estimated_vat_total: estVat,
        estimated_total_with_vat: estTtc,
        warnings,
      });
      return;
    }

    setPreviewLoading(true);
    try {
      let parsedClientId;
      let selectedOpp = null;
      if (payerType === 'patient') {
        selectedOpp = clients.find((c) => String(c.id) === String(clientId));
        if (!selectedOpp) {
          setError(
            'Ce patient ne correspond pas à la liste chargée pour cette période. Attendez la fin du chargement ou sélectionnez à nouveau un patient.'
          );
          setPreviewLoading(false);
          return;
        }
        parsedClientId = Number(
          selectedOpp.carrier_client_id ?? selectedOpp.client_id ?? selectedOpp.id
        );
        if (!Number.isFinite(parsedClientId) || parsedClientId <= 0) {
          setError('Sélectionnez un patient dans la liste.');
          setPreviewLoading(false);
          return;
        }
      }
      if (payerType === 'clinic') {
        const cc = clinicCompanyId;
        if (cc == null || !Number.isFinite(Number(cc))) {
          setError('Sélectionnez une institution avec identifiant de facturation (S2).');
          setPreviewLoading(false);
          return;
        }
        const clinicAllowed = institutions.some((i) => Number(i.clinic_company_id) === Number(cc));
        if (!clinicAllowed) {
          setError(
            'Cette institution ne correspond pas à la liste chargée. Sélectionnez-la à nouveau après le chargement.'
          );
          setPreviewLoading(false);
          return;
        }
      }
      const res = await invoiceService.fetchPeriodPreview(companyId, {
        year: periodYear,
        month: periodMonth,
        clientId: payerType === 'patient' ? parsedClientId : undefined,
        clinicCompanyId: payerType === 'clinic' ? clinicCompanyId : undefined,
        institutionPatientId:
          payerType === 'patient' && selectedOpp?.institution_patient_id
            ? selectedOpp.institution_patient_id
            : undefined,
        billingPartyId:
          payerType === 'patient' && selectedOpp?.billing_party_id
            ? selectedOpp.billing_party_id
            : undefined,
      });
      setPreview(unwrapApi(res));
    } catch (err) {
      setError(getApiErrorMessage(err, 'Prévisualisation impossible'));
    } finally {
      setPreviewLoading(false);
    }
  };

  const toggleLinesPreview = async () => {
    if (showLinesPreview) {
      setShowLinesPreview(false);
      return;
    }
    if (payerType === 'clinic') {
      setShowLinesPreview(true);
      return;
    }
    if (!preview) {
      await runPreview();
    }
    setShowLinesPreview(true);
  };

  const runGenerate = async () => {
    if (!canPreview()) {
      setError('Sélectionnez un payeur.');
      return;
    }
    if (payerType === 'clinic') {
      await prepareClinicFromPlan();
      return;
    }
    if (payerType === 'patient') {
      await preparePatientFromOpportunity();
      return;
    }
    if (payerType === 'partner') {
      await preparePartnerFromRow();
      return;
    }
    setError('Sélectionnez un payeur.');
  };

  /** Référence stable : un callback recréé à chaque rendu relancerait le chargement du panneau brouillon. */
  const handleDraftPanelUpdated = useCallback(() => {
    onInvoiceGenerated?.(draftInvoiceStub);
  }, [onInvoiceGenerated, draftInvoiceStub]);

  if (!open || !portalTarget) return null;

  return createPortal(
    <>
    <div
      className={`${styles.overlay}${treatingExcludedRow ? ` ${styles.overlayDisputeLocked}` : ''}`}
      onClick={onClose}
      role="presentation"
    >
      <div
        className={styles.panel}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="bill-period-modal-title"
      >
          <>
            <div className={styles.head}>
              <div className={styles.headText}>
                <h2 className={styles.title} id="bill-period-modal-title">Nouvelle facture</h2>
              </div>
              <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
                <FiX size={18} />
              </button>
            </div>

            <div className={styles.formColumn}>
              <div
                className={`${styles.formScroll}${
                  draftInvoiceStub ? ` ${styles.formScrollWithDraft}` : ''
                }`}
                data-testid="bill-period-form-scroll"
              >
          <div className={styles.section}>
            <div className={styles.fieldGroup} data-testid="bill-period-payer-type">
              <span className={styles.fieldLabel}>Type de payeur</span>
              <div className={styles.payerSegment} role="radiogroup" aria-label="Type de payeur">
                <label
                  className={`${styles.payerChoice} ${payerType === 'patient' ? styles.payerChoiceActive : ''}`}
                  title="Facture directe au patient"
                >
                  <input
                    type="radio"
                    name="payerType"
                    value="patient"
                    className={styles.payerRadio}
                    checked={payerType === 'patient'}
                    onChange={() => {
                      setPayerType('patient');
                      setPartnershipId('');
                      setPreview(null);
                    }}
                  />
                  <span className={styles.payerChoiceIcon} aria-hidden="true">
                    <FiUser strokeWidth={2} size={15} />
                  </span>
                  <span className={styles.payerChoiceTitle}>Direct patient</span>
                </label>
                <label
                  className={`${styles.payerChoice} ${payerType === 'clinic' ? styles.payerChoiceActive : ''}`}
                  title="Facture à une institution"
                >
                  <input
                    type="radio"
                    name="payerType"
                    value="clinic"
                    className={styles.payerRadio}
                    checked={payerType === 'clinic'}
                    onChange={() => {
                      setPayerType('clinic');
                      setPartnershipId('');
                      setPreview(null);
                    }}
                  />
                  <span className={styles.payerChoiceIcon} aria-hidden="true">
                    <FiHome strokeWidth={2} size={15} />
                  </span>
                  <span className={styles.payerChoiceTitle}>Institution</span>
                </label>
                <label
                  className={`${styles.payerChoice} ${payerType === 'partner' ? styles.payerChoiceActive : ''}`}
                  title="Facturation inter-entreprises (partenaire)"
                >
                  <input
                    type="radio"
                    name="payerType"
                    value="partner"
                    className={styles.payerRadio}
                    checked={payerType === 'partner'}
                    onChange={() => {
                      setPayerType('partner');
                      setPreview(null);
                    }}
                  />
                  <span className={styles.payerChoiceIcon} aria-hidden="true">
                    <FiUsers strokeWidth={2} size={15} />
                  </span>
                  <span className={styles.payerChoiceTitle}>Partenaire</span>
                </label>
              </div>
            </div>

            <div
              className={styles.billContextCard}
              role="group"
              aria-label={
                payerType === 'patient'
                  ? 'Période facturée et patient (facturation directe)'
                  : payerType === 'clinic'
                    ? 'Période facturée et institution'
                    : payerType === 'partner'
                      ? 'Période facturée et partenaire'
                      : undefined
              }
            >
              {payerType === 'patient' ? (
                <div className={styles.billUnifiedLine}>
                  <label htmlFor="bill-period-ym" className={styles.billInlineLbl}>
                    Période
                  </label>
                  <InlineMonthYearPicker
                    inputId="bill-period-ym"
                    className={`${styles.billMonthPickerWrap} ${styles.billMonthPickerUnified}`}
                    value={`${String(periodYear).padStart(4, '0')}-${String(periodMonth).padStart(2, '0')}`}
                    onChange={handlePeriodYmPickerChange}
                    disabled={loadingLists}
                  />
                  <label
                    id="bill-patient-inline-label"
                    htmlFor="bill-period-client"
                    className={styles.billInlineLbl}
                    title="Bénéficiaire — montants affichés : transports à facturer au patient uniquement"
                  >
                    Patient
                  </label>
                  <ChipSelect
                    id="bill-period-client"
                    aria-labelledby="bill-patient-inline-label"
                    className={styles.billChipSelectGrow}
                    options={billPatientChipOptions}
                    value={clientId}
                    placeholder="— Choisir un patient —"
                    onChange={(v) => {
                      setClientId(v == null || v === '' ? '' : String(v));
                      setPreview(null);
                      setShowLinesPreview(false);
                    }}
                    disabled={loadingLists}
                    menuMinWidth={280}
                    filterable
                  />
                </div>
              ) : payerType === 'clinic' ? (
                <div className={styles.billUnifiedLine}>
                  <label htmlFor="bill-period-ym" className={styles.billInlineLbl}>
                    Période
                  </label>
                  <InlineMonthYearPicker
                    inputId="bill-period-ym"
                    className={`${styles.billMonthPickerWrap} ${styles.billMonthPickerUnified}`}
                    value={`${String(periodYear).padStart(4, '0')}-${String(periodMonth).padStart(2, '0')}`}
                    onChange={handlePeriodYmPickerChange}
                    disabled={loadingLists}
                  />
                  <label
                    id="bill-clinic-inline-label"
                    htmlFor="bill-period-clinic"
                    className={styles.billInlineLbl}
                    title="Institution à facturer"
                  >
                    Institution
                  </label>
                  <ChipSelect
                    id="bill-period-clinic"
                    aria-labelledby="bill-clinic-inline-label"
                    className={styles.billChipSelectGrow}
                    options={billClinicChipOptions}
                    value={clinicKey}
                    placeholder="— Choisir une institution —"
                    onChange={(v) => {
                      setClinicKey(v == null || v === '' ? '' : String(v));
                      setPreview(null);
                      setShowLinesPreview(false);
                    }}
                    disabled={loadingLists}
                    menuMinWidth={300}
                    filterable
                  />
                </div>
              ) : (
                <div className={styles.billUnifiedLine}>
                  <label htmlFor="bill-period-ym" className={styles.billInlineLbl}>
                    Période
                  </label>
                  <InlineMonthYearPicker
                    inputId="bill-period-ym"
                    className={`${styles.billMonthPickerWrap} ${styles.billMonthPickerUnified}`}
                    value={`${String(periodYear).padStart(4, '0')}-${String(periodMonth).padStart(2, '0')}`}
                    onChange={handlePeriodYmPickerChange}
                    disabled={loadingLists}
                  />
                  <label
                    id="bill-partner-inline-label"
                    htmlFor="bill-period-partner"
                    className={styles.billInlineLbl}
                    title="Entreprise partenaire à facturer"
                  >
                    Partenaire
                  </label>
                  <ChipSelect
                    id="bill-period-partner"
                    aria-labelledby="bill-partner-inline-label"
                    className={styles.billChipSelectGrow}
                    options={billPartnerChipOptions}
                    value={partnershipId}
                    placeholder="— Choisir une entreprise partenaire —"
                    onChange={(v) => {
                      setPartnershipId(v == null || v === '' ? '' : String(v));
                      setPreview(null);
                      setShowLinesPreview(false);
                    }}
                    disabled={loadingLists}
                    menuMinWidth={300}
                    filterable
                  />
                </div>
              )}
            </div>

            {payerType === 'patient' && clientId ? (
              <div className={styles.detectedPlan} data-testid="patient-invoice-summary">
                <div
                  className={styles.institutionSummary}
                  role="region"
                  aria-label="Facture patient"
                >
                  <div className={styles.institutionSummaryTitle}>
                    Facture {patientSummary.displayName}
                  </div>
                  <div className={styles.institutionSummaryPeriod}>
                    {periodLabelFr(periodYear, periodMonth)}
                  </div>
                  {patientSummary.hasBillable ? (
                    <>
                      <div className={styles.institutionSummaryTotals}>
                        <span data-testid="patient-summary-count">
                          {patientSummary.transportsCount} prestation
                          {patientSummary.transportsCount !== 1 ? 's' : ''}
                        </span>
                        <strong data-testid="patient-summary-amount">
                          {formatCurrencyCHF(patientSummary.totalHt)}
                        </strong>
                      </div>
                      <p className={styles.institutionSummaryNote}>
                        Toutes les prestations à charge de ce patient sont incluses.
                      </p>
                    </>
                  ) : (
                    <p className={styles.institutionSummaryNote}>
                      {patientSummary.blocked
                        ? 'Identité ou destinataire à compléter avant facturation.'
                        : 'Aucune prestation à facturer à ce patient pour cette période.'}
                    </p>
                  )}
                </div>
              </div>
            ) : null}

            {payerType === 'partner' && partnershipId ? (
              <div className={styles.detectedPlan} data-testid="partner-invoice-summary">
                <div
                  className={styles.institutionSummary}
                  role="region"
                  aria-label="Facture partenaire"
                >
                  <div className={styles.institutionSummaryTitle}>
                    Facture {partnerSummary.displayName}
                  </div>
                  <div className={styles.institutionSummaryPeriod}>
                    {periodLabelFr(periodYear, periodMonth)}
                  </div>
                  {partnerSummary.hasBillable ? (
                    <>
                      <div className={styles.institutionSummaryTotals}>
                        <span data-testid="partner-summary-count">
                          {partnerSummary.transportsCount} prestation
                          {partnerSummary.transportsCount !== 1 ? 's' : ''}
                        </span>
                        <strong data-testid="partner-summary-amount">
                          {formatCurrencyCHF(partnerSummary.totalHt)}
                        </strong>
                      </div>
                      <p className={styles.institutionSummaryNote}>
                        Toutes les prestations à charge de ce partenaire sont incluses.
                      </p>
                    </>
                  ) : (
                    <p className={styles.institutionSummaryNote}>
                      Aucune prestation à facturer à ce partenaire pour cette période.
                    </p>
                  )}
                  {partnerSummary.excluded.visible ? (
                    <div
                      className={styles.institutionExcluded}
                      data-testid="partner-excluded-warning"
                    >
                      <span>
                        {partnerSummary.excluded.count === 1
                          ? "1 prestation n'est pas encore facturable"
                          : `${partnerSummary.excluded.count} prestations ne sont pas encore facturables`}
                      </span>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}

            {payerType === 'clinic' && clinicCompanyId ? (
              <div className={styles.detectedPlan} data-testid="institution-invoice-plan">
                {showInstitutionPlanSkeleton ? (
                  <p className={styles.formHint} role="status">
                    Calcul des prestations à charge de cette institution…
                  </p>
                ) : institutionPlanIsCurrent ? (
                  <>
                    <div
                      className={styles.institutionSummary}
                      role="region"
                      aria-label="Facture institution"
                    >
                      <div className={styles.institutionSummaryTitle}>
                        Facture {selectedClinic?.institution_name || institutionSummary.displayName}
                      </div>
                      <div className={styles.institutionSummaryPeriod}>
                        {`${String(MONTHS_FR[periodMonth - 1] || '').charAt(0).toUpperCase()}${String(MONTHS_FR[periodMonth - 1] || '').slice(1)} ${periodYear}`}
                      </div>
                      {institutionSummary.hasBillable ? (
                        <>
                          <div className={styles.institutionSummaryTotals}>
                            <span data-testid="institution-summary-count">
                              {institutionSummary.transportsCount} prestation
                              {institutionSummary.transportsCount !== 1 ? 's' : ''}
                            </span>
                            <strong data-testid="institution-summary-amount">
                              {formatCurrencyCHF(institutionSummary.totalHt)}
                            </strong>
                          </div>
                          <p className={styles.institutionSummaryNote}>
                            Toutes les prestations encore à facturer à cette institution
                            sont incluses. Un changement de payeur côté institution
                            (patient ↔ clinique) est pris en compte tout de suite.
                          </p>
                          {institutionPlanRefreshing || institutionPlanLoading ? (
                            <p
                              className={styles.planLiveHint}
                              data-testid="institution-plan-live-hint"
                              role="status"
                            >
                              Actualisation…
                            </p>
                          ) : null}
                        </>
                      ) : (
                        <p className={styles.institutionSummaryNote}>
                          Aucune prestation à facturer à cette institution pour cette période.
                        </p>
                      )}
                    </div>
                    {institutionSummary.alreadyInvoiced.visible ? (
                      <div
                        className={styles.institutionAlreadyInvoiced}
                        data-testid="institution-already-invoiced-warning"
                      >
                        <span>
                          {institutionSummary.alreadyInvoiced.count === 1
                            ? '1 prestation déjà facturée'
                            : `${institutionSummary.alreadyInvoiced.count} prestations déjà facturées`}
                        </span>
                        <button
                          type="button"
                          className={styles.linkBtn}
                          onClick={() =>
                            setShowAlreadyInvoicedLines((openLines) => !openLines)
                          }
                          data-testid="institution-already-invoiced-toggle"
                        >
                          {showAlreadyInvoicedLines ? 'Masquer' : 'Voir'}
                        </button>
                      </div>
                    ) : null}
                    {showAlreadyInvoicedLines ? (
                      <div
                        className={styles.alreadyInvoicedDetail}
                        data-testid="institution-already-invoiced-lines"
                      >
                        <div className={styles.alreadyInvoicedDetailTitle}>
                          {institutionAlreadyInvoicedRows.length === 1
                            ? 'Déjà facturée'
                            : 'Déjà facturées'}
                        </div>
                        <ul className={styles.excludedDetailList}>
                          {institutionAlreadyInvoicedRows.map((row) => (
                            <li key={row.bookingId} className={styles.excludedDetailItem}>
                              <span>
                                #{row.bookingId} · {formatCurrencyCHF(row.amountHt)}
                              </span>
                              <span>{alreadyInvoicedInvoiceLabel(row)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {institutionSummary.excluded.visible ? (
                      <div
                        className={styles.institutionExcluded}
                        data-testid="institution-excluded-warning"
                      >
                        <span>
                          {institutionSummary.excluded.count === 1
                            ? '1 prestation non facturable'
                            : `${institutionSummary.excluded.count} prestations non facturables`}
                        </span>
                        <button
                          type="button"
                          className={styles.linkBtn}
                          onClick={() => setShowExcludedLines((openLines) => !openLines)}
                          data-testid="institution-excluded-toggle"
                        >
                          {showExcludedLines ? 'Masquer' : 'Voir'}
                        </button>
                      </div>
                    ) : null}
                    {institutionPlanLiveError ? (
                      <div
                        className={styles.planLiveError}
                        data-testid="institution-plan-live-error"
                        role="alert"
                      >
                        <span>{institutionPlanLiveError}</span>
                        <button
                          type="button"
                          className={styles.linkBtn}
                          onClick={() => void fetchInstitutionPlan({ silent: true })}
                        >
                          Réessayer
                        </button>
                      </div>
                    ) : null}
                    {showExcludedLines ? (
                      <div className={styles.excludedDetail} data-testid="institution-excluded-lines">
                        <div className={styles.excludedDetailTitle}>
                          {excludedBlockTitle(institutionExcludedRows.length)}
                        </div>
                        <ul className={styles.excludedDetailList}>
                          {institutionExcludedRows.map((row) => {
                            const who = excludedRowWhoLabel(row);
                            const when = formatPreviewDayMonth(row.scheduledAt);
                            return (
                              <li
                                key={row.bookingId}
                                className={styles.excludedDetailItem}
                                data-testid={`dispute-excluded-row-${row.bookingId}`}
                              >
                                <p className={styles.excludedDetailWhy}>{exclusionWhyText(row)}</p>
                                <div className={styles.excludedDetailWho}>
                                  {when ? (
                                    <span className={styles.excludedDetailDate}>{when}</span>
                                  ) : null}
                                  {who ? <span>{who}</span> : null}
                                  <strong>{formatCurrencyCHF(row.amountHt)}</strong>
                                </div>
                                {canTreatDispute(row) ? (
                                  <button
                                    type="button"
                                    className={styles.disputeTreatBtn}
                                    data-testid={`dispute-treat-${row.bookingId}`}
                                    onClick={() => setTreatingExcludedRow(row)}
                                  >
                                    Traiter la contestation
                                  </button>
                                ) : null}
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    ) : null}
                  </>
                ) : null}
              </div>
            ) : null}

            {shouldShowSimpleInvoiceLinesPreview({
              showLinesPreview,
              hasPreparedDraft: Boolean(draftInvoiceStub),
            }) ? (
              <section
                className={styles.invoiceLinesPreview}
                data-testid="invoice-lines-preview"
                aria-label="Lignes qui seront facturées"
              >
                <h3 className={styles.invoiceLinesPreviewTitle}>Lignes qui seront facturées</h3>
                {previewLoading ? (
                  <p className={styles.formHint} role="status">
                    Chargement des lignes…
                  </p>
                ) : linesPreview.rows.length === 0 ? (
                  <p className={styles.formHint}>Aucune ligne à afficher pour cette sélection.</p>
                ) : (
                  <>
                    <ul className={styles.invoiceLinesList}>
                      {linesPreview.rows.map((row) => (
                        <li key={row.key} className={styles.invoiceLineItem}>
                          <span className={styles.invoiceLineDate}>{row.dateLabel}</span>
                          <div className={styles.invoiceLineBody}>
                            <div className={styles.invoiceLinePatient}>{row.patientName}</div>
                            <div className={styles.invoiceLineRoute}>
                              <span>{row.route}</span>
                              {row.isRoundTrip ? (
                                <span className={styles.invoiceLineAr}>
                                  A/R · {row.segmentsCount} course
                                  {row.segmentsCount !== 1 ? 's' : ''}
                                </span>
                              ) : null}
                            </div>
                            {row.isRoundTrip ? (
                              <details className={styles.invoiceLineLegs}>
                                <summary>Voir aller / retour</summary>
                                <p>
                                  Aller booking #{row.outboundBookingId}
                                  {row.outboundAmountHt != null
                                    ? ` · ${formatCurrencyCHF(row.outboundAmountHt)}`
                                    : ''}
                                </p>
                                <p>
                                  Retour booking #{row.returnBookingId}
                                  {row.returnAmountHt != null
                                    ? ` · ${formatCurrencyCHF(row.returnAmountHt)}`
                                    : ''}
                                </p>
                              </details>
                            ) : null}
                          </div>
                          <strong className={styles.invoiceLineAmount}>
                            {formatCurrencyCHF(row.amountHt)}
                          </strong>
                        </li>
                      ))}
                    </ul>
                    <div className={styles.invoiceLinesFooter} data-testid="invoice-lines-preview-meta">
                      <span>
                        {linesPreview.visualLineCount} ligne
                        {linesPreview.visualLineCount !== 1 ? 's' : ''} de facture
                        {' · '}
                        {linesPreview.prestationCount} prestation
                        {linesPreview.prestationCount !== 1 ? 's' : ''}
                      </span>
                      <strong>{formatCurrencyCHF(linesPreview.totalHt)}</strong>
                    </div>
                  </>
                )}
              </section>
            ) : null}

            {loadingLists && (
              <p className={styles.formHint} role="status">
                Chargement des listes (patients, institutions, partenaires)…
              </p>
            )}
          </div>

          {error && <div className={styles.err}>{error}</div>}

          {shouldShowDraftInvoiceToolbar({
            hasPreparedDraft: Boolean(draftInvoiceStub),
          }) ? (
            <div className={styles.draftEditorMount} data-testid="invoice-draft-editor">
              <DraftInvoiceEditorPanel
                key={draftInvoiceStub?.id ?? 'draft-stub'}
                open
                initialInvoice={draftInvoiceStub}
                companyId={companyId}
                toolbarSubtitle={periodPreviewBarSubtitle}
                onUpdated={handleDraftPanelUpdated}
                onOpenSendEmail={onOpenSendEmail}
                onMarkAsSent={onMarkAsSent}
              />
            </div>
          ) : null}

          {false && composerPhase === 'draft' && hasAssemblyLines && preview && (
            <div
              ref={periodPdfWrapRef}
              className={`${draftEditorStyles.draftPdfWrap}${
                assemblyPreviewExpanded ? ` ${draftEditorStyles.draftPdfWrapExpanded}` : ''
              }`}
            >
              <div
                className={draftEditorStyles.draftPdfBar}
                role="toolbar"
                aria-label="Brouillon et aperçu facture"
              >
                <div className={draftEditorStyles.draftPdfBarLeft}>
                  <div className={draftEditorStyles.draftPdfBarHeadText}>
                    <h3 className={draftEditorStyles.draftPdfHeadTitle}>Aperçu facture</h3>
                    <p className={draftEditorStyles.draftPdfHeadSubtitle}>{periodPreviewBarSubtitle}</p>
                  </div>
                </div>
                <div className={draftEditorStyles.draftPdfBarRightMerged}>
                  <div
                    className={draftEditorStyles.draftPdfBarToolGroup}
                    role="group"
                    aria-label="Édition du brouillon"
                  >
                    <button
                      type="button"
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly}${
                        periodEditSheet === 'remise' || hasActivePeriodRemise
                          ? ` ${draftEditorStyles.draftPdfToolBtnActive}`
                          : ''
                      }`}
                      title="Remises"
                      aria-label="Remises"
                      aria-pressed={Boolean(periodEditSheet === 'remise' || hasActivePeriodRemise)}
                      onClick={() => openPeriodSheet('remise')}
                    >
                      <FiPercent size={16} aria-hidden />
                    </button>
                    <button
                      type="button"
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly}${
                        periodEditSheet === 'addLine' ? ` ${draftEditorStyles.draftPdfToolBtnActive}` : ''
                      }`}
                      title="Ligne supplémentaire HT"
                      aria-label="Ajouter une ligne supplémentaire HT"
                      aria-pressed={periodEditSheet === 'addLine'}
                      onClick={() => openPeriodSheet('addLine')}
                    >
                      <FiPlus size={16} aria-hidden />
                    </button>
                    <button
                      type="button"
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly}${
                        periodEditSheet === 'lines' ? ` ${draftEditorStyles.draftPdfToolBtnActive}` : ''
                      }`}
                      title="Modifier les lignes (libellés, montants, notes)"
                      aria-label="Ouvrir l’édition des lignes sous l’aperçu PDF"
                      aria-pressed={periodEditSheet === 'lines'}
                      onClick={() => openPeriodSheet('lines')}
                    >
                      <FiList size={16} aria-hidden />
                    </button>
                    <button
                      type="button"
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly}`}
                      disabled={previewLoading || generateLoading || !canPreview()}
                      title="Actualiser l’aperçu HTML de la période (pas de régénération PDF)"
                      aria-label="Actualiser l’aperçu HTML depuis le serveur"
                      onClick={() => void runPreview()}
                    >
                      <FiRefreshCw
                        size={16}
                        aria-hidden
                        className={previewLoading ? styles.btnIconSpin : undefined}
                      />
                    </button>
                  </div>
                  <div className={draftEditorStyles.draftPdfBarToolGroup} role="group" aria-label="Affichage">
                    <button
                      type="button"
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly}`}
                      title={
                        assemblyPreviewExpanded
                          ? 'Réduire la zone d’aperçu'
                          : 'Agrandir la zone d’aperçu'
                      }
                      aria-label={
                        assemblyPreviewExpanded
                          ? 'Réduire la zone d’aperçu PDF'
                          : 'Agrandir la zone d’aperçu PDF'
                      }
                      aria-pressed={assemblyPreviewExpanded}
                      onClick={() => setAssemblyPreviewExpanded((v) => !v)}
                    >
                      {assemblyPreviewExpanded ? (
                        <FiChevronsUp size={18} aria-hidden />
                      ) : (
                        <FiChevronsDown size={18} aria-hidden />
                      )}
                    </button>
                    <button
                      type="button"
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly}`}
                      title={
                        periodPdfFullscreen ? 'Quitter le plein écran' : 'Plein écran (navigateur)'
                      }
                      aria-label={
                        periodPdfFullscreen
                          ? 'Quitter le plein écran'
                          : 'Plein écran dans le navigateur'
                      }
                      aria-pressed={periodPdfFullscreen}
                      onClick={() => handlePeriodBrowserFullscreen()}
                    >
                      {periodPdfFullscreen ? (
                        <FiMinimize2 size={18} aria-hidden />
                      ) : (
                        <FiMaximize2 size={18} aria-hidden />
                      )}
                    </button>
                  </div>
                  <div
                    className={`${draftEditorStyles.draftPdfBarToolGroup} ${draftEditorStyles.draftPdfBarToolGroupFile}`}
                    role="group"
                    aria-label="Fichier PDF"
                  >
                    <button
                      type="button"
                      disabled
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly} ${draftEditorStyles.draftPdfToolLink}`}
                      title={PERIOD_PREVIEW_DISABLED_HINT}
                      aria-label={`Générer et télécharger le PDF — ${PERIOD_PREVIEW_DISABLED_HINT}`}
                    >
                      <FiDownload size={18} aria-hidden />
                    </button>
                    <button
                      type="button"
                      disabled
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly}`}
                      title={PERIOD_PREVIEW_DISABLED_HINT}
                      aria-label={`Imprimer la facture — ${PERIOD_PREVIEW_DISABLED_HINT}`}
                    >
                      <FiPrinter size={18} aria-hidden />
                    </button>
                    <button
                      type="button"
                      disabled
                      className={`${draftEditorStyles.draftPdfToolBtn} ${draftEditorStyles.draftPdfToolBtnIconOnly} ${draftEditorStyles.draftPdfToolLink}`}
                      title={PERIOD_PREVIEW_DISABLED_HINT}
                      aria-label={`Ouvrir le PDF dans un nouvel onglet — ${PERIOD_PREVIEW_DISABLED_HINT}`}
                    >
                      <FiExternalLink size={18} aria-hidden />
                    </button>
                  </div>
                </div>
              </div>
              <div
                className={`${draftEditorStyles.draftPdfViewerStack}${
                  periodEditSheet ? ` ${draftEditorStyles.draftPdfViewerStackWithSheet}` : ''
                }`}
              >
                {mergedPeriodInvoice ? (
                  <InvoiceLivePreview
                    invoice={mergedPeriodInvoice}
                    companyVatApplicable={companyVatApplicable}
                    auditableRoundTrip
                    className={draftEditorStyles.draftLivePreviewMount}
                  />
                ) : null}
                {periodEditSheet ? (
                  <div
                    className={`${draftEditorStyles.draftPdfLineSheetFlow}${
                      periodEditSheet === 'lines' || periodEditSheet === 'addLine'
                        ? ` ${draftEditorStyles.draftPdfLineSheetFlowLines}`
                        : ''
                    }${periodEditSheet === 'remise' ? ` ${draftEditorStyles.draftPdfLineSheetFlowRemise}` : ''}`}
                  >
                    <div
                      className={draftEditorStyles.draftPdfLineSheetDoc}
                      role="dialog"
                      aria-modal="true"
                      aria-labelledby={
                        periodEditSheet === 'lines'
                          ? periodLinesHeadingId
                          : periodEditSheet === 'remise'
                            ? periodRemiseHeadingId
                            : periodAddLineHeadingId
                      }
                    >
                      <div className={draftEditorStyles.draftPdfLineSheetInner}>
                        {periodEditSheet === 'lines' && syntheticPeriodInvoice ? (
                          <div className={draftEditorStyles.linesSheetPanel}>
                            <header className={draftEditorStyles.linesSheetHead}>
                              <h3
                                className={draftEditorStyles.linesSheetHeading}
                                id={periodLinesHeadingId}
                              >
                                Lignes
                              </h3>
                            </header>
                            <div className={draftEditorStyles.linesSheetEditor}>
                              {showPeriodLinesToolbar ? (
                                <div
                                  className={`${draftEditorStyles.linesToolbar} ${draftEditorStyles.linesToolbarDense}`}
                                >
                                  <div className={draftEditorStyles.linesToolbarFilterWrap}>
                                    <FiSearch
                                      className={draftEditorStyles.linesToolbarFilterIcon}
                                      size={16}
                                      aria-hidden
                                    />
                                    <input
                                      type="search"
                                      className={draftEditorStyles.linesToolbarInput}
                                      placeholder="Client, date (JJ.MM.AAAA), libellé, montant, n° ligne…"
                                      title="Ex. : Bouchardy · Jean-michel Bouchardy · 02.05.2026 · 02.05 · mai · HUG · 80"
                                      value={periodLineFilter}
                                      onChange={(e) => setPeriodLineFilter(e.target.value)}
                                      autoComplete="off"
                                      aria-label="Filtrer les lignes de l’aperçu période"
                                    />
                                  </div>
                                  <span className={draftEditorStyles.linesToolbarMeta}>
                                    {periodFilteredLines.length === 0
                                      ? 'Aucun résultat'
                                      : `Lignes ${periodLineRangeStart}–${periodLineRangeEnd} sur ${periodFilteredLines.length}`}
                                  </span>
                                  {periodTotalLinePages > 1 ? (
                                    <div className={draftEditorStyles.linesToolbarPager}>
                                      <button
                                        type="button"
                                        className={draftEditorStyles.btnPager}
                                        disabled={periodEffectiveLinePage <= 1}
                                        title="Page précédente"
                                        aria-label="Page précédente"
                                        onClick={() => setPeriodLinePage((p) => Math.max(1, p - 1))}
                                      >
                                        <FiChevronLeft size={18} aria-hidden />
                                      </button>
                                      <span className={draftEditorStyles.linesToolbarMeta}>
                                        {periodEffectiveLinePage} / {periodTotalLinePages}
                                      </span>
                                      <button
                                        type="button"
                                        className={draftEditorStyles.btnPager}
                                        disabled={periodEffectiveLinePage >= periodTotalLinePages}
                                        title="Page suivante"
                                        aria-label="Page suivante"
                                        onClick={() =>
                                          setPeriodLinePage((p) =>
                                            Math.min(periodTotalLinePages, p + 1)
                                          )
                                        }
                                      >
                                        <FiChevronRight size={18} aria-hidden />
                                      </button>
                                    </div>
                                  ) : null}
                                </div>
                              ) : null}
                              <div
                                className={`${draftEditorStyles.tableScroll} ${draftEditorStyles.tableScrollDense}${
                                  periodEditorLines.length >= HEAVY_LINES_THRESHOLD
                                    ? ` ${draftEditorStyles.tableScrollHeavy}`
                                    : ''
                                }`}
                              >
                                {periodEditorLines.length > 0 && periodFilteredLines.length > 0 ? (
                                  <div
                                    className={draftEditorStyles.linesColumnLegend}
                                    aria-hidden="true"
                                  >
                                    <span>Libellé</span>
                                    <span className={draftEditorStyles.linesColumnLegendHt}>
                                      HT (CHF)
                                    </span>
                                    <span>Note</span>
                                    <span className={draftEditorStyles.linesColumnLegendActions}> </span>
                                  </div>
                                ) : null}
                                <table
                                  className={`${draftEditorStyles.table} ${draftEditorStyles.tableDense}`}
                                >
                                  <caption className={draftEditorStyles.srOnly}>
                                    Lignes d’aperçu période&nbsp;: libellé, montant HT, note optionnelle, puis
                                    réinitialiser les surcharges ou exclure le transport de la facture.
                                  </caption>
                                  <colgroup>
                                    <col className={draftEditorStyles.colDesc} />
                                    <col className={draftEditorStyles.colHt} />
                                    <col className={draftEditorStyles.colNoteCol} />
                                    <col className={draftEditorStyles.colActions} />
                                  </colgroup>
                                  <tbody>
                                    {periodEditorLines.length === 0 ? (
                                      <tr>
                                        <td colSpan={4} className={draftEditorStyles.tableEmptyTight}>
                                          Aucune ligne.
                                        </td>
                                      </tr>
                                    ) : periodFilteredLines.length === 0 ? (
                                      <tr>
                                        <td colSpan={4} className={draftEditorStyles.tableEmptyTight}>
                                          Rien pour ce filtre.
                                        </td>
                                      </tr>
                                    ) : (
                                      periodPaginatedLines.map((ln) => {
                                        const descVal =
                                          periodLinePatch[ln.id]?.description != null
                                            ? periodLinePatch[ln.id].description
                                            : (ln.description ?? '');
                                        const htVal =
                                          periodLinePatch[ln.id]?.line_total !== undefined
                                            ? periodLinePatch[ln.id].line_total
                                            : String(ln.line_total ?? '');
                                        const noteVal =
                                          periodLinePatch[ln.id]?.adjustment_note !== undefined
                                            ? periodLinePatch[ln.id].adjustment_note
                                            : (ln.adjustment_note ?? '');
                                        const hasPatch = Boolean(periodLinePatch[ln.id]);
                                        const rawBid = bookingIdFromPeriodLineId(ln.id);
                                        const typeLbl = periodLineTypeLabel(ln.type);
                                        const rowTitle =
                                          rawBid != null ? `#${rawBid} · ${typeLbl}` : `${ln.id} · ${typeLbl}`;
                                        const descId = `period-line-desc-${ln.id}`;
                                        const htId = `period-line-ht-${ln.id}`;
                                        const noteId = `period-line-note-${ln.id}`;
                                        const canExclude = rawBid != null;
                                        const showArLegExclude = canShowRoundTripLegExcludeActions(ln);
                                        const rowClassNames = [
                                          isRoundTripPreviewHiddenLine(ln)
                                            ? draftEditorStyles.rowRoundTripReturn
                                            : '',
                                        ]
                                          .filter(Boolean)
                                          .join(' ');
                                        return (
                                          <tr
                                            key={ln.id}
                                            title={rowTitle}
                                            className={rowClassNames || undefined}
                                          >
                                            <td className={draftEditorStyles.colDescCell}>
                                              <div className={draftEditorStyles.lineEditorDescStack}>
                                                <InvoiceLineEditorContext
                                                  line={ln}
                                                  styles={draftEditorStyles}
                                                  legActions={
                                                    showArLegExclude
                                                      ? {
                                                          enabled: true,
                                                          returnTitle:
                                                            'Conserver l’aller, retirer le retour de la facture (aperçu)',
                                                          outboundTitle:
                                                            'Conserver le retour, retirer l’aller de la facture (aperçu)',
                                                          onExcludeLeg: (leg) =>
                                                            excludePeriodRoundTripLeg(ln.id, leg),
                                                        }
                                                      : undefined
                                                  }
                                                />
                                                <div className={draftEditorStyles.denseDesc}>
                                                  <textarea
                                                    id={descId}
                                                    className={draftEditorStyles.denseTextarea}
                                                    rows={1}
                                                    value={descVal}
                                                    onChange={(e) =>
                                                      setPeriodLinePatch((prev) => ({
                                                        ...prev,
                                                        [ln.id]: {
                                                          ...prev[ln.id],
                                                          description: e.target.value,
                                                        },
                                                      }))
                                                    }
                                                    aria-label={`Libellé · ${rowTitle}`}
                                                    title={rowTitle}
                                                  />
                                                </div>
                                              </div>
                                            </td>
                                            <td className={draftEditorStyles.colHtCell}>
                                              <input
                                                id={htId}
                                                className={`${draftEditorStyles.denseHt} ${draftEditorStyles.denseHtGrow}`}
                                                type="text"
                                                inputMode="decimal"
                                                value={htVal}
                                                onChange={(e) =>
                                                  setPeriodLinePatch((prev) => ({
                                                    ...prev,
                                                    [ln.id]: {
                                                      ...prev[ln.id],
                                                      line_total: e.target.value,
                                                    },
                                                  }))
                                                }
                                                aria-label="Montant HT"
                                              />
                                            </td>
                                            <td className={draftEditorStyles.colNote}>
                                              <textarea
                                                id={noteId}
                                                className={`${draftEditorStyles.denseTextarea} ${draftEditorStyles.denseTextareaNote}`}
                                                rows={1}
                                                value={noteVal}
                                                onChange={(e) =>
                                                  setPeriodLinePatch((prev) => ({
                                                    ...prev,
                                                    [ln.id]: {
                                                      ...prev[ln.id],
                                                      adjustment_note: e.target.value,
                                                    },
                                                  }))
                                                }
                                                aria-label="Note"
                                              />
                                            </td>
                                            <td className={draftEditorStyles.colActionsCell}>
                                              <div className={draftEditorStyles.denseActions}>
                                                <button
                                                  type="button"
                                                  className={`${draftEditorStyles.btnIconOkXs} ${draftEditorStyles.btnLineSave}`}
                                                  disabled={!hasPatch}
                                                  title="Réinitialiser les modifications locales sur cette ligne"
                                                  aria-label={`Réinitialiser la ligne ${ln.id}`}
                                                  onClick={() => resetPeriodLinePatch(ln.id)}
                                                >
                                                  <FiCheck size={14} aria-hidden />
                                                </button>
                                                {canExclude ? (
                                                  <button
                                                    type="button"
                                                    className={`${draftEditorStyles.btnTrashXs} ${draftEditorStyles.danger}`}
                                                    title={
                                                      showArLegExclude
                                                        ? 'Retirer l’aller-retour complet de la facture (aperçu)'
                                                        : 'Retirer ce transport de la facture (aperçu)'
                                                    }
                                                    aria-label={`Exclure la ligne ${ln.id}`}
                                                    onClick={() => excludePeriodLineFromPreview(ln.id)}
                                                  >
                                                    <FiTrash2 size={13} aria-hidden />
                                                  </button>
                                                ) : null}
                                              </div>
                                            </td>
                                          </tr>
                                        );
                                      })
                                    )}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          </div>
                        ) : null}
                        {periodEditSheet === 'remise' && syntheticPeriodInvoice ? (
                          <div className={draftEditorStyles.remiseSheet}>
                            <div className={draftEditorStyles.remiseSheetTitleRow}>
                              <h3
                                className={draftEditorStyles.remiseSheetTitle}
                                id={periodRemiseHeadingId}
                              >
                                Remises
                              </h3>
                              {hasActivePeriodRemise ? (
                                <span className={draftEditorStyles.remiseActiveBadge}>
                                  Remise active
                                  {(() => {
                                    if (periodRemisePercentMode === PERIOD_REMISE_PCT_MODE.global) {
                                      const p = parseFloat(
                                        String(periodGlobalRemiseInput).replace(',', '.')
                                      );
                                      if (Number.isFinite(p) && p > 0) return ` : ${p} %`;
                                    }
                                    if (periodRemisePercentMode === PERIOD_REMISE_PCT_MODE.perLine) {
                                      return ' : par ligne';
                                    }
                                    return periodDeductionLines.length > 0 ? ' · déduction HT' : '';
                                  })()}
                                </span>
                              ) : null}
                            </div>

                            <div
                              className={draftEditorStyles.remiseModeSeg}
                              role="group"
                              aria-label="Remise en pourcentage sur les transports"
                            >
                              <button
                                type="button"
                                className={
                                  periodRemisePercentMode === PERIOD_REMISE_PCT_MODE.global
                                    ? draftEditorStyles.remiseModeSegBtnActive
                                    : draftEditorStyles.remiseModeSegBtn
                                }
                                onClick={() => setPeriodRemisePercentMode(PERIOD_REMISE_PCT_MODE.global)}
                              >
                                Globale
                              </button>
                              <button
                                type="button"
                                className={
                                  periodRemisePercentMode === PERIOD_REMISE_PCT_MODE.perLine
                                    ? draftEditorStyles.remiseModeSegBtnActive
                                    : draftEditorStyles.remiseModeSegBtn
                                }
                                onClick={() => setPeriodRemisePercentMode(PERIOD_REMISE_PCT_MODE.perLine)}
                              >
                                Par ligne
                              </button>
                            </div>

                            {periodRemisePercentMode === PERIOD_REMISE_PCT_MODE.global && (
                              <>
                                <section
                                  className={draftEditorStyles.remiseSheetSection}
                                  aria-label="Remise globale"
                                >
                                  <div
                                    className={`${draftEditorStyles.formRow} ${draftEditorStyles.formRowHarmonized} ${draftEditorStyles.remiseSheetRow}`}
                                  >
                                    <label
                                      className={draftEditorStyles.srOnly}
                                      htmlFor="bill-period-gd-pct-toolbar"
                                    >
                                      Pourcentage
                                    </label>
                                    <input
                                      id="bill-period-gd-pct-toolbar"
                                      className={draftEditorStyles.input}
                                      type="text"
                                      inputMode="decimal"
                                      placeholder="%"
                                      autoComplete="off"
                                      value={periodGlobalRemiseInput}
                                      onChange={(e) => setPeriodGlobalRemiseInput(e.target.value)}
                                    />
                                    <input
                                      className={draftEditorStyles.inputGrow}
                                      type="text"
                                      placeholder="Note"
                                      value={periodRemiseNote}
                                      onChange={(e) => setPeriodRemiseNote(e.target.value)}
                                    />
                                    <button
                                      type="button"
                                      className={draftEditorStyles.btn}
                                      disabled={!syntheticPeriodInvoice.lines?.length}
                                      onClick={() => handlePeriodApplyGlobalRemise()}
                                    >
                                      Appliquer
                                    </button>
                                    <button
                                      type="button"
                                      className={draftEditorStyles.btnMuted}
                                      disabled={!hasActivePeriodRemise}
                                      onClick={() => handlePeriodRemoveRemise()}
                                    >
                                      Retirer
                                    </button>
                                  </div>
                                </section>
                                <section
                                  className={draftEditorStyles.remiseSheetSection}
                                  aria-label="Déduction fixe HT"
                                >
                                  <div className={draftEditorStyles.remiseSheetSectionHead}>
                                    Déduction CHF HT
                                  </div>
                                  <p className={draftEditorStyles.remiseSheetFoot}>
                                    Dans cet aperçu, la déduction apparaît comme ligne personnalisée en montant HT
                                    négatif ; le total TTC estimé se met à jour au-dessus. La facture définitive se
                                    règle à l’étape « Préparer la facture ».
                                  </p>
                                  <div
                                    className={`${draftEditorStyles.formRow} ${draftEditorStyles.formRowHarmonized} ${draftEditorStyles.remiseSheetRow}`}
                                  >
                                    <input
                                      className={draftEditorStyles.inputGrow}
                                      type="text"
                                      placeholder="Libellé"
                                      value={periodFreeDeductionDesc}
                                      onChange={(e) => setPeriodFreeDeductionDesc(e.target.value)}
                                      aria-label="Libellé"
                                    />
                                    <label
                                      className={draftEditorStyles.srOnly}
                                      htmlFor="bill-period-free-remise-amt-toolbar"
                                    >
                                      Montant HT
                                    </label>
                                    <input
                                      id="bill-period-free-remise-amt-toolbar"
                                      className={draftEditorStyles.input}
                                      type="text"
                                      inputMode="decimal"
                                      placeholder="CHF HT"
                                      autoComplete="off"
                                      value={periodFreeDeductionAmt}
                                      onChange={(e) => setPeriodFreeDeductionAmt(e.target.value)}
                                    />
                                    <button
                                      type="button"
                                      className={draftEditorStyles.btn}
                                      onClick={() => handlePeriodAddDeductionLine()}
                                    >
                                      Ajouter
                                    </button>
                                  </div>
                                  {periodDeductionLines.length > 0 ? (
                                    <ul className={styles.periodSheetExtraList}>
                                      {periodDeductionLines.map((d) => (
                                        <li key={d.id}>
                                          <span>{d.description}</span>
                                          {' · '}
                                          <strong>{String(d.htNegative)}</strong>
                                          {' CHF HT '}
                                          <button
                                            type="button"
                                            className={styles.periodSheetExtraRemove}
                                            onClick={() =>
                                              setPeriodDeductionLines((prev) =>
                                                prev.filter((x) => x.id !== d.id)
                                              )
                                            }
                                          >
                                            Retirer
                                          </button>
                                        </li>
                                      ))}
                                    </ul>
                                  ) : null}
                                </section>
                              </>
                            )}

                            {periodRemisePercentMode === PERIOD_REMISE_PCT_MODE.perLine && (
                              <section
                                className={draftEditorStyles.remiseSheetSection}
                                aria-label="Remise par ligne"
                              >
                                {syntheticPeriodInvoice.lines.length > 0 && (
                                  <ul className={draftEditorStyles.remisePerLineList}>
                                    {syntheticPeriodInvoice.lines.map((line) => {
                                      const lid = line.id;
                                      const rawDesc =
                                        String(line.description || '').trim() || 'Transport';
                                      const shortDesc =
                                        rawDesc.length > 72 ? `${rawDesc.slice(0, 69)}…` : rawDesc;
                                      const ht = line.line_total;
                                      const htLabel = Number.isFinite(Number(ht))
                                        ? formatCurrencyCHF(Number(ht))
                                        : '—';
                                      return (
                                        <li key={lid} className={draftEditorStyles.remisePerLineItem}>
                                          <div
                                            className={draftEditorStyles.remisePerLineDesc}
                                            title={rawDesc}
                                          >
                                            {shortDesc}
                                          </div>
                                          <label
                                            className={draftEditorStyles.remisePerLinePctWrap}
                                            htmlFor={`bill-period-pl-pct-${lid}`}
                                          >
                                            <span className={draftEditorStyles.srOnly}>Pourcentage</span>
                                            <input
                                              id={`bill-period-pl-pct-${lid}`}
                                              className={draftEditorStyles.remisePerLinePctInput}
                                              type="text"
                                              inputMode="decimal"
                                              placeholder="%"
                                              autoComplete="off"
                                              value={periodPerLineRemisePct[lid] ?? ''}
                                              onChange={(e) =>
                                                setPeriodPerLineRemisePct((prev) => ({
                                                  ...prev,
                                                  [lid]: e.target.value,
                                                }))
                                              }
                                            />
                                          </label>
                                          <span className={draftEditorStyles.remisePerLineHt}>{htLabel}</span>
                                        </li>
                                      );
                                    })}
                                  </ul>
                                )}
                                <div
                                  className={`${draftEditorStyles.formRow} ${draftEditorStyles.formRowHarmonized} ${draftEditorStyles.remiseSheetRow}`}
                                >
                                  <button
                                    type="button"
                                    className={draftEditorStyles.btn}
                                    disabled={syntheticPeriodInvoice.lines.length === 0}
                                    onClick={() => handlePeriodApplyPerLineRemise()}
                                  >
                                    Appliquer
                                  </button>
                                  <button
                                    type="button"
                                    className={draftEditorStyles.btnMuted}
                                    disabled={!hasActivePeriodRemise}
                                    onClick={() => handlePeriodRemoveRemise()}
                                  >
                                    Retirer
                                  </button>
                                </div>
                              </section>
                            )}
                          </div>
                        ) : null}
                        {periodEditSheet === 'addLine' ? (
                          <>
                            <div
                              className={draftEditorStyles.addLineForm}
                              title="Prix unitaire × quantité, ou taux × durée (unité) selon le mode."
                            >
                              <div className={draftEditorStyles.addLineFormHeader}>
                                <h3
                                  className={draftEditorStyles.addLineFormTitle}
                                  id={periodAddLineHeadingId}
                                >
                                  Ligne supplémentaire HT
                                </h3>
                                <div
                                  className={draftEditorStyles.modeSegSm}
                                  role="group"
                                  aria-label="Type de facturation"
                                >
                                  <button
                                    type="button"
                                    className={
                                      addLineMode === EXTRA_LINE_MODE.time
                                        ? draftEditorStyles.modeSegBtnActiveSm
                                        : draftEditorStyles.modeSegBtnSm
                                    }
                                    onClick={() => setAddLineMode(EXTRA_LINE_MODE.time)}
                                  >
                                    Temps
                                  </button>
                                  <button
                                    type="button"
                                    className={
                                      addLineMode === EXTRA_LINE_MODE.quantity
                                        ? draftEditorStyles.modeSegBtnActiveSm
                                        : draftEditorStyles.modeSegBtnSm
                                    }
                                    onClick={() => setAddLineMode(EXTRA_LINE_MODE.quantity)}
                                  >
                                    Qté
                                  </button>
                                </div>
                              </div>
                              <div className={draftEditorStyles.addLineFormFields}>
                                <input
                                  className={draftEditorStyles.inputLibelle}
                                  type="text"
                                  placeholder="Libellé"
                                  value={addLineDesc}
                                  onChange={(e) => setAddLineDesc(e.target.value)}
                                />
                                <div className={draftEditorStyles.addLineFormMainRow}>
                                  <div className={draftEditorStyles.addLineFormDateGroup}>
                                    {addLineServiceDateMonthYearOnly ? (
                                      <InlineMonthYearPicker
                                        inputId={`${periodAddLineHeadingId}-svc-date`}
                                        className={draftEditorStyles.addLineFormDatePickerWrap}
                                        value={
                                          addLineServiceDate.trim().length >= 7
                                            ? addLineServiceDate.trim().slice(0, 7)
                                            : ''
                                        }
                                        onChange={(ym) => {
                                          const next = ym ? `${ym}-01` : '';
                                          addLineServiceDateRef.current = next;
                                          setAddLineServiceDate(next);
                                        }}
                                        ariaLabel="Mois de la prestation (optionnel)"
                                        title="Mois (optionnel)"
                                      />
                                    ) : (
                                      <InlineDatePicker
                                        inputId={`${periodAddLineHeadingId}-svc-date`}
                                        className={draftEditorStyles.addLineFormDatePickerWrap}
                                        value={addLineServiceDate}
                                        onChange={(iso) => {
                                          const next = iso || '';
                                          addLineServiceDateRef.current = next;
                                          setAddLineServiceDate(next);
                                        }}
                                        ariaLabel="Date de la prestation (optionnel)"
                                        title="Date (optionnel)"
                                      />
                                    )}
                                  </div>
                                  {addLineMode === EXTRA_LINE_MODE.time ? (
                                    <div className={draftEditorStyles.addLineFormToolbar}>
                                      <input
                                        className={draftEditorStyles.inCell}
                                        type="text"
                                        inputMode="decimal"
                                        autoComplete="off"
                                        placeholder="Taux"
                                        aria-label={`Taux ${tauxSuffixForUnit(addLineTimeUnit)}`}
                                        value={addLineTaux}
                                        onChange={(e) => setAddLineTaux(e.target.value)}
                                      />
                                      <input
                                        className={draftEditorStyles.inCellSm}
                                        type="text"
                                        inputMode="decimal"
                                        autoComplete="off"
                                        placeholder="1"
                                        aria-label="Durée (valeur)"
                                        value={addLineTimeValue}
                                        onChange={(e) => setAddLineTimeValue(e.target.value)}
                                      />
                                      <select
                                        className={draftEditorStyles.selectSm}
                                        value={addLineTimeUnit}
                                        onChange={(e) => setAddLineTimeUnit(e.target.value)}
                                        aria-label="Unité de temps"
                                      >
                                        {TIME_UNITS.map((o) => (
                                          <option key={o.value} value={o.value}>
                                            {o.label}
                                          </option>
                                        ))}
                                      </select>
                                      {addLinePreview != null && (
                                        <span className={draftEditorStyles.previewChfSm} aria-live="polite">
                                          {formatCurrencyCHF(addLinePreview)}
                                        </span>
                                      )}
                                      <button
                                        type="button"
                                        className={draftEditorStyles.btnIconAdd}
                                        title="Ajouter la ligne"
                                        onClick={() => void handlePeriodAddExtraLine()}
                                      >
                                        <FiPlus size={18} />
                                      </button>
                                    </div>
                                  ) : (
                                    <div className={draftEditorStyles.addLineFormToolbar}>
                                      <input
                                        className={draftEditorStyles.inCell}
                                        type="text"
                                        inputMode="decimal"
                                        autoComplete="off"
                                        placeholder="Prix u. (CHF)"
                                        aria-label="Prix unitaire HT en CHF"
                                        value={addLineUnitPrice}
                                        onChange={(e) => setAddLineUnitPrice(e.target.value)}
                                      />
                                      <input
                                        className={draftEditorStyles.inCellSm}
                                        type="text"
                                        inputMode="decimal"
                                        autoComplete="off"
                                        placeholder="Qté"
                                        aria-label="Quantité"
                                        value={addLineQty}
                                        onChange={(e) => setAddLineQty(e.target.value)}
                                      />
                                      {addLinePreview != null && (
                                        <span className={draftEditorStyles.previewChfSm} aria-live="polite">
                                          {formatCurrencyCHF(addLinePreview)}
                                        </span>
                                      )}
                                      <button
                                        type="button"
                                        className={draftEditorStyles.btnIconAdd}
                                        title="Ajouter la ligne"
                                        onClick={() => void handlePeriodAddExtraLine()}
                                      >
                                        <FiPlus size={18} />
                                      </button>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                            {periodExtraLines.length > 0 ? (
                              <ul className={styles.periodSheetExtraList}>
                                {periodExtraLines.map((ex) => (
                                  <li key={ex.id}>
                                    <span>{ex.description}</span>
                                    {' · '}
                                    <strong>{String(ex.line_total)}</strong>
                                    {' CHF HT '}
                                    <button
                                      type="button"
                                      className={styles.periodSheetExtraRemove}
                                      onClick={() =>
                                        setPeriodExtraLines((prev) => prev.filter((x) => x.id !== ex.id))
                                      }
                                    >
                                      Retirer
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            ) : null}
                          </>
                        ) : null}
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          )}

              </div>

          {!draftInvoiceStub ? (
          <div className={styles.stickyFooter}>
            {footerHint && (
              <p className={styles.footerHint} id="bill-period-footer-hint">
                {footerHint}
              </p>
            )}
            <div
              className={styles.footerGroup}
              aria-describedby={footerHint ? 'bill-period-footer-hint' : undefined}
            >
              <button
                type="button"
                className={styles.btn}
                onClick={toggleLinesPreview}
                disabled={!canPreview() || previewLoading || generateLoading}
                data-testid="invoice-lines-preview-toggle"
              >
                {previewLoading ? (
                  <FiLoader className={styles.btnIconSpin} size={14} aria-hidden />
                ) : (
                  <FiEye size={14} aria-hidden />
                )}
                {previewLoading
                  ? 'Prévisualisation…'
                  : showLinesPreview
                    ? 'Masquer les lignes'
                    : 'Prévisualiser les lignes'}
              </button>
              <button
                type="button"
                className={styles.prepareInvoiceBtn}
                onClick={runGenerate}
                disabled={
                  !canPreview() ||
                  generateLoading ||
                  previewLoading ||
                  (payerType === 'clinic'
                    ? !institutionPlanIsCurrent ||
                      institutionPlanLoading ||
                      !institutionSummary.hasBillable
                    : payerType === 'patient'
                      ? !patientSummary.hasBillable
                      : !partnerSummary.hasBillable)
                }
              >
                {generateLoading ? (
                  <FiLoader className={styles.btnIconSpin} size={14} aria-hidden />
                ) : (
                  <FiFileText size={14} aria-hidden />
                )}
                {generateLoading ? 'Préparation…' : 'Préparer la facture'}
              </button>
            </div>
          </div>
          ) : null}
            </div>
          </>
      </div>
    </div>
      {treatingExcludedRow ? (
        <DisputeResolutionPanel
          companyId={companyId}
          row={treatingExcludedRow}
          onClose={() => {
            const bookingId = treatingExcludedRow.bookingId;
            setTreatingExcludedRow(null);
            window.requestAnimationFrame(() => {
              document
                .querySelector(`[data-testid="dispute-excluded-row-${bookingId}"]`)
                ?.scrollIntoView({ block: 'nearest' });
            });
          }}
          onChanged={() => void fetchInstitutionPlan({ silent: true })}
        />
      ) : null}
    </>,
    portalTarget
  );
};

export default BillPeriodModal;
