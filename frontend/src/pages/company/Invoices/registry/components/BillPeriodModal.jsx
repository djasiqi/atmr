import React, { useState, useEffect, useCallback, useMemo, useRef, useId } from 'react';
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
} from 'react-icons/fi';
import { toast } from 'sonner';
import { invoiceService, formatCurrencyCHF, generateInvoice } from '../../../../../services/invoiceService';
import { getApiErrorMessage } from '../../../../../utils/apiErrorMessage';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import InvoiceLivePreview from './InvoiceLivePreview';
import draftEditorStyles from './InvoiceDraftEditModal.module.css';
import styles from './BillPeriodModal.module.css';
import InlineMonthYearPicker from '../../../../../components/ui/InlineMonthYearPicker';
import InlineDatePicker from '../../../../../components/ui/InlineDatePicker';
import ChipSelect from '../../../../../components/ui/ChipSelect';
import { syncDraftInvoiceWithMergedAssemblyPreview } from '../utils/periodAssemblyInvoiceSync';
import { normalizeServiceDateToIsoForApi } from '../../../../../utils/invoiceServiceDate';

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
    const nextLm = { ...prevLm };
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
  const pr = row.preview_row_id;
  if (pr != null && Number.isFinite(Number(pr))) return Number(pr);
  const bid = row.booking_id;
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
        ? { is_round_trip_leg: true, transport_type: 'A/R' }
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
      line_meta: {
        service_date: row.scheduled_at,
        ...endDateMeta,
        ...arMeta,
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
  /** form = sélection payeur/lignes ; draft = préparation dans la même modale. */
  const [composerPhase, setComposerPhase] = useState('form');
  const [draftInvoiceStub, setDraftInvoiceStub] = useState(null);
  const [payerType, setPayerType] = useState('patient');
  const [periodYear, setPeriodYear] = useState(defaultYear);
  const [periodMonth, setPeriodMonth] = useState(defaultMonth);
  const [clients, setClients] = useState([]);
  const [institutions, setInstitutions] = useState([]);
  const [clientId, setClientId] = useState('');
  const [clinicKey, setClinicKey] = useState(''); // institution id as string
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
      const eligParams = {
        year: periodYear,
        month: periodMonth,
        limit: 500,
        billed_to_type: 'patient',
      };
      /** Sans billed_to_type, l’API eligible agrège tous les types — ne charger les patients qu’en mode direct patient. */
      const eligPromise =
        payerType === 'patient'
          ? invoiceService.fetchEligibleClients(companyId, eligParams)
          : Promise.resolve(null);

      const [elig, inst, bpRaw] = await Promise.all([
        eligPromise,
        invoiceService.fetchInstitutions(companyId),
        invoiceService.fetchBillablePartners(companyId, {
          year: periodYear,
          month: periodMonth,
        }),
      ]);
      const ec =
        payerType === 'patient'
          ? elig?.data?.clients ?? elig?.clients ?? []
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
    setSelectedBookingIds(new Set());
  }, [companyId, open, periodYear, periodMonth, payerType]);

  useEffect(() => {
    if (!open) {
      setPartnershipId('');
      setClientId('');
      setClinicKey('');
      setPreview(null);
      setSelectedBookingIds(new Set());
      setComposerPhase('form');
      setDraftInvoiceStub(null);
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
        const n = `${c.first_name || ''} ${c.last_name || ''}`.trim();
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
    const rows = clients.map((c) => ({
      value: String(c.id),
      label: `${c.first_name} ${c.last_name}${
        c.unbilled_total_amount
          ? ` (${c.unbilled_total_amount} CHF non fact. sur la période, direct patient)`
          : ''
      }`,
    }));
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
      if (periodEditSheet) {
        setPeriodEditSheet(null);
        return;
      }
      onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose, generateLoading, previewLoading, periodEditSheet]);

  const selectedClinic = institutions.find((i) => String(i.id) === clinicKey);
  const clinicCompanyId = selectedClinic?.clinic_company_id ?? null;

  const canPreview = useCallback(() => {
    if (payerType === 'patient') return Boolean(clientId);
    if (payerType === 'clinic') return Boolean(clinicKey && clinicCompanyId);
    if (payerType === 'partner') return Boolean(partnershipId);
    return false;
  }, [payerType, clientId, clinicKey, clinicCompanyId, partnershipId]);

  const footerHint = useMemo(() => {
    if (loadingLists) return null;
    if (payerType === 'patient' && !clientId) {
      return 'Sélectionnez un patient pour prévisualiser.';
    }
    if (payerType === 'clinic') {
      if (!clinicKey) return 'Sélectionnez une clinique pour prévisualiser.';
      if (!clinicCompanyId) {
        return 'Cette institution n’a pas d’entreprise S2 associée — impossible de facturer en mode clinique.';
      }
    }
    if (payerType === 'partner' && !partnershipId) {
      return 'Sélectionnez un partenaire à facturer pour cette période.';
    }
    if (preview && preview.transports_count === 0) {
      return payerType === 'partner'
        ? 'Aucun transfert à facturer sur cette période pour ce partenaire.'
        : 'Aucun transport à facturer sur cette période pour ce payeur.';
    }
    if (canPreview() && !preview && !previewLoading && !generateLoading) {
      return 'Utilisez « Prévisualiser les lignes » pour voir le détail avant de préparer la facture.';
    }
    return null;
  }, [
    loadingLists,
    payerType,
    clientId,
    clinicKey,
    clinicCompanyId,
    preview,
    previewLoading,
    generateLoading,
    partnershipId,
    canPreview,
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
      if (payerType === 'patient') {
        parsedClientId = parseInt(clientId, 10);
        if (!Number.isFinite(parsedClientId) || parsedClientId <= 0) {
          setError('Sélectionnez un patient dans la liste.');
          setPreviewLoading(false);
          return;
        }
        const clientAllowed = clients.some((c) => Number(c.id) === parsedClientId);
        if (!clientAllowed) {
          setError(
            'Ce patient ne correspond pas à la liste chargée pour cette période. Attendez la fin du chargement ou sélectionnez à nouveau un patient.'
          );
          setPreviewLoading(false);
          return;
        }
      }
      if (payerType === 'clinic') {
        const cc = clinicCompanyId;
        if (cc == null || !Number.isFinite(Number(cc))) {
          setError('Sélectionnez une clinique avec identifiant de facturation (S2).');
          setPreviewLoading(false);
          return;
        }
        const clinicAllowed = institutions.some((i) => Number(i.clinic_company_id) === Number(cc));
        if (!clinicAllowed) {
          setError(
            'Cette clinique ne correspond pas à la liste chargée. Sélectionnez-la à nouveau après le chargement.'
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
      });
      setPreview(unwrapApi(res));
    } catch (err) {
      setError(getApiErrorMessage(err, 'Prévisualisation impossible'));
    } finally {
      setPreviewLoading(false);
    }
  };

  const runGenerate = async () => {
    if (!canPreview()) {
      setError('Sélectionnez un payeur.');
      return;
    }
    if (!preview) {
      setError('Prévisualisez d’abord l’aperçu.');
      return;
    }
    if (preview.transports_count === 0) {
      setError(
        payerType === 'partner'
          ? 'Aucun transfert à facturer sur cette période. Vérifiez le partenaire, le mois, ou des transferts déjà facturés.'
          : 'Aucun transport à facturer sur cette période. Vérifiez le payeur, le mois, ou des courses déjà facturées.'
      );
      return;
    }
    setError('');
    setGenerateLoading(true);
    try {
      if (payerType === 'partner') {
        const result = await invoiceService.generatePartnerInvoice(companyId, {
          partnership_id: parseInt(partnershipId, 10),
          period_year: periodYear,
          period_month: periodMonth,
        });
        const inv = result?.data ?? result;
        if (inv?.id) {
          setDraftInvoiceStub(inv);
          setComposerPhase('draft');
          onInvoiceGenerated?.(inv);
        } else {
          setError('Réponse inattendue du serveur.');
        }
        return;
      }

      if (payerType === 'patient') {
        const cid = parseInt(clientId, 10);
        if (!clients.some((c) => Number(c.id) === cid)) {
          setError('Patient invalide pour cette période — sélectionnez-le à nouveau dans la liste.');
          setGenerateLoading(false);
          return;
        }
      } else if (payerType === 'clinic') {
        if (
          clinicCompanyId == null ||
          !institutions.some((i) => Number(i.clinic_company_id) === Number(clinicCompanyId))
        ) {
          setError('Clinique invalide pour cette période — sélectionnez-la à nouveau dans la liste.');
          setGenerateLoading(false);
          return;
        }
      }

      let payload;
      if (payerType === 'patient') {
        const ids = [...selectedBookingIds];
        if (hasAssemblyLines && ids.length === 0) {
          setError('Cochez au moins une ligne valide pour préparer la facture.');
          setGenerateLoading(false);
          return;
        }
        payload = {
          client_id: parseInt(clientId, 10),
          period_year: periodYear,
          period_month: periodMonth,
          ...(hasAssemblyLines ? { reservation_ids: ids } : {}),
        };
      } else {
        const ids = [...selectedBookingIds];
        if (hasAssemblyLines && ids.length === 0) {
          setError('Cochez au moins une ligne valide pour préparer la facture.');
          setGenerateLoading(false);
          return;
        }
        payload = {
          mode: 'clinic_monthly',
          clinic_company_id: clinicCompanyId,
          period_year: periodYear,
          period_month: periodMonth,
          ...(hasAssemblyLines ? { reservation_ids: ids } : {}),
        };
      }
      const result = await generateInvoice(companyId, payload);
      let inv = result?.data ?? result;
      if (inv?.data?.id != null) {
        inv = inv.data;
      }
      if (inv?.id) {
        if (
          hasAssemblyLines &&
          mergedPeriodInvoice &&
          (payerType === 'patient' || payerType === 'clinic')
        ) {
          try {
            inv = await syncDraftInvoiceWithMergedAssemblyPreview(
              companyId,
              inv.id,
              mergedPeriodInvoice
            );
            // Le PDF initial est généré avant cette synchro : sans régénération il reste obsolète
            // (une seule ligne, montants catalogue, pas les CUSTOM ajoutés à l’aperçu).
            try {
              const pdfRes = await invoiceService.regenerateInvoicePdf(companyId, inv.id);
              const pdfUrl =
                (pdfRes && typeof pdfRes === 'object' && pdfRes.pdf_url) ||
                (pdfRes?.data && typeof pdfRes.data === 'object' && pdfRes.data.pdf_url);
              if (typeof pdfUrl === 'string' && pdfUrl.trim()) {
                inv = { ...inv, pdf_url: pdfUrl.trim() };
              }
            } catch (pdfErr) {
              console.error(pdfErr);
              toast.warning(
                getApiErrorMessage(
                  pdfErr,
                  'Le PDF n’a pas pu être régénéré automatiquement. Ouvrez le brouillon et utilisez « Régénérer le PDF » si besoin.'
                ),
                { duration: 9000 }
              );
            }
          } catch (syncErr) {
            console.error(syncErr);
            setError(
              getApiErrorMessage(
                syncErr,
                'Facture créée, mais les changements de l’aperçu (lignes, remises, prestations) n’ont pas tous été appliqués. Vérifiez le brouillon.'
              )
            );
          }
        }
        setDraftInvoiceStub(inv);
        setComposerPhase('draft');
        onInvoiceGenerated?.(inv);
      } else {
        setError('Réponse inattendue du serveur.');
      }
    } catch (err) {
      setError(getApiErrorMessage(err, 'Échec de génération'));
    } finally {
      setGenerateLoading(false);
    }
  };

  const assemblyNoSelectable =
    hasAssemblyLines &&
    preview &&
    Array.isArray(preview?.preview_lines) &&
    preview.preview_lines.length > 0 &&
    preview.preview_lines.every((l) => l.is_locked);

  if (!open) return null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.panel} onClick={(e) => e.stopPropagation()}>
        {composerPhase === 'draft' && draftInvoiceStub ? (
          <>
            <div className={styles.head}>
              <div className={styles.headText}>
                <h2 className={styles.title}>Préparer la facture</h2>
                <p className={styles.subtitle}>Ajuster les lignes, puis PDF ou envoi</p>
              </div>
              <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
                <FiX size={18} />
              </button>
            </div>
            <DraftInvoiceEditorPanel
              key={draftInvoiceStub?.id ?? 'draft-stub'}
              open
              initialInvoice={draftInvoiceStub}
              companyId={companyId}
              onUpdated={() => onInvoiceGenerated?.(draftInvoiceStub)}
              onOpenSendEmail={onOpenSendEmail}
              onMarkAsSent={onMarkAsSent}
            />
          </>
        ) : (
          <>
            <div className={styles.head}>
              <div className={styles.headText}>
                <h2 className={styles.title}>Nouvelle facture</h2>
              </div>
              <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
                <FiX size={18} />
              </button>
            </div>

            <div className={styles.formColumn}>
              <div className={styles.formScroll}>
          <div className={styles.section}>
            <div className={styles.fieldGroup}>
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
                  title="Facturation mensuelle clinique (S2)"
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
                  <span className={styles.payerChoiceTitle}>Clinique</span>
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
                  <span className={styles.payerChoiceTitle}>Partenaires</span>
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
                    ? 'Période facturée et clinique'
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
                    Clinique
                  </label>
                  <ChipSelect
                    id="bill-period-clinic"
                    aria-labelledby="bill-clinic-inline-label"
                    className={styles.billChipSelectGrow}
                    options={billClinicChipOptions}
                    value={clinicKey}
                    placeholder="— Choisir une clinique —"
                    onChange={(v) => {
                      setClinicKey(v == null || v === '' ? '' : String(v));
                      setPreview(null);
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
                    }}
                    disabled={loadingLists}
                    menuMinWidth={300}
                    filterable
                  />
                </div>
              )}
            </div>

            {loadingLists && (
              <p className={styles.formHint} role="status">
                Chargement des listes (patients, institutions, partenaires)…
              </p>
            )}
          </div>

          {error && <div className={styles.err}>{error}</div>}

          {preview && !error && !hasAssemblyLines && (
            <div className={styles.previewBox}>
              <div className={styles.previewHead}>
                <h3>Aperçu</h3>
                <span className={styles.previewModeBadge}>
                  {preview.mode === 'clinic_monthly'
                    ? 'S2 clinique'
                    : preview.mode === 'partner_monthly'
                      ? 'Facturation partenaire'
                      : 'Direct patient'}
                </span>
              </div>
              <div className={styles.previewStats}>
                <div className={styles.previewStat}>
                  <span className={styles.previewStatValue}>{preview.transports_count ?? 0}</span>
                  <span className={styles.previewStatLabel}>
                    {preview.mode === 'partner_monthly'
                      ? `transfert${preview.transports_count !== 1 ? 's' : ''} validé${preview.transports_count !== 1 ? 's' : ''}`
                      : `transport${preview.transports_count !== 1 ? 's' : ''} éligible${preview.transports_count !== 1 ? 's' : ''}`}
                  </span>
                </div>
                <div className={styles.previewStatHighlight}>
                  <span className={styles.previewStatLabel}>Total estimé</span>
                  <span className={styles.previewStatMoney}>
                    {formatCurrencyCHF(preview.estimated_total ?? 0).replace(' CHF', '')}{' '}
                    <span className={styles.previewStatCurrency}>CHF</span>
                  </span>
                </div>
              </div>
              {Array.isArray(preview.warnings) && preview.warnings.length > 0 && (
                <ul className={styles.warnings}>
                  {preview.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {payerType === 'partner' && preview && !hasAssemblyLines && (
            <p className={styles.partnerHint} role="note">
              Aucun détail de ligne n&apos;a pu être chargé pour ce partenaire. Vérifiez les transferts validés sur
              la période.
            </p>
          )}

          {hasAssemblyLines && preview && (
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
                      title="Recharger la facture depuis le serveur"
                      aria-label="Actualiser les données depuis le serveur"
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
                              <div
                                className={`${draftEditorStyles.tableScroll} ${draftEditorStyles.tableScrollDense}${
                                  syntheticPeriodInvoice.lines.length >= 12
                                    ? ` ${draftEditorStyles.tableScrollHeavy}`
                                    : ''
                                }`}
                              >
                                {syntheticPeriodInvoice.lines.length > 0 ? (
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
                                    {syntheticPeriodInvoice.lines.length === 0 ? (
                                      <tr>
                                        <td colSpan={4} className={draftEditorStyles.tableEmptyTight}>
                                          Aucune ligne.
                                        </td>
                                      </tr>
                                    ) : (
                                      syntheticPeriodInvoice.lines.map((ln) => {
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
                                        return (
                                          <tr key={ln.id} title={rowTitle}>
                                            <td className={draftEditorStyles.colDescCell}>
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
                                                <button
                                                  type="button"
                                                  className={`${draftEditorStyles.btnTrashXs} ${draftEditorStyles.danger}`}
                                                  disabled={!canExclude}
                                                  title="Retirer ce transport de la facture (aperçu)"
                                                  aria-label={`Exclure la ligne ${ln.id}`}
                                                  onClick={() => excludePeriodLineFromPreview(ln.id)}
                                                >
                                                  <FiTrash2 size={13} aria-hidden />
                                                </button>
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

          {hasAssemblyLines && preview && Array.isArray(preview.warnings) && preview.warnings.length > 0 && (
            <ul className={styles.warnings}>
              {preview.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}

          {assemblyNoSelectable && (
            <p className={styles.emptyHint} role="status">
              Aucune course facturable à sélectionner pour cette période (courses déjà facturées ou verrouillées).
            </p>
          )}

              </div>

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
                onClick={runPreview}
                disabled={!canPreview() || previewLoading || generateLoading}
              >
                {previewLoading ? (
                  <FiLoader className={styles.btnIconSpin} size={14} aria-hidden />
                ) : (
                  <FiEye size={14} aria-hidden />
                )}
                {previewLoading ? 'Prévisualisation…' : 'Prévisualiser les lignes'}
              </button>
              <button
                type="button"
                className={styles.prepareInvoiceBtn}
                onClick={runGenerate}
                disabled={
                  !canPreview() ||
                  !preview ||
                  generateLoading ||
                  previewLoading ||
                  preview.transports_count === 0 ||
                  (hasAssemblyLines && selectedBookingIds.size === 0) ||
                  assemblyNoSelectable
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
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default BillPeriodModal;
