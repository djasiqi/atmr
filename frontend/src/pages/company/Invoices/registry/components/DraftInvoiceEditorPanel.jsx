import React, {
  useState,
  useEffect,
  useLayoutEffect,
  useCallback,
  useMemo,
  useRef,
  useId,
} from 'react';
import {
  FiRefreshCw,
  FiSend,
  FiTrash2,
  FiCheck,
  FiPlus,
  FiChevronLeft,
  FiChevronRight,
  FiSearch,
  FiPrinter,
  FiPercent,
  FiExternalLink,
  FiChevronsDown,
  FiChevronsUp,
  FiMaximize2,
  FiMinimize2,
  FiDownload,
  FiList,
} from 'react-icons/fi';
import {
  getInvoice,
  invoiceService,
  formatCurrencyCHF,
} from '../../../../../services/invoiceService';
import { printPdfBytes, preloadInvoicePdfPrint } from '../../../../../utils/invoicePdfPrint';
import {
  appendPdfEmbedChromiumViewerFragment,
  buildInvoicePdfApiUrl,
} from '../../../../../utils/pdfUrlFallback';
import {
  downloadProtectedPdfAsFile,
  fetchProtectedPdfBytes,
  fetchProtectedPdfObjectUrl,
  openProtectedPdfInNewTab,
} from '../../../../../utils/protectedPdf';
import { buildInvoicePdfDownloadFilename } from '../../../../../utils/invoicePdfFilename';
import { getApiErrorMessage } from '../../../../../utils/apiErrorMessage';
import { normalizeServiceDateToIsoForApi } from '../../../../../utils/invoiceServiceDate';
import { filterInvoiceLines } from '../../../../../utils/invoiceLineFilter';
import {
  isAnyRoundTripLine,
  isRoundTripPreviewHiddenLine,
  canShowRoundTripLegExcludeActions,
  sortInvoiceLinesForEditor,
} from '../../../../../utils/invoiceLineRoundTrip';
import '../../../../../styles/acrobatPdfEmbedHide.css';
import styles from './InvoiceDraftEditModal.module.css';
import InvoiceLivePreview from './InvoiceLivePreview';
import InvoiceLineEditorContext from './InvoiceLineEditorContext';
import InlineDatePicker from '../../../../../components/ui/InlineDatePicker';
import InlineMonthYearPicker from '../../../../../components/ui/InlineMonthYearPicker';

/** Classe sur body pour masquer l’UI injectée par l’extension Adobe Acrobat sur les PDF embarqués. */
const HIDE_ACROBAT_PDF_OVERLAY_BODY_CLASS = 'atmr-hide-acrobat-pdf-embed-overlay';

/** Force remount des champs defaultValue quand le serveur met à jour montants/libellés. */
const lineKey = (l) => `${l.id}-${String(l.line_total ?? '')}-${String(l.description ?? '').slice(0, 48)}`;

const EXTRA_LINE_MODE = {
  time: 'time',
  quantity: 'quantity',
};

/** Remise % : soit un taux unique sur tous les transports, soit un taux par ligne (exclusif). */
const REMISE_PCT_MODE = {
  global: 'global',
  perLine: 'perLine',
};

const TIME_UNITS = [
  { value: 'min', label: 'min' },
  { value: 'h', label: 'h' },
  { value: 'd', label: 'j' },
  { value: 'mois', label: 'mois' },
];

/** Au-delà, pagination + filtre (factures avec de nombreux transports). */
const LINE_PAGE_SIZE = 25;
const HEAVY_LINES_THRESHOLD = 12;

/** Méta facture : parfois objet, parfois chaîne JSON selon la couche API / cache. */
function parseInvoiceMeta(raw) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    try {
      const p = JSON.parse(raw);
      return typeof p === 'object' && p !== null ? p : null;
    } catch {
      return null;
    }
  }
  if (typeof raw === 'object') return raw;
  return null;
}

/**
 * True si meta.pdf indique un PDF absent / périmé / en échec (hint UI).
 * Pour les factures éditables, download/open/print forcent toujours une regen.
 */
function invoiceDraftPdfNeedsRefresh(inv) {
  if (!String(inv?.pdf_url || '').trim()) return true;
  const st = parseInvoiceMeta(inv?.meta)?.pdf?.status;
  return st === 'stale' || st === 'failed';
}

/** Extrait la facture depuis les réponses API ({ data: { invoice } }, { data: facture }, etc.). */
function unwrapInvoicePayload(payload) {
  if (!payload || typeof payload !== 'object') return null;
  if (payload.invoice && typeof payload.invoice === 'object' && payload.invoice.id != null) {
    return payload.invoice;
  }
  const inner = payload.data;
  if (inner && typeof inner === 'object') {
    if (inner.invoice && typeof inner.invoice === 'object' && inner.invoice.id != null) {
      return inner.invoice;
    }
    if (inner.id != null && Array.isArray(inner.lines)) {
      return inner;
    }
  }
  if (payload.id != null && Array.isArray(payload.lines)) {
    return payload;
  }
  return null;
}

/** Pourcentage remise globale (méta API : nombre, chaîne «10», ou déduit des montants HT). */
function normalizeGlobalDiscountPercent(gd) {
  if (!gd || typeof gd !== 'object') return null;
  const raw = gd.percent;
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
  if (raw != null && raw !== '') {
    const n = parseFloat(String(raw).replace(',', '.'));
    if (Number.isFinite(n)) return n;
  }
  const gross = Number(gd.subtotal_before_ht);
  const disc = Number(gd.amount_ht);
  if (Number.isFinite(gross) && gross > 0 && Number.isFinite(disc) && disc >= 0) {
    return Math.round((disc / gross) * 10000) / 100;
  }
  return null;
}

/**
 * Déduit un % affichable si les lignes portent encore `original_line_total` (remise globale brouillon).
 */
function inferGlobalDiscountPercentFromOriginalLineMeta(lines) {
  if (!Array.isArray(lines)) return null;
  let origSum = 0;
  let curSum = 0;
  for (const l of lines) {
    const m = l?.line_meta;
    if (!m || typeof m !== 'object') continue;
    const o = m.original_line_total;
    if (o == null || o === '') continue;
    const orig = parseFloat(String(o).replace(',', '.'));
    const cur = Number(l.line_total);
    if (!Number.isFinite(orig) || orig <= 0 || !Number.isFinite(cur)) continue;
    origSum += orig;
    curSum += cur;
  }
  if (origSum <= 0 || curSum > origSum + 0.05) return null;
  const pct = Math.round(((origSum - curSum) / origSum) * 10000) / 100;
  if (!Number.isFinite(pct) || pct <= 0 || pct > 100) return null;
  return pct;
}

/** Remise globale % encore présente en méta (bouton Retirer, champs, badge). */
function invoiceHasActiveGlobalPercentDiscountFromMeta(meta) {
  const m = parseInvoiceMeta(meta);
  const gd = m?.global_discount;
  if (!gd || typeof gd !== 'object') return false;
  if (normalizeGlobalDiscountPercent(gd) != null) return true;
  const gross = Number(gd.subtotal_before_ht);
  const disc = Number(gd.amount_ht);
  if (Number.isFinite(gross) && gross > 0 && Number.isFinite(disc) && disc >= 0) return true;
  if (Array.isArray(gd.line_snapshots) && gd.line_snapshots.length > 0) return true;
  if (gd.snapshot_version != null) return true;
  if (Array.isArray(gd.ride_line_ids) && gd.ride_line_ids.length > 0) return true;
  return false;
}

function hasGlobalDiscountEvidenceFromLines(lines) {
  return (
    inferGlobalDiscountPercentFromOriginalLineMeta(lines) != null ||
    inferPercentFromRemiseCommercialeLine(lines) != null
  );
}

/** Brouillon créé avec remise globale à la génération : ligne CUSTOM négative « Remise commerciale X % ». */
function inferPercentFromRemiseCommercialeLine(lines) {
  if (!Array.isArray(lines)) return null;
  for (const l of lines) {
    if (String(l?.type || '').toLowerCase() !== 'custom') continue;
    const lt = Number(l?.line_total);
    if (!Number.isFinite(lt) || lt >= 0) continue;
    const desc = String(l?.description || '');
    const m = desc.match(/Remise commerciale\s+([\d.,]+)\s*%/i);
    if (m) {
      const p = parseFloat(String(m[1]).replace(',', '.'));
      if (Number.isFinite(p) && p > 0 && p <= 100) return p;
    }
  }
  return null;
}

function tauxSuffixForUnit(u) {
  if (u === 'min') return 'CHF/min';
  if (u === 'h') return 'CHF/h';
  if (u === 'd') return 'CHF/j';
  if (u === 'mois') return 'CHF/mois';
  return 'CHF';
}

/** Corps éditable facture (brouillon) — réutilisable dans InvoiceDraftEditModal ou flux compositeur. */
const DraftInvoiceEditorPanel = ({
  open,
  initialInvoice,
  companyId,
  onUpdated,
  onOpenSendEmail,
  onMarkAsSent,
}) => {
  const [inv, setInv] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [globalPct, setGlobalPct] = useState('');
  const [globalNote, setGlobalNote] = useState('');
  const [localAmounts, setLocalAmounts] = useState({});
  const [localNotes, setLocalNotes] = useState({});
  const [customLineDesc, setCustomLineDesc] = useState('');
  const [customLineMode, setCustomLineMode] = useState(() => EXTRA_LINE_MODE.time);
  /** Temps : prix par unité (min, h, j, mois) */
  const [customLineTaux, setCustomLineTaux] = useState('');
  const [customLineTimeValue, setCustomLineTimeValue] = useState('1');
  const [customLineTimeUnit, setCustomLineTimeUnit] = useState('h');
  /** Quantité : montant unitaire HT × quantité */
  const [customLineUnitPrice, setCustomLineUnitPrice] = useState('');
  const [customLineQty, setCustomLineQty] = useState('1');
  /** Optionnel : colonne « Date » (aperçu / PDF), format API YYYY-MM-DD. */
  const [customLineServiceDate, setCustomLineServiceDate] = useState('');
  const [localDescriptions, setLocalDescriptions] = useState({});
  const [lineFilter, setLineFilter] = useState('');
  const [linePage, setLinePage] = useState(1);
  const [pdfNonce, setPdfNonce] = useState(0);
  const [showRemisePanel, setShowRemisePanel] = useState(false);
  const [remisePercentMode, setRemisePercentMode] = useState(() => REMISE_PCT_MODE.global);
  const [perLinePercents, setPerLinePercents] = useState({});
  const [freeRemiseDesc, setFreeRemiseDesc] = useState('');
  const [freeRemiseAmount, setFreeRemiseAmount] = useState('');
  const [showAddLinePanel, setShowAddLinePanel] = useState(false);
  const [showLinesSheet, setShowLinesSheet] = useState(false);
  const [pdfZoneExpanded, setPdfZoneExpanded] = useState(false);
  const [isPdfFullscreen, setIsPdfFullscreen] = useState(false);
  /** Blob URL (API JWT SEC-06) pour l’iframe PDF quand l’aperçu HTML n’est pas utilisé. */
  const [protectedPdfBlobUrl, setProtectedPdfBlobUrl] = useState('');
  /** Aligné sur CompanyBillingSettings.vat_applicable — contrôle colonnes TVA/TTC dans l’aperçu HTML. */
  const [companyVatApplicable, setCompanyVatApplicable] = useState(true);
  const pdfWrapRef = useRef(null);
  const pdfIframeRef = useRef(null);
  const protectedPdfBlobUrlRef = useRef('');
  /** Cache octets PDF pour réimpression / prefetch (évite un 2e GET API). */
  const printPdfBytesCacheRef = useRef({ key: '', bytes: null });
  const mountedRef = useRef(true);
  /** Clé « société:facture » déjà chargée — empêche un GET répété si les callbacks parent changent. */
  const initialLoadKeyRef = useRef('');
  /** Dernière date « ligne suppl. » connue (évite perte si blur/changement d’état pas encore rejoué). */
  const customLineServiceDateRef = useRef('');
  const addLineHeadingId = useId();
  const remiseHeadingId = useId();
  const linesSheetHeadingId = useId();

  /** Unité « mois » : la date de prestation est mois+année uniquement (stockage premier jour du mois). */
  const serviceDateMonthYearOnly =
    customLineMode === EXTRA_LINE_MODE.time && customLineTimeUnit === 'mois';

  useEffect(() => {
    if (!serviceDateMonthYearOnly) return;
    setCustomLineServiceDate((prev) => {
      if (!prev || String(prev).trim().length < 7) return prev;
      const head = String(prev).trim().slice(0, 7);
      return `${head}-01`;
    });
  }, [serviceDateMonthYearOnly]);

  useEffect(() => {
    customLineServiceDateRef.current = customLineServiceDate;
  }, [customLineServiceDate]);

  /**
   * Notification parent via ref : un `onUpdated` recréé à chaque rendu du parent ne doit pas
   * changer l’identité de `load`, sinon chaque GET relance un GET (rechargement en boucle).
   */
  const onUpdatedRef = useRef(onUpdated);
  useEffect(() => {
    onUpdatedRef.current = onUpdated;
  }, [onUpdated]);
  const notifyUpdated = useCallback(() => {
    onUpdatedRef.current?.();
  }, []);

  /** Verrou optimiste (optionnel côté API si absent). */
  const draftConcurrencyPayload = useMemo(() => {
    const ts = inv?.updated_at;
    if (ts == null || String(ts).trim() === '') return {};
    return { expected_updated_at: ts };
  }, [inv?.updated_at]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!open || !companyId) return;
    let cancelled = false;
    (async () => {
      try {
        const raw = await invoiceService.fetchBillingSettings(companyId);
        if (cancelled || !raw) return;
        const settings =
          raw && typeof raw === 'object' && raw.data && typeof raw.data === 'object'
            ? raw.data
            : raw;
        setCompanyVatApplicable(settings?.vat_applicable !== false);
      } catch {
        if (!cancelled) setCompanyVatApplicable(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, companyId]);

  const applyInvoiceData = useCallback((data) => {
    if (!data || typeof data !== 'object' || data.id == null) return;
    setInv(data);
    setPdfNonce((n) => n + 1);
    const meta = parseInvoiceMeta(data.meta);
    const gd = meta?.global_discount;
    const pld = meta?.per_line_discounts;
    const inferredFromLines =
      inferGlobalDiscountPercentFromOriginalLineMeta(data.lines) ??
      inferPercentFromRemiseCommercialeLine(data.lines);
    if (invoiceHasActiveGlobalPercentDiscountFromMeta(meta) || inferredFromLines != null) {
      setRemisePercentMode(REMISE_PCT_MODE.global);
      let pct = gd ? normalizeGlobalDiscountPercent(gd) : null;
      if (pct == null && inferredFromLines != null) pct = inferredFromLines;
      setGlobalPct(pct != null ? String(pct) : '');
      setGlobalNote(gd?.note ? String(gd.note) : '');
    } else if (pld?.lines?.length) {
      setRemisePercentMode(REMISE_PCT_MODE.perLine);
      setGlobalPct('');
      setGlobalNote('');
    } else {
      setGlobalPct('');
      setGlobalNote('');
      setRemisePercentMode(REMISE_PCT_MODE.global);
    }
  }, []);

  const applyInvoiceFromPayload = useCallback(
    (payload) => {
      const data = unwrapInvoicePayload(payload);
      if (data) {
        applyInvoiceData(data);
        notifyUpdated();
        return data;
      }
      return null;
    },
    [applyInvoiceData, notifyUpdated]
  );

  /**
   * GET détail (cacheBust) — JSON facture + lignes à jour. Ne régénère pas le PDF côté serveur pour un brouillon.
   * Deps : `inv?.id` / `initialInvoice?.id` seulement — pas l’objet `inv` (nouvelle référence à chaque GET → boucle load).
   * @param {{notify?: boolean}} [options] `notify: false` pour une simple lecture : rien n’a changé côté serveur,
   *   la liste des factures du parent n’a donc pas besoin d’être rechargée.
   * @throws {Error} INVALID_INVOICE_PAYLOAD | erreurs réseau/API
   */
  const reloadPdfPreviewFromServer = useCallback(
    async ({ notify = true } = {}) => {
      const id = inv?.id ?? initialInvoice?.id;
      if (!companyId || !id) {
        throw new Error('MISSING_INVOICE_CONTEXT');
      }
      const res = await getInvoice(companyId, id, { cacheBust: true });
      const data = unwrapInvoicePayload(res) ?? res?.data ?? res;
      if (!data || typeof data !== 'object' || data.id == null) {
        throw new Error('INVALID_INVOICE_PAYLOAD');
      }
      applyInvoiceData(data);
      if (notify) notifyUpdated();
      return data;
    },
    [companyId, inv?.id, initialInvoice?.id, applyInvoiceData, notifyUpdated]
  );

  /** Après mutation brouillon : GET détail pour JSON à jour (totaux, lignes). */
  const syncInvoiceAfterDraftMutation = useCallback(async () => {
    if (!companyId || !inv?.id) return;
    try {
      printPdfBytesCacheRef.current = { key: '', bytes: null };
      const res = await getInvoice(companyId, inv.id, { cacheBust: true });
      const data = unwrapInvoicePayload(res) ?? res?.data ?? res;
      if (data && typeof data === 'object' && data.id != null) {
        applyInvoiceData(data);
        notifyUpdated();
      }
    } catch {
      setError('Impossible de recharger la facture.');
    }
  }, [companyId, inv?.id, applyInvoiceData, notifyUpdated]);

  /** Réponse mutation contient déjà ``invoice`` → pas de GET redondant. */
  const afterDraftMutation = useCallback(
    async (payload) => {
      printPdfBytesCacheRef.current = { key: '', bytes: null };
      if (applyInvoiceFromPayload(payload)) return;
      await syncInvoiceAfterDraftMutation();
    },
    [applyInvoiceFromPayload, syncInvoiceAfterDraftMutation]
  );

  const load = useCallback(async () => {
    if (!open || !companyId || !initialInvoice?.id) return;
    setLoading(true);
    setError('');
    try {
      await reloadPdfPreviewFromServer({ notify: false });
    } catch (e) {
      if (e?.message === 'INVALID_INVOICE_PAYLOAD') {
        setError('Réponse facture invalide.');
      } else {
        setError('Impossible de charger la facture.');
      }
    } finally {
      setLoading(false);
    }
  }, [open, companyId, initialInvoice?.id, reloadPdfPreviewFromServer]);

  /** Hydratation immédiate depuis la liste (avant peinture) pour éviter un écran « Chargement… » inutile. */
  useLayoutEffect(() => {
    if (!open || initialInvoice?.id == null) return;
    setInv(initialInvoice);
    // Dépendances volontairement limitées à open + id : sinon une nouvelle référence parent à chaque rendu réécrase inv après le GET.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, initialInvoice?.id]);

  /** Un seul GET initial par (société, facture) : le bouton « Recharger » reste la relance manuelle. */
  useEffect(() => {
    if (!open || !companyId || initialInvoice?.id == null) {
      initialLoadKeyRef.current = '';
      return;
    }
    const key = `${companyId}:${initialInvoice.id}`;
    if (initialLoadKeyRef.current === key) return;
    initialLoadKeyRef.current = key;
    void load();
  }, [open, companyId, initialInvoice?.id, load]);

  useEffect(() => {
    if (!open) {
      setInv(null);
      setLocalAmounts({});
      setLocalNotes({});
      setLocalDescriptions({});
      setGlobalPct('');
      setGlobalNote('');
      setCustomLineDesc('');
      setCustomLineMode(EXTRA_LINE_MODE.time);
      setCustomLineTaux('');
      setCustomLineTimeValue('1');
      setCustomLineTimeUnit('h');
      setCustomLineUnitPrice('');
      setCustomLineQty('1');
      customLineServiceDateRef.current = '';
      setCustomLineServiceDate('');
      setLineFilter('');
      setLinePage(1);
      setShowRemisePanel(false);
      setFreeRemiseDesc('');
      setFreeRemiseAmount('');
      setRemisePercentMode(REMISE_PCT_MODE.global);
      setPerLinePercents({});
      setShowAddLinePanel(false);
      setPdfZoneExpanded(false);
    }
  }, [open]);

  const customLinePreview = useMemo(() => {
    if (customLineMode === EXTRA_LINE_MODE.quantity) {
      const u = parseFloat(String(customLineUnitPrice).replace(',', '.'));
      const q = parseFloat(String(customLineQty).replace(',', '.'));
      if (!Number.isFinite(u) || u <= 0 || !Number.isFinite(q) || q <= 0) return null;
      return u * q;
    }
    const t = parseFloat(String(customLineTaux).replace(',', '.'));
    const v = parseFloat(String(customLineTimeValue).replace(',', '.'));
    if (!Number.isFinite(t) || t <= 0 || !Number.isFinite(v) || v <= 0) return null;
    return t * v;
  }, [
    customLineMode,
    customLineUnitPrice,
    customLineQty,
    customLineTaux,
    customLineTimeValue,
  ]);

  /** TTC affiché : somme des lignes (total_with_vat), pour rester aligné avec le tableau (incl. lignes temps/Qté). */
  const draftTotalTtc = useMemo(() => {
    const list = inv?.lines;
    if (!list?.length) return Number(inv?.total_amount ?? 0);
    let sum = 0;
    for (const line of list) {
      const tw = Number(line.total_with_vat);
      const ht = Number(line.line_total);
      if (Number.isFinite(tw)) sum += tw;
      else if (Number.isFinite(ht)) sum += ht;
    }
    return Math.round(sum * 100) / 100;
  }, [inv]);

  /**
   * Déduction fixe HT (ligne CUSTOM négative) : pas de TVA sur la ligne (backend) → l’impact TTC
   * est égal à −montant HT saisi. Aperçu avant clic « Ajouter ».
   */
  const freeRemiseImpactPreview = useMemo(() => {
    const raw = String(freeRemiseAmount || '').replace(',', '.').trim();
    if (!raw) return null;
    const amt = parseFloat(raw);
    if (!Number.isFinite(amt) || amt <= 0) return null;
    const newTtc = Math.round((draftTotalTtc - amt) * 100) / 100;
    return { htDeduction: amt, newTtc };
  }, [freeRemiseAmount, draftTotalTtc]);

  const lines = useMemo(
    () => sortInvoiceLinesForEditor(Array.isArray(inv?.lines) ? inv.lines : []),
    [inv?.lines]
  );

  const invoiceStatusLowerResolved = useMemo(
    () => String(inv?.status || initialInvoice?.status || '').toLowerCase(),
    [inv?.status, initialInvoice?.status]
  );

  const isDraft = invoiceStatusLowerResolved === 'draft';

  /** Même barre d’édition que le brouillon : envoyée / partielle / en retard (pas payée / annulée). */
  const allowsLineEditing = useMemo(
    () =>
      ['draft', 'sent', 'partially_paid', 'overdue'].includes(invoiceStatusLowerResolved),
    [invoiceStatusLowerResolved]
  );

  /** Aperçu HTML aligné PDF (A/R, montants fusionnés) : émise éditable ou payée en lecture seule. */
  const showHtmlInvoicePreview = useMemo(() => {
    if (!inv) return false;
    if (allowsLineEditing) return true;
    return (
      invoiceStatusLowerResolved === 'paid' &&
      Array.isArray(inv.lines) &&
      inv.lines.length > 0
    );
  }, [inv, allowsLineEditing, invoiceStatusLowerResolved]);

  const filteredLines = useMemo(
    () => filterInvoiceLines(lines, lineFilter),
    [lines, lineFilter]
  );

  const totalLinePages = Math.max(1, Math.ceil(filteredLines.length / LINE_PAGE_SIZE));
  const effectiveLinePage = Math.min(Math.max(1, linePage), totalLinePages);

  const paginatedLines = useMemo(() => {
    const start = (effectiveLinePage - 1) * LINE_PAGE_SIZE;
    return filteredLines.slice(start, start + LINE_PAGE_SIZE);
  }, [filteredLines, effectiveLinePage]);

  useEffect(() => {
    setLinePage((p) => Math.min(p, totalLinePages));
  }, [totalLinePages]);

  useEffect(() => {
    setLinePage(1);
  }, [lineFilter]);

  useEffect(() => {
    setLinePage(1);
    setLineFilter('');
  }, [inv?.id]);

  const perLineDiscountableLines = useMemo(() => {
    const list = Array.isArray(inv?.lines) ? inv.lines : [];
    return list.filter((l) => {
      const ht = Number(l?.line_total);
      if (!Number.isFinite(ht) || ht <= 0) return false;
      const t = String(l?.type || '').toLowerCase();
      if (t === 'ride') return true;
      if (t === 'custom') {
        const m = l?.line_meta || {};
        return !m.manual_discount && !m.global_discount_line && !m.per_line_discount_line;
      }
      return false;
    });
  }, [inv?.lines]);

  /** Au moins une ligne HT positive remisable (transport ou prestation CUSTOM), aligné backend. */
  const hasRemisablePositiveLines = useMemo(() => {
    const list = Array.isArray(inv?.lines) ? inv.lines : [];
    return list.some((l) => {
      const ht = Number(l?.line_total);
      if (!Number.isFinite(ht) || ht <= 0) return false;
      const t = String(l?.type || '').toLowerCase();
      if (t === 'ride') return true;
      if (t === 'custom') {
        const m = l?.line_meta || {};
        if (m.manual_discount || m.global_discount_line || m.per_line_discount_line) return false;
        return true;
      }
      return false;
    });
  }, [inv?.lines]);

  const hasActivePercentRemise = useMemo(() => {
    if (invoiceHasActiveGlobalPercentDiscountFromMeta(inv?.meta)) return true;
    if (hasGlobalDiscountEvidenceFromLines(inv?.lines)) return true;
    const m = parseInvoiceMeta(inv?.meta);
    const pl = m?.per_line_discounts?.lines;
    return Array.isArray(pl) && pl.length > 0;
  }, [inv?.meta, inv?.lines]);

  const perLineDiscountsMetaKey = useMemo(() => {
    const raw = parseInvoiceMeta(inv?.meta)?.per_line_discounts?.lines;
    if (!Array.isArray(raw)) return '';
    try {
      return JSON.stringify(
        raw.map((r) => ({ line_id: r.line_id, percent: r.percent }))
      );
    } catch {
      return '';
    }
  }, [inv?.meta]);

  useEffect(() => {
    const raw = parseInvoiceMeta(inv?.meta)?.per_line_discounts?.lines;
    const m = {};
    if (Array.isArray(raw)) {
      for (const row of raw) {
        if (row.line_id != null && row.percent != null) {
          m[row.line_id] = String(row.percent);
        }
      }
    }
    setPerLinePercents(m);
    /* perLineDiscountsMetaKey reflète déjà la sous-partie pertinente de inv?.meta pour les remises par ligne */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inv?.id, perLineDiscountsMetaKey]);

  const draftPdfStaleHint = useMemo(() => {
    if (!allowsLineEditing) return null;
    const st = parseInvoiceMeta(inv?.meta)?.pdf?.status;
    if (st === 'failed') {
      return 'Échec génération PDF — actualisez pour réessayer.';
    }
    if (st === 'stale') {
      return 'PDF peut être obsolète — actualisez pour régénérer.';
    }
    return null;
  }, [allowsLineEditing, inv?.meta]);

  /** Chemin API JWT pour le PDF (Lot 0 SEC-06) — ne plus utiliser `/uploads/invoices/...`. */
  const invoicePdfApiPath = useMemo(() => {
    const id = inv?.id;
    const cid = companyId || inv?.company_id;
    if (!id || !cid || !String(inv?.pdf_url || '').trim()) return null;
    return buildInvoicePdfApiUrl({ id, company_id: cid });
  }, [inv?.id, inv?.company_id, inv?.pdf_url, companyId]);

  const hasStoredPdf = Boolean(String(inv?.pdf_url || '').trim());

  const pdfDownloadName = useMemo(
    () => buildInvoicePdfDownloadFilename(inv),
    [inv]
  );

  const revokeProtectedPdfBlobUrl = useCallback(() => {
    const prev = protectedPdfBlobUrlRef.current;
    if (prev) {
      URL.revokeObjectURL(prev);
      protectedPdfBlobUrlRef.current = '';
    }
    setProtectedPdfBlobUrl('');
  }, []);

  /** Précharge pdf.js + octets PDF dès l’ouverture (hors clic Imprimer). */
  useEffect(() => {
    if (!open) return undefined;
    void preloadInvoicePdfPrint().catch(() => {});

    if (!companyId || !inv?.id) return undefined;
    if (allowsLineEditing && invoiceDraftPdfNeedsRefresh(inv)) {
      printPdfBytesCacheRef.current = { key: '', bytes: null };
      return undefined;
    }
    const apiPath = buildInvoicePdfApiUrl({
      id: inv.id,
      company_id: companyId || inv.company_id,
    });
    if (!apiPath) return undefined;

    const pdfStatus = parseInvoiceMeta(inv?.meta)?.pdf?.status || '';
    const cacheKey = `${inv.id}:${String(inv?.updated_at || '')}:${pdfStatus}`;
    if (
      printPdfBytesCacheRef.current.key === cacheKey &&
      printPdfBytesCacheRef.current.bytes &&
      printPdfBytesCacheRef.current.bytes.byteLength >= 5
    ) {
      return undefined;
    }

    let cancelled = false;
    (async () => {
      try {
        const bytes = await fetchProtectedPdfBytes(apiPath, {
          cacheBust: cacheKey,
        });
        if (cancelled || !bytes || bytes.byteLength < 5) return;
        printPdfBytesCacheRef.current = { key: cacheKey, bytes: bytes.slice() };
      } catch {
        /* prefetch best-effort */
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    open,
    companyId,
    inv,
    allowsLineEditing,
  ]);

  /** Charge le PDF via API authentifiée pour l’iframe (aperçu HTML désactivé). */
  useEffect(() => {
    if (!open || showHtmlInvoicePreview || !invoicePdfApiPath) {
      revokeProtectedPdfBlobUrl();
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const url = await fetchProtectedPdfObjectUrl(invoicePdfApiPath, {
          filename: pdfDownloadName,
          cacheBust: `${inv?.updated_at || ''}:${pdfNonce}`,
        });
        if (cancelled) {
          if (url) URL.revokeObjectURL(url);
          return;
        }
        const prev = protectedPdfBlobUrlRef.current;
        if (prev) URL.revokeObjectURL(prev);
        protectedPdfBlobUrlRef.current = url || '';
        setProtectedPdfBlobUrl(url || '');
      } catch (err) {
        console.error('Chargement aperçu PDF protégé échoué:', err);
        if (!cancelled) revokeProtectedPdfBlobUrl();
      }
    })();
    return () => {
      cancelled = true;
    };
    // pdfNonce / updated_at : forcer un rechargement après régénération
  }, [
    open,
    showHtmlInvoicePreview,
    invoicePdfApiPath,
    pdfNonce,
    inv?.updated_at,
    pdfDownloadName,
    revokeProtectedPdfBlobUrl,
  ]);

  /** Révoque le blob PDF à la fermeture / démontage. */
  useEffect(() => {
    if (!open) revokeProtectedPdfBlobUrl();
    return () => {
      revokeProtectedPdfBlobUrl();
    };
  }, [open, revokeProtectedPdfBlobUrl]);

  const pdfEmbedSrc = useMemo(() => {
    if (!protectedPdfBlobUrl) return '';
    /** Réduit la barre native Chromium (`viewer-toolbar`) dans l’iframe ; pas les overlays d’extensions. */
    return appendPdfEmbedChromiumViewerFragment(protectedPdfBlobUrl);
  }, [protectedPdfBlobUrl]);

  /** Édition autorisée : aperçu HTML ; sinon iframe si PDF chargé via API. */
  const showDocumentViewer = useMemo(
    () =>
      Boolean(
        inv &&
          (showHtmlInvoicePreview || hasStoredPdf || String(pdfEmbedSrc || '').trim())
      ),
    [inv, showHtmlInvoicePreview, hasStoredPdf, pdfEmbedSrc]
  );

  /** Masque les calques injectés par l’extension Adobe Acrobat (Chrome) à côté des iframes PDF. */
  useEffect(() => {
    const active =
      open &&
      (showHtmlInvoicePreview ? Boolean(inv) : Boolean(String(pdfEmbedSrc || '').trim()));
    if (active) {
      document.body.classList.add(HIDE_ACROBAT_PDF_OVERLAY_BODY_CLASS);
    }
    return () => {
      document.body.classList.remove(HIDE_ACROBAT_PDF_OVERLAY_BODY_CLASS);
    };
  }, [open, pdfEmbedSrc, showHtmlInvoicePreview, inv]);

  useEffect(() => {
    const syncFs = () => {
      const fs =
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.msFullscreenElement;
      setIsPdfFullscreen(!!fs);
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

  /** Panneaux sous l’aperçu (ligne / remises / lignes) : Échap ferme. */
  useEffect(() => {
    if (
      (!showAddLinePanel && !showRemisePanel && !showLinesSheet) ||
      !(showHtmlInvoicePreview ? inv : pdfEmbedSrc)
    ) {
      return undefined;
    }
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setShowAddLinePanel(false);
        setShowRemisePanel(false);
        setShowLinesSheet(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showAddLinePanel, showRemisePanel, showLinesSheet, pdfEmbedSrc, showHtmlInvoicePreview, inv]);

  useEffect(() => {
    setPdfZoneExpanded(false);
    setShowLinesSheet(false);
  }, [inv?.id]);

  const handlePdfBrowserFullscreen = useCallback(() => {
    const el = pdfWrapRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
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

  /**
   * Même page (aperçu HTML inchangé) → dialogue Chrome uniquement,
   * avec le PDF backend réel (jamais l’HTML live, jamais un nouvel onglet).
   */
  const handlePdfPreviewPrint = useCallback(async () => {
    if (!companyId || !inv?.id) return;

    setSaving(true);
    setError('');

    try {
      const apiPath = buildInvoicePdfApiUrl({
        id: inv.id,
        company_id: companyId || inv.company_id,
      });

      if (!apiPath) {
        throw new Error('Aucun PDF disponible.');
      }

      const pdfStatus = parseInvoiceMeta(inv?.meta)?.pdf?.status || '';
      const cacheKey = `${inv.id}:${String(inv?.updated_at || '')}:${pdfStatus}`;

      const cached = printPdfBytesCacheRef.current;
      let bytes =
        cached.key === cacheKey &&
        cached.bytes &&
        cached.bytes.byteLength >= 5
          ? cached.bytes.slice()
          : null;

      if (!bytes) {
        bytes = await fetchProtectedPdfBytes(apiPath, {
          cacheBust: `${inv?.updated_at || ''}:${pdfStatus}`,
        });
      }

      // Facture éditable → toujours régénérer pour coller aux lignes DB (HTML).
      const needsRegen =
        allowsLineEditing || !bytes || bytes.byteLength < 5;

      if (needsRegen) {
        if (allowsLineEditing) {
          await invoiceService.regenerateInvoicePdf(companyId, inv.id);
        }
        printPdfBytesCacheRef.current = { key: '', bytes: null };
        void reloadPdfPreviewFromServer().catch(() => {});

        bytes = null;
        for (let attempt = 0; attempt < 6; attempt += 1) {
          bytes = await fetchProtectedPdfBytes(apiPath, {
            cacheBust: `${Date.now()}:${attempt}`,
          });
          if (bytes && bytes.byteLength >= 5) break;
          await new Promise((r) => {
            window.setTimeout(r, 350);
          });
        }
      }

      if (!bytes || bytes.byteLength < 5) {
        throw new Error(
          'Le PDF de la facture est indisponible. Utilisez « Télécharger » puis réessayez.'
        );
      }

      // Conserver une copie indépendante (pdf.js détache le buffer à l’impression).
      printPdfBytesCacheRef.current = {
        key: cacheKey,
        bytes: bytes.slice(),
      };

      const started = await printPdfBytes(bytes.slice());
      if (!started) {
        throw new Error(
          "Impossible d'ouvrir le dialogue d'impression. Utilisez « Télécharger » puis imprimez le fichier."
        );
      }
    } catch (error) {
      if (mountedRef.current) {
        setError(
          getApiErrorMessage(
            error,
            "Impossible de préparer le PDF pour l'impression."
          )
        );
      }
    } finally {
      if (mountedRef.current) {
        setSaving(false);
      }
    }
  }, [
    companyId,
    inv,
    allowsLineEditing,
    reloadPdfPreviewFromServer,
  ]);

  const handlePdfDownload = useCallback(async () => {
    if (!companyId || !inv?.id) return;
    setError('');
    setSaving(true);
    try {
      if (allowsLineEditing) {
        await invoiceService.regenerateInvoicePdf(companyId, inv.id);
        printPdfBytesCacheRef.current = { key: '', bytes: null };
        // Recharge l’aperçu en arrière-plan : le téléchargement n’attend pas le GET détail.
        void reloadPdfPreviewFromServer().catch(() => {});
      }
      const apiPath = buildInvoicePdfApiUrl({
        id: inv.id,
        company_id: companyId || inv.company_id,
      });
      if (!apiPath) {
        if (mountedRef.current) setError('Aucun PDF disponible.');
        return;
      }
      const ok = await downloadProtectedPdfAsFile(apiPath, pdfDownloadName, {
        cacheBust: Date.now(),
      });
      if (!mountedRef.current) return;
      if (!ok) {
        setError('Téléchargement du PDF impossible.');
      }
    } catch (e) {
      if (mountedRef.current) {
        setError(getApiErrorMessage(e, 'Téléchargement du PDF impossible.'));
      }
    } finally {
      if (mountedRef.current) setSaving(false);
    }
  }, [
    companyId,
    inv,
    allowsLineEditing,
    pdfDownloadName,
    reloadPdfPreviewFromServer,
  ]);

  const handleOpenPdfInNewTab = useCallback(async () => {
    if (!companyId || !inv?.id) return;
    const apiPath = buildInvoicePdfApiUrl({
      id: inv.id,
      company_id: companyId || inv.company_id,
    });
    if (!apiPath) {
      if (mountedRef.current) setError('Aucun PDF disponible.');
      return;
    }
    setError('');
    setSaving(true);
    try {
      if (allowsLineEditing) {
        await invoiceService.regenerateInvoicePdf(companyId, inv.id);
        printPdfBytesCacheRef.current = { key: '', bytes: null };
        void reloadPdfPreviewFromServer().catch(() => {});
      }
      const ok = await openProtectedPdfInNewTab(apiPath, null, {
        filename: pdfDownloadName,
        cacheBust: Date.now(),
      });
      if (!ok && mountedRef.current) {
        setError('Impossible d’ouvrir le PDF.');
      }
    } catch (e) {
      if (mountedRef.current) {
        setError(getApiErrorMessage(e, 'Impossible d’ouvrir le PDF.'));
      }
    } finally {
      if (mountedRef.current) setSaving(false);
    }
  }, [
    companyId,
    inv,
    allowsLineEditing,
    pdfDownloadName,
    reloadPdfPreviewFromServer,
  ]);

  /** Barre PDF : si édition lignes possible, régénère le fichier puis recharge ; sinon GET détail seulement. */
  const handleToolbarPdfRefresh = useCallback(async () => {
    if (!companyId || !(inv?.id ?? initialInvoice?.id)) return;
    const id = inv?.id ?? initialInvoice?.id;
    setSaving(true);
    setError('');
    try {
      // Invalide le cache mémoire + HTTP : même route `/pdf` sert un nouveau fichier après regen.
      printPdfBytesCacheRef.current = { key: '', bytes: null };
      if (allowsLineEditing) {
        const regen = await invoiceService.regenerateInvoicePdf(companyId, id);
        const regenUrl = regen?.pdf_url || regen?.data?.pdf_url || null;
        if (regenUrl) {
          setInv((prev) => {
            if (!prev || prev.id !== id) return prev;
            const prevMeta = parseInvoiceMeta(prev.meta) || {};
            const prevPdf =
              prevMeta.pdf && typeof prevMeta.pdf === 'object' ? prevMeta.pdf : {};
            return {
              ...prev,
              pdf_url: regenUrl,
              meta: {
                ...prevMeta,
                pdf: {
                  ...prevPdf,
                  status: 'ready',
                },
              },
            };
          });
        }
      }
      await reloadPdfPreviewFromServer();
    } catch (e) {
      if (e?.message === 'INVALID_INVOICE_PAYLOAD') {
        setError('Réponse facture invalide.');
      } else if (e?.message === 'MISSING_INVOICE_CONTEXT') {
        setError('Impossible de mettre à jour l’aperçu PDF.');
      } else {
        setError(getApiErrorMessage(e, 'Impossible de mettre à jour l’aperçu PDF.'));
      }
    } finally {
      setSaving(false);
    }
  }, [companyId, inv?.id, initialInvoice?.id, allowsLineEditing, reloadPdfPreviewFromServer]);

  if (!open || !initialInvoice) return null;

  /** Aligné sur periodAssemblyInvoiceSync : transport + livraison matière. */
  const isRideLike = (t) => {
    const typ = String(t || '').toLowerCase();
    return typ === 'ride' || typ === 'material_delivery';
  };
  const isCustom = (t) => String(t || '').toLowerCase() === 'custom';
  const isRemiseLine = (line) =>
    isCustom(line.type) && line.line_total != null && Number(line.line_total) < 0;

  /** Remise saisie comme ligne perso négative (libellé + montant) — distincte de la remise globale %. */
  const isManualDiscountLine = (line) => {
    const m = line?.line_meta;
    return (
      isCustom(line?.type) &&
      line?.line_total != null &&
      Number(line.line_total) < 0 &&
      m &&
      m.manual_discount === true
    );
  };

  const isPerLinePercentDiscountLine = (line) => {
    const m = line?.line_meta;
    return (
      isCustom(line?.type) &&
      m &&
      m.per_line_discount_line === true
    );
  };

  /** Libellé court du type de ligne pour l’en-tête de carte */
  const lineCategoryLabel = (line) => {
    const t = String(line?.type || '').toLowerCase();
    if (t === 'ride') return 'Transport';
    if (t === 'material_delivery') return 'Livraison';
    if (t !== 'custom') return t ? line.type : 'Ligne';
    if (line?.line_meta?.per_line_discount_line === true) return 'Remise par ligne';
    if (line?.line_meta?.manual_discount === true) return 'Déduction fixe';
    if (line.line_total != null && Number(line.line_total) < 0) return 'Remise';
    return 'Prestation';
  };

  const rideLikeNoun = (line) =>
    String(line?.type || '').toLowerCase() === 'material_delivery' ? 'livraison' : 'transport';

  const showLinesToolbar = lines.length >= 8;
  const rangeStart =
    filteredLines.length === 0 ? 0 : (effectiveLinePage - 1) * LINE_PAGE_SIZE + 1;
  const rangeEnd =
    filteredLines.length === 0
      ? 0
      : Math.min(effectiveLinePage * LINE_PAGE_SIZE, filteredLines.length);
  const linesListScrollClass = `${styles.tableScroll} ${styles.tableScrollDense}${
    lines.length >= HEAVY_LINES_THRESHOLD || (allowsLineEditing && showLinesSheet)
      ? ` ${styles.tableScrollHeavy}`
      : ''
  }`;

  const lineRowTitle = (line) =>
    `#${line.id} · ${lineCategoryLabel(line)}`;

  /** Un seul PATCH par ligne pour libellé, HT et note (champs présents uniquement pour ce qui est éditable). */
  const handleSaveLine = async (line) => {
    if (!allowsLineEditing) return;
    const rn = isRemiseLine(line);
    const lec = isCustom(line.type) && (!rn || isManualDiscountLine(line));
    const dEd = isRideLike(line.type) || lec;
    const aEd = isRideLike(line.type) || lec;
    const nEd = isRideLike(line.type) || lec;
    if (!dEd && !aEd && !nEd) return;

    const body = { ...draftConcurrencyPayload };

    if (dEd) {
      const rawD =
        localDescriptions[line.id] !== undefined ? localDescriptions[line.id] : line.description;
      const next = String(rawD ?? '').trim();
      if (!next) {
        setError('Le libellé ne peut pas être vide.');
        return;
      }
      body.description = next.slice(0, 500);
    }

    if (aEd) {
      const rawA = localAmounts[line.id];
      const strAmt =
        rawA !== undefined && String(rawA).trim() !== ''
          ? String(rawA).replace(',', '.')
          : String(line.line_total?.toFixed?.(2) ?? line.line_total ?? '');
      const parsed = parseFloat(strAmt);
      if (Number.isNaN(parsed)) {
        setError('Montant HT invalide.');
        return;
      }
      body.line_total = parsed;
    }

    if (nEd) {
      const rawN =
        localNotes[line.id] !== undefined ? localNotes[line.id] : line.adjustment_note;
      body.adjustment_note = rawN && String(rawN).trim() ? String(rawN) : null;
    }

    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.updateDraftInvoiceLine(companyId, inv.id, line.id, body);
      await afterDraftMutation(res);
    } catch (e) {
      setError(e?.response?.data?.error || 'Enregistrement ligne impossible.');
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveLine = async (line) => {
    if (!allowsLineEditing) return;
    const ar = isAnyRoundTripLine(line);
    const noun = rideLikeNoun(line);
    if (
      !window.confirm(
        ar
          ? `Exclure ce ${noun} de la facture ? L’autre jambe aller-retour restera facturée séparément si elle existe.`
          : `Exclure ce ${noun} de la facture ? Le montant sera retiré du brouillon et le ${noun} redeviendra facturable.`
      )
    ) {
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.removeDraftInvoiceLine(
        companyId,
        inv.id,
        line.id,
        draftConcurrencyPayload
      );
      await afterDraftMutation(res);
    } catch (e) {
      setError(getApiErrorMessage(e, 'Suppression impossible'));
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveRoundTripLeg = async (line, leg) => {
    if (!allowsLineEditing) return;
    const legLabel = leg === 'return' ? 'retour' : 'aller';
    if (
      !window.confirm(
        `Exclure uniquement la jambe ${legLabel} de ce transport aller-retour ? L’autre jambe restera sur la facture.`
      )
    ) {
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.removeDraftInvoiceLine(
        companyId,
        inv.id,
        line.id,
        {
          ...draftConcurrencyPayload,
          exclude_round_trip_leg: leg,
        }
      );
      await afterDraftMutation(res);
    } catch (e) {
      setError(getApiErrorMessage(e, 'Exclusion de la jambe impossible'));
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveRemise = async () => {
    if (!allowsLineEditing || !inv?.id) return;
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.removeDraftGlobalDiscount(
        companyId,
        inv.id,
        draftConcurrencyPayload
      );
      await afterDraftMutation(res);
      setShowRemisePanel(false);
    } catch (e) {
      setError(getApiErrorMessage(e, 'Annulation de remise impossible'));
    } finally {
      setSaving(false);
    }
  };

  const handleAddCustomLine = async () => {
    if (!allowsLineEditing || !inv?.id) return;
    if (!customLineDesc.trim()) {
      setError('Indiquez un libellé.');
      return;
    }
    /* Laisse le blur du date picker appliquer onChange(iso) avant la lecture (clic « Ajouter »). */
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
    let lineTotal;
    let qty;
    if (customLineMode === EXTRA_LINE_MODE.quantity) {
      const u = parseFloat(String(customLineUnitPrice).replace(',', '.'));
      const q = parseFloat(String(customLineQty).replace(',', '.'));
      if (!Number.isFinite(u) || u <= 0) {
        setError('Prix unitaire HT invalide.');
        return;
      }
      if (!Number.isFinite(q) || q <= 0) {
        setError('Quantité invalide.');
        return;
      }
      lineTotal = u * q;
      qty = q;
    } else {
      const t = parseFloat(String(customLineTaux).replace(',', '.'));
      const v = parseFloat(String(customLineTimeValue).replace(',', '.'));
      if (!Number.isFinite(t) || t <= 0) {
        setError('Prix (CHF) par unité de temps invalide.');
        return;
      }
      if (!Number.isFinite(v) || v <= 0) {
        setError('Valeur temps invalide.');
        return;
      }
      lineTotal = t * v;
      qty = v;
    }
    const rawSvcDate = (
      customLineServiceDateRef.current ||
      customLineServiceDate ||
      ''
    ).trim();
    let serviceDateIso;
    if (rawSvcDate) {
      serviceDateIso = normalizeServiceDateToIsoForApi(rawSvcDate);
      if (!serviceDateIso) {
        setError('Date de prestation invalide (utilisez JJ.MM.AAAA ou AAAA-MM-JJ).');
        return;
      }
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.addDraftCustomLine(companyId, inv.id, {
        ...draftConcurrencyPayload,
        description: customLineDesc.trim(),
        line_total: lineTotal,
        qty,
        custom_mode: customLineMode === EXTRA_LINE_MODE.time ? 'time' : 'quantity',
        time_unit: customLineMode === EXTRA_LINE_MODE.time ? customLineTimeUnit : undefined,
        service_date_iso: serviceDateIso,
      });
      await afterDraftMutation(res);
      setCustomLineDesc('');
      setCustomLineTaux('');
      setCustomLineTimeValue('1');
      setCustomLineTimeUnit('h');
      setCustomLineUnitPrice('');
      setCustomLineQty('1');
      customLineServiceDateRef.current = '';
      setCustomLineServiceDate('');
      setShowAddLinePanel(false);
    } catch (e) {
      setError(e?.response?.data?.error || 'Ligne non ajoutée');
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveCustomOrExtraLine = async (line) => {
    if (!allowsLineEditing) return;
    if (
      !window.confirm('Retirer cette ligne du brouillon ?')
    ) {
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.removeDraftInvoiceLine(
        companyId,
        inv.id,
        line.id,
        draftConcurrencyPayload
      );
      await afterDraftMutation(res);
    } catch (e) {
      setError(getApiErrorMessage(e, 'Suppression impossible'));
    } finally {
      setSaving(false);
    }
  };

  const handleAddFreeRemiseLine = async () => {
    if (!allowsLineEditing || !inv?.id) return;
    const desc = freeRemiseDesc.trim();
    const amt = parseFloat(String(freeRemiseAmount).replace(',', '.'));
    if (!desc) {
      setError('Indiquez un libellé pour cette déduction (ex. Geste commercial, avoir ponctuel).');
      return;
    }
    if (!Number.isFinite(amt) || amt <= 0) {
      setError(
        'Indiquez un montant HT en nombre positif (ex. 50 pour retrancher 50,00 CHF HT sur la facture).'
      );
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.addDraftCustomLine(companyId, inv.id, {
        ...draftConcurrencyPayload,
        description: desc,
        line_total: -amt,
        qty: 1,
      });
      await afterDraftMutation(res);
      setFreeRemiseDesc('');
      setFreeRemiseAmount('');
    } catch (e) {
      setError(e?.response?.data?.error || 'Déduction non ajoutée');
    } finally {
      setSaving(false);
    }
  };

  const handleApplyDiscount = async () => {
    const p = parseFloat(String(globalPct).replace(',', '.'));
    if (!allowsLineEditing || !Number.isFinite(p) || p <= 0 || p > 100) {
      setError('Indiquez un pourcentage de remise entre 0 et 100.');
      return;
    }
    if (!hasRemisablePositiveLines) {
      setError('Aucune ligne HT à remiser (transport ou prestation).');
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.applyDraftGlobalDiscount(companyId, inv.id, {
        ...draftConcurrencyPayload,
        global_discount_percent: p,
        global_discount_note: globalNote || null,
      });
      await afterDraftMutation(res);
      setShowRemisePanel(false);
    } catch (e) {
      setError(e?.response?.data?.error || 'Remise non appliquée');
    } finally {
      setSaving(false);
    }
  };

  const handleApplyPerLineDiscounts = async () => {
    if (!allowsLineEditing || !inv?.id) return;
    const line_discounts = [];
    for (const line of perLineDiscountableLines) {
      const lid = line.id;
      if (lid == null) continue;
      const raw = perLinePercents[lid];
      if (raw === undefined || raw === '' || String(raw).trim() === '') continue;
      const p = parseFloat(String(raw).replace(',', '.'));
      if (!Number.isFinite(p) || p <= 0 || p > 100) {
        setError('Chaque pourcentage doit être un nombre entre 0 et 100 (champs remplis uniquement).');
        return;
      }
      line_discounts.push({ line_id: lid, percent: p });
    }
    if (line_discounts.length === 0) {
      setError(
        'Indiquez au moins un pourcentage sur une ligne remisable (laissez vide les lignes sans remise).'
      );
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.applyDraftPerLineDiscounts(companyId, inv.id, {
        ...draftConcurrencyPayload,
        line_discounts,
      });
      await afterDraftMutation(res);
      setShowRemisePanel(false);
    } catch (e) {
      setError(e?.response?.data?.error || 'Remises par ligne non appliquées');
    } finally {
      setSaving(false);
    }
  };

  const remisePanelContentCompact = (
    <div className={styles.remiseSheet}>
      <div className={styles.remiseSheetTitleRow}>
        <h3 className={styles.remiseSheetTitle} id={remiseHeadingId}>
          Remises
        </h3>
        {hasActivePercentRemise ? (
          <span className={styles.remiseActiveBadge}>
            Remise active
            {(() => {
              const meta = parseInvoiceMeta(inv?.meta);
              const g = meta?.global_discount;
              let p = g ? normalizeGlobalDiscountPercent(g) : null;
              if (p == null) {
                p =
                  inferGlobalDiscountPercentFromOriginalLineMeta(inv?.lines) ??
                  inferPercentFromRemiseCommercialeLine(inv?.lines);
              }
              return p != null ? ` : ${p} %` : '';
            })()}
          </span>
        ) : null}
      </div>

      <div
        className={styles.remiseModeSeg}
        role="group"
        aria-label="Remise en pourcentage sur les transports"
      >
        <button
          type="button"
          className={
            remisePercentMode === REMISE_PCT_MODE.global
              ? styles.remiseModeSegBtnActive
              : styles.remiseModeSegBtn
          }
          disabled={saving}
          onClick={() => setRemisePercentMode(REMISE_PCT_MODE.global)}
        >
          Globale
        </button>
        <button
          type="button"
          className={
            remisePercentMode === REMISE_PCT_MODE.perLine
              ? styles.remiseModeSegBtnActive
              : styles.remiseModeSegBtn
          }
          disabled={saving}
          onClick={() => setRemisePercentMode(REMISE_PCT_MODE.perLine)}
        >
          Par ligne
        </button>
      </div>

      {remisePercentMode === REMISE_PCT_MODE.global && (
        <>
          <section className={styles.remiseSheetSection} aria-label="Remise globale">
            <div className={`${styles.formRow} ${styles.formRowHarmonized} ${styles.remiseSheetRow}`}>
              <label className={styles.srOnly} htmlFor="draft-gd-pct-toolbar">
                Pourcentage
              </label>
              <input
                id="draft-gd-pct-toolbar"
                className={styles.input}
                type="text"
                inputMode="decimal"
                placeholder="%"
                autoComplete="off"
                value={globalPct}
                onChange={(e) => setGlobalPct(e.target.value)}
              />
              <input
                className={styles.inputGrow}
                type="text"
                placeholder="Note"
                value={globalNote}
                onChange={(e) => setGlobalNote(e.target.value)}
              />
              <button
                type="button"
                className={styles.btn}
                disabled={saving || !hasRemisablePositiveLines}
                onClick={() => void handleApplyDiscount()}
              >
                Appliquer
              </button>
              <button
                type="button"
                className={styles.btnMuted}
                disabled={saving || !hasActivePercentRemise}
                onClick={() => void handleRemoveRemise()}
              >
                Retirer
              </button>
            </div>
          </section>
          <section className={styles.remiseSheetSection} aria-label="Déduction fixe HT">
            <div className={styles.remiseSheetSectionHead}>Déduction CHF HT</div>
            <p className={styles.remiseSheetFoot}>
              Après validation, la déduction apparaît dans le tableau{' '}
              <strong>« Tableau des lignes »</strong> (ligne personnalisée en montant HT négatif), mise à jour du bandeau{' '}
              <strong>TTC</strong> et du PDF.
            </p>
            <div className={`${styles.formRow} ${styles.formRowHarmonized} ${styles.remiseSheetRow}`}>
              <input
                className={styles.inputGrow}
                type="text"
                placeholder="Libellé"
                value={freeRemiseDesc}
                onChange={(e) => setFreeRemiseDesc(e.target.value)}
                aria-label="Libellé"
              />
              <label className={styles.srOnly} htmlFor="draft-free-remise-amt">
                Montant HT
              </label>
              <input
                id="draft-free-remise-amt"
                className={styles.input}
                type="text"
                inputMode="decimal"
                placeholder="CHF HT"
                autoComplete="off"
                value={freeRemiseAmount}
                onChange={(e) => setFreeRemiseAmount(e.target.value)}
              />
              <button
                type="button"
                className={styles.btn}
                disabled={saving}
                onClick={() => void handleAddFreeRemiseLine()}
              >
                Ajouter
              </button>
            </div>
            {freeRemiseImpactPreview ? (
              <>
                <p className={styles.remiseDeductionCalc} aria-live="polite">
                  Impact sur le total TTC actuel ({formatCurrencyCHF(draftTotalTtc)}) :{' '}
                  <strong>−{formatCurrencyCHF(freeRemiseImpactPreview.htDeduction)}</strong> HT (sans TVA sur cette
                  ligne) → total TTC estimé{' '}
                  <strong>{formatCurrencyCHF(freeRemiseImpactPreview.newTtc)}</strong>.
                </p>
                {freeRemiseImpactPreview.newTtc < 0 ? (
                  <p className={styles.remiseDeductionWarn}>
                    Le total TTC deviendrait négatif : réduisez le montant ou vérifiez les lignes.
                  </p>
                ) : null}
              </>
            ) : null}
          </section>
        </>
      )}

      {remisePercentMode === REMISE_PCT_MODE.perLine && (
        <section className={styles.remiseSheetSection} aria-label="Remise par ligne">
          {perLineDiscountableLines.length > 0 && (
            <ul className={styles.remisePerLineList}>
              {perLineDiscountableLines.map((line) => {
                const lid = line.id;
                const isCustomLine = String(line?.type || '').toLowerCase() === 'custom';
                const rawDesc = String(line.description || '').trim() || (isCustomLine ? 'Prestation' : 'Transport');
                const shortDesc = rawDesc.length > 72 ? `${rawDesc.slice(0, 69)}…` : rawDesc;
                const ht = line.line_total;
                const htLabel = Number.isFinite(Number(ht)) ? formatCurrencyCHF(Number(ht)) : '—';
                return (
                  <li key={lid} className={styles.remisePerLineItem}>
                    <div className={styles.remisePerLineDesc} title={rawDesc}>
                      {shortDesc}
                    </div>
                    <label className={styles.remisePerLinePctWrap} htmlFor={`draft-pl-pct-${lid}`}>
                      <span className={styles.srOnly}>Pourcentage</span>
                      <input
                        id={`draft-pl-pct-${lid}`}
                        className={styles.remisePerLinePctInput}
                        type="text"
                        inputMode="decimal"
                        placeholder="%"
                        autoComplete="off"
                        disabled={saving}
                        value={perLinePercents[lid] ?? ''}
                        onChange={(e) =>
                          setPerLinePercents((prev) => ({
                            ...prev,
                            [lid]: e.target.value,
                          }))
                        }
                      />
                    </label>
                    <span className={styles.remisePerLineHt}>{htLabel}</span>
                  </li>
                );
              })}
            </ul>
          )}
          <div className={`${styles.formRow} ${styles.formRowHarmonized} ${styles.remiseSheetRow}`}>
            <button
              type="button"
              className={styles.btn}
              disabled={saving || perLineDiscountableLines.length === 0}
              onClick={() => void handleApplyPerLineDiscounts()}
            >
              Appliquer
            </button>
            <button
              type="button"
              className={styles.btnMuted}
              disabled={saving || !hasActivePercentRemise}
              onClick={() => void handleRemoveRemise()}
            >
              Retirer
            </button>
          </div>
        </section>
      )}
    </div>
  );

  const addLinePanelContent = (
      <div
        className={styles.addLineForm}
        title="Prix unitaire × quantité, ou taux × durée (unité) selon le mode."
      >
        <div className={styles.addLineFormHeader}>
          <h3 className={styles.addLineFormTitle} id={addLineHeadingId}>
            Ligne supplémentaire HT
          </h3>
          <div className={styles.modeSegSm} role="group" aria-label="Type de facturation">
            <button
              type="button"
              className={
                customLineMode === EXTRA_LINE_MODE.time
                  ? styles.modeSegBtnActiveSm
                  : styles.modeSegBtnSm
              }
              onClick={() => setCustomLineMode(EXTRA_LINE_MODE.time)}
            >
              Temps
            </button>
            <button
              type="button"
              className={
                customLineMode === EXTRA_LINE_MODE.quantity
                  ? styles.modeSegBtnActiveSm
                  : styles.modeSegBtnSm
              }
              onClick={() => setCustomLineMode(EXTRA_LINE_MODE.quantity)}
            >
              Qté
            </button>
          </div>
        </div>
        <div className={styles.addLineFormFields}>
          <input
            className={styles.inputLibelle}
            type="text"
            placeholder="Libellé"
            value={customLineDesc}
            onChange={(e) => setCustomLineDesc(e.target.value)}
          />
          <div className={styles.addLineFormMainRow}>
            <div className={styles.addLineFormDateGroup}>
              {serviceDateMonthYearOnly ? (
                <InlineMonthYearPicker
                  inputId={`${addLineHeadingId}-svc-date`}
                  className={styles.addLineFormDatePickerWrap}
                  value={
                    customLineServiceDate.trim().length >= 7
                      ? customLineServiceDate.trim().slice(0, 7)
                      : ''
                  }
                  onChange={(ym) => {
                    const next = ym ? `${ym}-01` : '';
                    customLineServiceDateRef.current = next;
                    setCustomLineServiceDate(next);
                  }}
                  ariaLabel="Mois de la prestation (optionnel)"
                  title="Mois (optionnel)"
                />
              ) : (
                <InlineDatePicker
                  inputId={`${addLineHeadingId}-svc-date`}
                  className={styles.addLineFormDatePickerWrap}
                  value={customLineServiceDate}
                  onChange={(iso) => {
                    const next = iso || '';
                    customLineServiceDateRef.current = next;
                    setCustomLineServiceDate(next);
                  }}
                  ariaLabel="Date de la prestation (optionnel)"
                  title="Date (optionnel)"
                />
              )}
            </div>
            {customLineMode === EXTRA_LINE_MODE.time ? (
              <div className={styles.addLineFormToolbar}>
                <input
                  className={styles.inCell}
                  type="text"
                  inputMode="decimal"
                  autoComplete="off"
                  placeholder="Taux"
                  aria-label={`Taux ${tauxSuffixForUnit(customLineTimeUnit)}`}
                  value={customLineTaux}
                  onChange={(e) => setCustomLineTaux(e.target.value)}
                />
                <input
                  className={styles.inCellSm}
                  type="text"
                  inputMode="decimal"
                  autoComplete="off"
                  placeholder="1"
                  aria-label="Durée (valeur)"
                  value={customLineTimeValue}
                  onChange={(e) => setCustomLineTimeValue(e.target.value)}
                />
                <select
                  className={styles.selectSm}
                  value={customLineTimeUnit}
                  onChange={(e) => setCustomLineTimeUnit(e.target.value)}
                  aria-label="Unité de temps"
                >
                  {TIME_UNITS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
                {customLinePreview != null && (
                  <span className={styles.previewChfSm} aria-live="polite">
                    {formatCurrencyCHF(customLinePreview)}
                  </span>
                )}
                <button
                  type="button"
                  className={styles.btnIconAdd}
                  disabled={saving}
                  title="Ajouter la ligne"
                  onClick={() => void handleAddCustomLine()}
                >
                  <FiPlus size={18} />
                </button>
              </div>
            ) : (
              <div className={styles.addLineFormToolbar}>
                <input
                  className={styles.inCell}
                  type="text"
                  inputMode="decimal"
                  autoComplete="off"
                  placeholder="Prix u. (CHF)"
                  aria-label="Prix unitaire HT en CHF"
                  value={customLineUnitPrice}
                  onChange={(e) => setCustomLineUnitPrice(e.target.value)}
                />
                <input
                  className={styles.inCellSm}
                  type="text"
                  inputMode="decimal"
                  autoComplete="off"
                  placeholder="Qté"
                  aria-label="Quantité"
                  value={customLineQty}
                  onChange={(e) => setCustomLineQty(e.target.value)}
                />
                {customLinePreview != null && (
                  <span className={styles.previewChfSm} aria-live="polite">
                    {formatCurrencyCHF(customLinePreview)}
                  </span>
                )}
                <button
                  type="button"
                  className={styles.btnIconAdd}
                  disabled={saving}
                  title="Ajouter la ligne"
                  onClick={() => void handleAddCustomLine()}
                >
                  <FiPlus size={18} />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
  );


  const draftLinesEditorInner = (
    <>
      {showLinesToolbar && (
        <div className={`${styles.linesToolbar} ${styles.linesToolbarDense}`}>
          <div className={styles.linesToolbarFilterWrap}>
            <FiSearch className={styles.linesToolbarFilterIcon} size={16} aria-hidden />
            <input
              type="search"
              className={styles.linesToolbarInput}
              placeholder="Client, date (JJ.MM.AAAA), libellé, montant, n° ligne…"
              title="Ex. : Bouchardy · Jean-michel Bouchardy · 02.05.2026 · 02.05 · mai · HUG · 80"
              value={lineFilter}
              onChange={(e) => setLineFilter(e.target.value)}
              autoComplete="off"
              aria-label="Filtrer les lignes de la facture"
            />
          </div>
          <span className={styles.linesToolbarMeta}>
            {filteredLines.length === 0
              ? 'Aucun résultat'
              : `Lignes ${rangeStart}–${rangeEnd} sur ${filteredLines.length}`}
          </span>
          {totalLinePages > 1 && (
            <div className={styles.linesToolbarPager}>
              <button
                type="button"
                className={styles.btnPager}
                disabled={effectiveLinePage <= 1}
                title="Page précédente"
                aria-label="Page précédente"
                onClick={() => setLinePage((p) => Math.max(1, p - 1))}
              >
                <FiChevronLeft size={18} aria-hidden />
              </button>
              <span className={styles.linesToolbarMeta}>
                {effectiveLinePage} / {totalLinePages}
              </span>
              <button
                type="button"
                className={styles.btnPager}
                disabled={effectiveLinePage >= totalLinePages}
                title="Page suivante"
                aria-label="Page suivante"
                onClick={() =>
                  setLinePage((p) => Math.min(totalLinePages, p + 1))
                }
              >
                <FiChevronRight size={18} aria-hidden />
              </button>
            </div>
          )}
        </div>
      )}

      <div className={linesListScrollClass}>
        {lines.length > 0 && filteredLines.length > 0 ? (
          <div className={styles.linesColumnLegend} aria-hidden="true">
            <span>Libellé</span>
            <span className={styles.linesColumnLegendHt}>HT (CHF)</span>
            <span>Note</span>
            <span className={styles.linesColumnLegendActions}> </span>
          </div>
        ) : null}
        <table className={`${styles.table} ${styles.tableDense}`}>
          <caption className={styles.srOnly}>
            Lignes de facture&nbsp;: libellé, montant HT, note, puis une action enregistrer par ligne avant
            régénération du PDF.
          </caption>
          <colgroup>
            <col className={styles.colDesc} />
            <col className={styles.colHt} />
            <col className={styles.colNoteCol} />
            <col className={styles.colActions} />
          </colgroup>
          <tbody>
            {lines.length === 0 ? (
              <tr>
                <td colSpan={4} className={styles.tableEmptyTight}>
                  Aucune ligne.
                </td>
              </tr>
            ) : filteredLines.length === 0 ? (
              <tr>
                <td colSpan={4} className={styles.tableEmptyTight}>
                  Rien pour ce filtre.
                </td>
              </tr>
            ) : (
              paginatedLines.map((line) => {
                const remiseNeg = isRemiseLine(line);
                const lineEditCustom =
                  isCustom(line.type) && (!remiseNeg || isManualDiscountLine(line));
                const descEditable = allowsLineEditing && (isRideLike(line.type) || lineEditCustom);
                const amountEditable = allowsLineEditing && (isRideLike(line.type) || lineEditCustom);
                const noteEditable = allowsLineEditing && (isRideLike(line.type) || lineEditCustom);
                const descId = `inv-line-desc-${line.id}`;
                const htId = `inv-line-ht-${line.id}`;
                const noteId = `inv-line-note-${line.id}`;
                const remiseHintTitle =
                  remiseNeg && !isManualDiscountLine(line)
                    ? isPerLinePercentDiscountLine(line)
                      ? 'Remise ligne % — panneau Remises.'
                      : 'Remise globale % — panneau Remises.'
                    : undefined;
                const rowNeedsApply = descEditable || amountEditable || noteEditable;
                const rowAmountNegative =
                  line.line_total != null && Number(line.line_total) < 0;
                const showArLegExclude = canShowRoundTripLegExcludeActions(line);
                const rowClassNames = [
                  rowAmountNegative ? styles.rowAmountNegative : '',
                  isRoundTripPreviewHiddenLine(line) ? styles.rowRoundTripReturn : '',
                ]
                  .filter(Boolean)
                  .join(' ');
                return (
                  <tr
                    key={lineKey(line)}
                    title={lineRowTitle(line)}
                    className={rowClassNames || undefined}
                  >
                    <td className={styles.colDescCell}>
                      <div className={styles.lineEditorDescStack}>
                        <InvoiceLineEditorContext
                          line={line}
                          styles={styles}
                          legActions={
                            allowsLineEditing &&
                            isRideLike(line.type) &&
                            line.reservation_id &&
                            showArLegExclude
                              ? {
                                  enabled: true,
                                  disabled: saving,
                                  onExcludeLeg: (leg) =>
                                    void handleRemoveRoundTripLeg(line, leg),
                                }
                              : undefined
                          }
                        />
                        <div className={styles.denseDesc}>
                          {descEditable ? (
                            <textarea
                              id={descId}
                              className={styles.denseTextarea}
                              rows={1}
                              defaultValue={line.description || ''}
                              onChange={(e) =>
                                setLocalDescriptions((prev) => ({
                                  ...prev,
                                  [line.id]: e.target.value,
                                }))
                              }
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                  e.preventDefault();
                                  if (!saving) void handleSaveLine(line);
                                }
                              }}
                              aria-label={`Libellé · ${lineRowTitle(line)}`}
                              title={`Ctrl+Entrée · enregistrer la ligne · ${line.id}${remiseNeg ? ` · ${remiseHintTitle || 'Remise'}` : ''}${(line.line_meta?.description_overridden || line.line_meta?.amount_overridden) ? ' · modifié' : ''}`}
                            />
                          ) : (
                            <div
                              className={styles.denseStatic}
                              title={[line.description || line.type, lineRowTitle(line), remiseHintTitle]
                                .filter(Boolean)
                                .join(' — ')}
                            >
                              {line.description || line.type}
                            </div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className={styles.colHtCell}>
                      {amountEditable ? (
                        <input
                          id={htId}
                          className={`${styles.denseHt} ${styles.denseHtGrow}`}
                          type="text"
                          inputMode="decimal"
                          defaultValue={String(
                            line.line_total?.toFixed?.(2) ?? line.line_total
                          )}
                          onChange={(e) =>
                            setLocalAmounts((prev) => ({
                              ...prev,
                              [line.id]: e.target.value,
                            }))
                          }
                          aria-label="Montant HT"
                        />
                      ) : (
                        <span className={styles.denseHtRead}>{formatCurrencyCHF(line.line_total)}</span>
                      )}
                    </td>
                    <td className={styles.colNote}>
                      {noteEditable ? (
                        <textarea
                          id={noteId}
                          className={`${styles.denseTextarea} ${styles.denseTextareaNote}`}
                          rows={1}
                          defaultValue={line.adjustment_note || ''}
                          onChange={(e) =>
                            setLocalNotes((prev) => ({
                              ...prev,
                              [line.id]: e.target.value,
                            }))
                          }
                          aria-label="Note"
                        />
                      ) : (
                        <span className={styles.denseNoteRead}>{line.adjustment_note?.trim() || '—'}</span>
                      )}
                    </td>
                    <td className={styles.colActionsCell}>
                      <div className={styles.denseActions}>
                        {rowNeedsApply ? (
                          <button
                            type="button"
                            className={`${styles.btnIconOkXs} ${styles.btnLineSave}`}
                            disabled={saving}
                            title="Enregistrer cette ligne sur le serveur"
                            aria-label={`Enregistrer la ligne ${line.id}`}
                            onClick={() => void handleSaveLine(line)}
                          >
                            <FiCheck size={14} aria-hidden />
                          </button>
                        ) : null}
                        {allowsLineEditing && isRideLike(line.type) && line.reservation_id && (
                          <button
                            type="button"
                            className={`${styles.btnTrashXs} ${styles.danger}`}
                            disabled={saving}
                            title={
                              showArLegExclude
                                ? 'Retirer l’aller-retour complet de la facture'
                                : `Exclure ce ${rideLikeNoun(line)} de la facture`
                            }
                            aria-label={`Exclure ligne ${line.id}`}
                            onClick={() => handleRemoveLine(line)}
                          >
                            <FiTrash2 size={13} />
                          </button>
                        )}
                        {allowsLineEditing && lineEditCustom && (
                          <button
                            type="button"
                            className={`${styles.btnTrashXs} ${styles.danger}`}
                            disabled={saving}
                            title="Retirer"
                            aria-label={`Retirer ligne ${line.id}`}
                            onClick={() => handleRemoveCustomOrExtraLine(line)}
                          >
                            <FiTrash2 size={13} />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </>
  );

  const draftLinesEditorBody = (
    <div className={styles.draftLinesFoldBody}>{draftLinesEditorInner}</div>
  );

  const linesSheetPanelContent = (
    <div className={styles.linesSheetPanel}>
      <header className={styles.linesSheetHead}>
        <h3 className={styles.linesSheetHeading} id={linesSheetHeadingId}>
          Lignes
        </h3>
        <div className={styles.linesSheetHeadHint} role="note">
          Une ✓ par ligne · Ctrl+Entrée même effet
        </div>
      </header>
      <div className={styles.linesSheetEditor}>{draftLinesEditorInner}</div>
    </div>
  );

  return (
        <div className={styles.body}>
          {loading && !inv && <p className={styles.muted}>Chargement…</p>}
          {error && <p className={styles.err}>{error}</p>}

          {inv && (
            <>
              {showDocumentViewer ? (
                <div
                  ref={pdfWrapRef}
                  className={`${styles.draftPdfWrap}${
                    pdfZoneExpanded ? ` ${styles.draftPdfWrapExpanded}` : ''
                  }`}
                >
                  <div
                    className={styles.draftPdfBar}
                    role="toolbar"
                    aria-label={
                      allowsLineEditing ? 'Édition et aperçu facture' : 'Document PDF'
                    }
                  >
                    <div className={styles.draftPdfBarLeft}>
                      <div className={styles.draftPdfBarHeadText}>
                        <h3 className={styles.draftPdfHeadTitle}>
                          {allowsLineEditing ? 'Aperçu facture' : 'Facture PDF'}
                        </h3>
                        <p className={styles.draftPdfHeadSubtitle}>
                          {inv?.invoice_number || '—'}
                          {draftPdfStaleHint ? (
                            <span className={styles.draftPdfHeadMeta} role="status">
                              {' '}
                              · {draftPdfStaleHint}
                            </span>
                          ) : null}
                          {saving ? (
                            <span className={styles.draftPdfHeadMeta} aria-live="polite">
                              {' '}
                              · Mise à jour…
                            </span>
                          ) : null}
                        </p>
                      </div>
                    </div>
                    <div className={styles.draftPdfBarRightMerged}>
                      {allowsLineEditing ? (
                        <div
                          className={styles.draftPdfBarToolGroup}
                          role="group"
                          aria-label="Édition des lignes et remises"
                        >
                          <button
                            type="button"
                            className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly}${
                              showRemisePanel || hasActivePercentRemise
                                ? ` ${styles.draftPdfToolBtnActive}`
                                : ''
                            }`}
                            disabled={saving}
                            title="Remises"
                            aria-label="Remises"
                            aria-pressed={Boolean(showRemisePanel || hasActivePercentRemise)}
                            onClick={() => {
                              setShowRemisePanel((v) => !v);
                              setShowAddLinePanel(false);
                              setShowLinesSheet(false);
                            }}
                          >
                            <FiPercent size={16} aria-hidden />
                          </button>
                          <button
                            type="button"
                            className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly}${
                              showAddLinePanel ? ` ${styles.draftPdfToolBtnActive}` : ''
                            }`}
                            disabled={saving}
                            title="Ligne supplémentaire HT"
                            aria-label="Ajouter une ligne supplémentaire HT"
                            aria-pressed={showAddLinePanel}
                            onClick={() => {
                              setShowAddLinePanel((v) => !v);
                              setShowRemisePanel(false);
                              setShowLinesSheet(false);
                            }}
                          >
                            <FiPlus size={16} aria-hidden />
                          </button>
                          <button
                            type="button"
                            className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly}${
                              showLinesSheet ? ` ${styles.draftPdfToolBtnActive}` : ''
                            }`}
                            disabled={saving}
                            title="Modifier les lignes (libellés, montants, notes)"
                            aria-label="Ouvrir l’édition des lignes sous l’aperçu PDF"
                            aria-pressed={showLinesSheet}
                            onClick={() => {
                              setShowLinesSheet((v) => !v);
                              setShowRemisePanel(false);
                              setShowAddLinePanel(false);
                            }}
                          >
                            <FiList size={16} aria-hidden />
                          </button>
                          <button
                            type="button"
                            className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly}`}
                            disabled={saving || loading || !inv?.id}
                            title="Régénérer le PDF (impression / téléchargement) et recharger les données"
                            aria-label="Régénérer le PDF et actualiser les données depuis le serveur"
                            onClick={() => void handleToolbarPdfRefresh()}
                          >
                            <FiRefreshCw size={16} aria-hidden />
                          </button>
                        </div>
                      ) : null}
                      <div
                        className={styles.draftPdfBarToolGroup}
                        role="group"
                        aria-label="Affichage"
                      >
                        <button
                          type="button"
                          className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly}`}
                          title={
                            pdfZoneExpanded
                              ? 'Réduire la zone d’aperçu'
                              : 'Agrandir la zone d’aperçu'
                          }
                          aria-label={
                            pdfZoneExpanded
                              ? 'Réduire la zone d’aperçu PDF'
                              : 'Agrandir la zone d’aperçu PDF'
                          }
                          aria-pressed={pdfZoneExpanded}
                          onClick={() => setPdfZoneExpanded((v) => !v)}
                        >
                          {pdfZoneExpanded ? (
                            <FiChevronsUp size={18} aria-hidden />
                          ) : (
                            <FiChevronsDown size={18} aria-hidden />
                          )}
                        </button>
                        <button
                          type="button"
                          className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly}`}
                          title={
                            isPdfFullscreen
                              ? 'Quitter le plein écran'
                              : 'Plein écran (navigateur)'
                          }
                          aria-label={
                            isPdfFullscreen
                              ? 'Quitter le plein écran'
                              : 'Plein écran dans le navigateur'
                          }
                          aria-pressed={isPdfFullscreen}
                          onClick={() => handlePdfBrowserFullscreen()}
                        >
                          {isPdfFullscreen ? (
                            <FiMinimize2 size={18} aria-hidden />
                          ) : (
                            <FiMaximize2 size={18} aria-hidden />
                          )}
                        </button>
                      </div>
                      <div
                        className={`${styles.draftPdfBarToolGroup} ${styles.draftPdfBarToolGroupFile}`}
                        role="group"
                        aria-label="Fichier PDF"
                      >
                        {allowsLineEditing || hasStoredPdf ? (
                          <button
                            type="button"
                            className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly} ${styles.draftPdfToolLink}`}
                            title={
                              allowsLineEditing
                                ? 'Générer le PDF à jour puis télécharger'
                                : 'Télécharger le fichier PDF'
                            }
                            aria-label={
                              allowsLineEditing
                                ? 'Générer et télécharger le PDF'
                                : 'Télécharger le fichier PDF'
                            }
                            onClick={() => void handlePdfDownload()}
                          >
                            <FiDownload size={18} aria-hidden />
                          </button>
                        ) : null}
                        <button
                          type="button"
                          className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly}`}
                          title={
                            allowsLineEditing
                              ? 'Imprimer la facture PDF (reste sur cette page)'
                              : 'Imprimer la facture PDF'
                          }
                          aria-label={
                            allowsLineEditing
                              ? 'Imprimer le PDF de la facture sans quitter l’aperçu'
                              : 'Imprimer le PDF de la facture'
                          }
                          onClick={() => void handlePdfPreviewPrint()}
                        >
                          <FiPrinter size={18} aria-hidden />
                        </button>
                        {hasStoredPdf ? (
                          <button
                            type="button"
                            className={`${styles.draftPdfToolBtn} ${styles.draftPdfToolBtnIconOnly} ${styles.draftPdfToolLink}`}
                            title="Ouvrir dans un nouvel onglet"
                            aria-label="Ouvrir le PDF dans un nouvel onglet"
                            onClick={() => void handleOpenPdfInNewTab()}
                          >
                            <FiExternalLink size={18} aria-hidden />
                          </button>
                        ) : null}
                      </div>
                    </div>
                  </div>
                  <div
                    className={`${styles.draftPdfViewerStack}${
                      allowsLineEditing && (showRemisePanel || showAddLinePanel || showLinesSheet)
                        ? ` ${styles.draftPdfViewerStackWithSheet}`
                        : ''
                    }`}
                  >
                    {showHtmlInvoicePreview ? (
                      <InvoiceLivePreview
                        invoice={inv}
                        companyVatApplicable={companyVatApplicable}
                        className={styles.draftLivePreviewMount}
                      />
                    ) : (
                      <iframe
                        key={pdfNonce}
                        ref={pdfIframeRef}
                        className={styles.draftPdfIframe}
                        title="Aperçu du PDF de la facture"
                        src={pdfEmbedSrc}
                        allow="fullscreen"
                      />
                    )}
                    {allowsLineEditing && (showRemisePanel || showAddLinePanel || showLinesSheet) ? (
                      <div
                        className={`${styles.draftPdfLineSheetFlow}${
                          showLinesSheet ? ` ${styles.draftPdfLineSheetFlowLines}` : ''
                        }${showRemisePanel ? ` ${styles.draftPdfLineSheetFlowRemise}` : ''}`}
                      >
                        <div
                          className={styles.draftPdfLineSheetDoc}
                          role="dialog"
                          aria-modal="true"
                          aria-labelledby={
                            showLinesSheet
                              ? linesSheetHeadingId
                              : showRemisePanel
                                ? remiseHeadingId
                                : addLineHeadingId
                          }
                        >
                          <div className={styles.draftPdfLineSheetInner}>
                            {showLinesSheet ? (
                              linesSheetPanelContent
                            ) : showRemisePanel ? (
                              remisePanelContentCompact
                            ) : (
                              addLinePanelContent
                            )}
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : (
                inv &&
                !loading && (
                  <p className={styles.draftPdfMissing}>
                    {allowsLineEditing
                      ? 'Impossible d’afficher l’aperçu. Rechargez la facture.'
                      : 'PDF non disponible. Actualisez ou régénérez le document.'}
                  </p>
                )
              )}

              {/* Brouillon : tableau des lignes uniquement dans le panneau sous l’aperçu (bouton liste). */}
              {!allowsLineEditing && draftLinesEditorBody}

              <div className={styles.footer}>
                <div className={styles.footerGroup}>
                {isDraft && onOpenSendEmail && (
                  <button
                    type="button"
                    className={`${styles.btn} ${styles.btnPrimary}`}
                    onClick={() => onOpenSendEmail(inv || initialInvoice)}
                  >
                    <FiSend size={14} /> Envoyer
                  </button>
                )}
                {isDraft && onMarkAsSent && (
                  <button type="button" className={styles.btnMuted} onClick={() => onMarkAsSent(inv || initialInvoice)}>
                    Marquer envoyée
                  </button>
                )}
                </div>
              </div>
            </>
          )}
        </div>
  );
};

export default DraftInvoiceEditorPanel;
