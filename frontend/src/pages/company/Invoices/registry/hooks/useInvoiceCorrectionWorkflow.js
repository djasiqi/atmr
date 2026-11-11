/**
 * Workflow d'avoir + rectificative (API existante).
 * Aucune mutation de la facture originale.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { invoiceService } from '../../../../../services/invoiceService';
import { getApiErrorMessage } from '../../../../../utils/apiErrorMessage';
import { buildCreditNotePdfApiUrl } from '../../../../../utils/pdfUrlFallback';
import { openProtectedPdfInNewTab } from '../../../../../utils/protectedPdf';

export function money(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.round(n * 100) / 100;
}

export function newIdempotencyKey(prefix, invoiceId) {
  const rand =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID().slice(0, 8)
      : String(Date.now());
  return `${prefix}-${invoiceId}-${Date.now()}-${rand}`.slice(0, 80);
}

/**
 * @param {{ open: boolean, companyId: number, invoiceId: number, invoice?: object }} args
 */
export default function useInvoiceCorrectionWorkflow({
  open,
  companyId,
  invoiceId,
  invoice,
}) {
  const [detail, setDetail] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [residuals, setResiduals] = useState([]);
  const [selected, setSelected] = useState({});
  const [partialAmounts, setPartialAmounts] = useState({});
  const [targetAmounts, setTargetAmounts] = useState({});
  const [serverPreview, setServerPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [reason, setReason] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [prepareCorrective, setPrepareCorrective] = useState(true);
  const [idempotencyKey, setIdempotencyKey] = useState('');
  const [correctiveKey, setCorrectiveKey] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const cid = Number(companyId || invoice?.company_id || 0);
  const iid = Number(invoiceId || invoice?.id || 0);

  const loadDetail = useCallback(async () => {
    if (!cid || !iid) return;
    setLoadingDetail(true);
    setError(null);
    try {
      const raw = await invoiceService.getInvoice(cid, iid, { cacheBust: true });
      const data = raw?.data || raw;
      setDetail(data);
      const preview = await invoiceService.previewCreditNote(cid, iid, { lines: [] });
      const previewData = preview?.data || preview;
      setResiduals(Array.isArray(previewData?.line_residuals) ? previewData.line_residuals : []);
      setSelected({});
      setPartialAmounts({});
      setTargetAmounts({});
      setServerPreview(null);
    } catch (err) {
      setError(getApiErrorMessage(err, 'Impossible de charger la facture.'));
      setDetail(invoice || null);
      setResiduals([]);
    } finally {
      setLoadingDetail(false);
    }
  }, [cid, iid, invoice]);

  useEffect(() => {
    if (!open) return undefined;
    setResult(null);
    setConfirmed(false);
    setPrepareCorrective(true);
    setReason('');
    setError(null);
    setIdempotencyKey(newIdempotencyKey('cn', iid));
    setCorrectiveKey(newIdempotencyKey('corr', iid));
    loadDetail();
    return undefined;
  }, [open, iid, loadDetail]);

  const selectionPayload = useMemo(() => {
    return residuals
      .filter((row) => selected[row.invoice_line_id] && money(row.residual_ttc) > 0)
      .map((row) => {
        const id = row.invoice_line_id;
        const payload = { invoice_line_id: id };
        const rawPartial = partialAmounts[id];
        if (rawPartial !== undefined && rawPartial !== '' && rawPartial !== null) {
          payload.amount_ttc = String(rawPartial);
        }
        return payload;
      });
  }, [residuals, selected, partialAmounts]);

  useEffect(() => {
    if (!open || !cid || !iid || result) return undefined;
    if (selectionPayload.length === 0) {
      setServerPreview(null);
      return undefined;
    }
    let cancelled = false;
    const timer = setTimeout(async () => {
      setPreviewLoading(true);
      try {
        const preview = await invoiceService.previewCreditNote(cid, iid, {
          lines: selectionPayload,
        });
        if (cancelled) return;
        setServerPreview(preview?.data || preview);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setServerPreview(null);
        setError(getApiErrorMessage(err, 'Aperçu serveur indisponible.'));
      } finally {
        if (!cancelled) setPreviewLoading(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [open, cid, iid, selectionPayload, result]);

  const snapshot = useMemo(() => {
    const src = detail || invoice || {};
    const total = money(src.total_amount ?? src.total_ttc);
    const paid = money(src.amount_paid);
    const notes = Array.isArray(src.credit_notes) ? src.credit_notes : [];
    const credited = notes.reduce((sum, note) => sum + money(note.amount_chf), 0);
    const room = Math.max(0, Math.round((total - credited) * 100) / 100);
    const correctedBy = Array.isArray(src.corrected_by_invoice_ids)
      ? src.corrected_by_invoice_ids
      : [];
    return {
      number: src.invoice_number || `#${src.id || '?'}`,
      status: src.status,
      total,
      paid,
      credited,
      room,
      balance: money(src.balance_due),
      notes,
      correctedBy,
      correctsInvoiceId: src.corrects_invoice_id || null,
      lines: Array.isArray(src.lines) ? src.lines : [],
    };
  }, [detail, invoice]);

  const previewAmount = money(serverPreview?.proposed_amount_ttc);
  const previewHt = money(serverPreview?.proposed_amount_ht);
  const previewVat = money(serverPreview?.proposed_amount_vat);
  const previewNet = money(serverPreview?.net_invoiced_total);
  const previewBalance = money(serverPreview?.balance_due);
  const previewOverpay = money(serverPreview?.overpayment_amount);

  const canSubmit =
    !submitting &&
    !result &&
    confirmed &&
    selectionPayload.length > 0 &&
    previewAmount > 0 &&
    String(reason || '').trim().length > 0 &&
    idempotencyKey &&
    !previewLoading &&
    Boolean(serverPreview);

  const toggleLine = useCallback((lineId) => {
    setSelected((prev) => {
      const next = { ...prev, [lineId]: !prev[lineId] };
      if (!next[lineId]) {
        setPartialAmounts((amounts) => {
          const copy = { ...amounts };
          delete copy[lineId];
          return copy;
        });
        setTargetAmounts((amounts) => {
          const copy = { ...amounts };
          delete copy[lineId];
          return copy;
        });
      }
      return next;
    });
  }, []);

  const selectAllResiduals = useCallback(() => {
    const next = {};
    residuals.forEach((row) => {
      if (money(row.residual_ttc) > 0) next[row.invoice_line_id] = true;
    });
    setSelected(next);
    setPartialAmounts({});
    setTargetAmounts({});
  }, [residuals]);

  /** Montant cible TTC après correction → crédit = résiduel − cible (plafonné). */
  const setTargetTtc = useCallback(
    (lineId, rawValue) => {
      setTargetAmounts((prev) => ({ ...prev, [lineId]: rawValue }));
      const row = residuals.find((r) => r.invoice_line_id === lineId);
      if (!row) return;
      const residual = money(row.residual_ttc);
      const target = money(rawValue);
      if (!Number.isFinite(target) || String(rawValue).trim() === '') {
        setPartialAmounts((prev) => {
          const copy = { ...prev };
          delete copy[lineId];
          return copy;
        });
        return;
      }
      const credit = Math.max(0, Math.round((residual - target) * 100) / 100);
      setSelected((prev) => ({ ...prev, [lineId]: credit > 0 || target === 0 }));
      if (credit <= 0) {
        setPartialAmounts((prev) => {
          const copy = { ...prev };
          delete copy[lineId];
          return copy;
        });
        return;
      }
      setPartialAmounts((prev) => ({
        ...prev,
        [lineId]: credit >= residual ? '' : String(credit),
      }));
    },
    [residuals]
  );

  const markLineRemoved = useCallback(
    (lineId) => {
      const row = residuals.find((r) => r.invoice_line_id === lineId);
      if (!row || money(row.residual_ttc) <= 0) return;
      setSelected((prev) => ({ ...prev, [lineId]: true }));
      setTargetAmounts((prev) => ({ ...prev, [lineId]: '0' }));
      setPartialAmounts((prev) => {
        const copy = { ...prev };
        delete copy[lineId];
        return copy;
      });
    },
    [residuals]
  );

  const submit = useCallback(async () => {
    if (!canSubmit) return null;
    setSubmitting(true);
    setError(null);
    try {
      const response = await invoiceService.issueCreditNote(cid, iid, {
        reason: String(reason).trim(),
        idempotency_key: idempotencyKey,
        lines: selectionPayload,
        prepare_corrective: prepareCorrective,
        corrective_idempotency_key: prepareCorrective ? correctiveKey : undefined,
      });
      const data = response?.data || response;
      setResult(data);
      return data;
    } catch (err) {
      setError(getApiErrorMessage(err, "Émission de l'avoir refusée."));
      return null;
    } finally {
      setSubmitting(false);
    }
  }, [
    canSubmit,
    cid,
    iid,
    reason,
    idempotencyKey,
    selectionPayload,
    prepareCorrective,
    correctiveKey,
  ]);

  const openCreditPdf = useCallback(
    async (noteLike) => {
      const noteId = noteLike?.credit_note_id || noteLike?.id;
      if (!noteId) return;
      const apiUrl = buildCreditNotePdfApiUrl(
        { id: iid, company_id: cid },
        { id: noteId }
      );
      await openProtectedPdfInNewTab(apiUrl, noteLike?.pdf_url || null, {
        filename: `${noteLike?.credit_number || `avoir-${noteId}`}.pdf`,
      });
    },
    [cid, iid]
  );

  return {
    detail,
    loadingDetail,
    residuals,
    selected,
    partialAmounts,
    targetAmounts,
    serverPreview,
    previewLoading,
    reason,
    setReason,
    confirmed,
    setConfirmed,
    prepareCorrective,
    setPrepareCorrective,
    submitting,
    error,
    result,
    snapshot,
    previewAmount,
    previewHt,
    previewVat,
    previewNet,
    previewBalance,
    previewOverpay,
    canSubmit,
    selectionPayload,
    toggleLine,
    selectAllResiduals,
    setTargetTtc,
    markLineRemoved,
    setPartialAmounts,
    submit,
    openCreditPdf,
    reload: loadDetail,
  };
}
