import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { FiX, FiRefreshCw, FiSend, FiTrash2, FiCheck, FiPlus } from 'react-icons/fi';
import {
  getInvoice,
  invoiceService,
  formatCurrencyCHF,
  regenerateInvoicePdf,
  canRegeneratePdf,
} from '../../../../../services/invoiceService';
import { INVOICE_PRINT_SIDES, openPdfUrlWithPrintDialog } from '../../../../../utils/invoicePdfPrint';
import { ensurePdfUrlWorksInDev } from '../../../../../utils/pdfUrlFallback';
import { getApiErrorMessage } from '../../../../../utils/apiErrorMessage';
import styles from './InvoiceDraftEditModal.module.css';

const lineKey = (l) => l.id;

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

function tauxSuffixForUnit(u) {
  if (u === 'min') return 'CHF/min';
  if (u === 'h') return 'CHF/h';
  if (u === 'd') return 'CHF/j';
  if (u === 'mois') return 'CHF/mois';
  return 'CHF';
}

const InvoiceDraftEditModal = ({
  open,
  initialInvoice,
  companyId,
  onClose,
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

  const load = useCallback(async () => {
    if (!open || !companyId || !initialInvoice?.id) return;
    setLoading(true);
    setError('');
    try {
      const res = await getInvoice(companyId, initialInvoice.id);
      const data = unwrapInvoicePayload(res) ?? res?.data ?? res;
      setInv(data);
      const gd = data?.meta?.global_discount;
      if (gd && typeof gd.percent === 'number') {
        setGlobalPct(String(gd.percent));
        setGlobalNote(gd.note ? String(gd.note) : '');
      } else {
        setGlobalPct('');
        setGlobalNote('');
      }
    } catch (e) {
      setError('Impossible de charger la facture.');
    } finally {
      setLoading(false);
    }
  }, [open, companyId, initialInvoice?.id]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  useEffect(() => {
    if (!open) {
      setInv(null);
      setLocalAmounts({});
      setLocalNotes({});
      setGlobalPct('');
      setGlobalNote('');
      setCustomLineDesc('');
      setCustomLineMode(EXTRA_LINE_MODE.time);
      setCustomLineTaux('');
      setCustomLineTimeValue('1');
      setCustomLineTimeUnit('h');
      setCustomLineUnitPrice('');
      setCustomLineQty('1');
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

  if (!open || !initialInvoice) return null;

  const lines = inv?.lines || [];
  const _st = (s) => (typeof s === 'string' ? s.toLowerCase() : '');
  const isDraft = _st(inv?.status || initialInvoice?.status) === 'draft';
  const statusLabel = (s) => {
    const t = _st(s);
    if (t === 'draft') return 'Brouillon';
    if (t === 'sent') return 'Envoyée';
    if (t === 'paid') return 'Payée';
    if (t === 'cancelled' || t === 'canceled') return 'Annulée';
    return typeof s === 'string' && s ? s : '—';
  };
  const isRide = (t) => String(t || '').toLowerCase() === 'ride';
  const isCustom = (t) => String(t || '').toLowerCase() === 'custom';
  const isRemiseLine = (line) =>
    isCustom(line.type) && line.line_total != null && Number(line.line_total) < 0;

  const handleSaveAmount = async (line) => {
    if (!isDraft) return;
    const lid = line.id;
    const raw = localAmounts[lid];
    if (raw === undefined || raw === '' || Number.isNaN(parseFloat(String(raw).replace(',', '.')))) return;
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.updateDraftInvoiceLine(companyId, inv.id, lid, {
        line_total: parseFloat(String(raw).replace(',', '.')),
      });
      const data = unwrapInvoicePayload(res);
      if (data) setInv(data);
      onUpdated?.();
    } catch (e) {
      setError(e?.response?.data?.error || 'Mise à jour impossible');
    } finally {
      setSaving(false);
    }
  };

  const handleSaveNote = async (line) => {
    if (!isDraft) return;
    const note = localNotes[line.id] !== undefined ? localNotes[line.id] : line.adjustment_note;
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.updateDraftInvoiceLine(companyId, inv.id, line.id, {
        adjustment_note: note || null,
      });
      const data = unwrapInvoicePayload(res);
      if (data) setInv(data);
      onUpdated?.();
    } catch (e) {
      setError(e?.response?.data?.error || 'Note non enregistrée');
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveLine = async (line) => {
    if (!isDraft) return;
    if (
      !window.confirm(
        'Exclure ce transport de la facture ? Le montant sera retiré du brouillon et le transport redeviendra facturable.'
      )
    ) {
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.removeDraftInvoiceLine(companyId, inv.id, line.id);
      const data = unwrapInvoicePayload(res);
      if (data) setInv(data);
      onUpdated?.();
    } catch (e) {
      setError(e?.response?.data?.error || 'Suppression impossible');
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveRemise = async () => {
    if (!isDraft || !inv?.id) return;
    if (!window.confirm(
      'Retirer la remise globale ? Les montants des transports seront recalculés au catalogue (réservation) et la ligne de remise sera supprimée.'
    )) {
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.removeDraftGlobalDiscount(companyId, inv.id);
      const data = unwrapInvoicePayload(res);
      if (data) setInv(data);
      onUpdated?.();
    } catch (e) {
      setError(e?.response?.data?.error || 'Annulation de remise impossible');
    } finally {
      setSaving(false);
    }
  };

  const handleAddCustomLine = async () => {
    if (!isDraft || !inv?.id) return;
    if (!customLineDesc.trim()) {
      setError('Indiquez un libellé.');
      return;
    }
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
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.addDraftCustomLine(companyId, inv.id, {
        description: customLineDesc.trim(),
        line_total: lineTotal,
        qty,
        custom_mode: customLineMode === EXTRA_LINE_MODE.time ? 'time' : 'quantity',
        time_unit: customLineMode === EXTRA_LINE_MODE.time ? customLineTimeUnit : undefined,
      });
      const data = unwrapInvoicePayload(res);
      if (data) setInv(data);
      setCustomLineDesc('');
      setCustomLineTaux('');
      setCustomLineTimeValue('1');
      setCustomLineTimeUnit('h');
      setCustomLineUnitPrice('');
      setCustomLineQty('1');
      onUpdated?.();
    } catch (e) {
      setError(e?.response?.data?.error || 'Ligne non ajoutée');
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveCustomOrExtraLine = async (line) => {
    if (!isDraft) return;
    if (
      !window.confirm('Retirer cette ligne du brouillon ?')
    ) {
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.removeDraftInvoiceLine(companyId, inv.id, line.id);
      const data = res?.data?.invoice ?? res?.invoice ?? res?.data;
      if (data) setInv(data);
      onUpdated?.();
    } catch (e) {
      setError(e?.response?.data?.error || 'Suppression impossible');
    } finally {
      setSaving(false);
    }
  };

  const handleApplyDiscount = async () => {
    const p = parseFloat(String(globalPct).replace(',', '.'));
    if (!isDraft || !Number.isFinite(p) || p <= 0 || p > 100) {
      setError('Indiquez un pourcentage de remise entre 0 et 100.');
      return;
    }
    setSaving(true);
    setError('');
    try {
      const res = await invoiceService.applyDraftGlobalDiscount(companyId, inv.id, {
        global_discount_percent: p,
        global_discount_note: globalNote || null,
      });
      const data = unwrapInvoicePayload(res);
      if (data) setInv(data);
      onUpdated?.();
    } catch (e) {
      setError(e?.response?.data?.error || 'Remise non appliquée');
    } finally {
      setSaving(false);
    }
  };

  /** Régénère si possible, puis ouvre le PDF et la boîte d’impression (recto uniquement par défaut). */
  const handlePdfPrint = async (
    printSides = INVOICE_PRINT_SIDES.SIMPLEX
  ) => {
    if (!inv?.id) return;
    setSaving(true);
    setError('');
    try {
      let detail = inv;
      /** Réponse regenerate-pdf contient `pdf_url` ; le GET détail peut l’omettre selon le sérialiseur. */
      let pdfUrl =
        typeof inv.pdf_url === 'string' && inv.pdf_url.trim()
          ? inv.pdf_url.trim()
          : '';

      if (canRegeneratePdf(inv)) {
        const regen = await regenerateInvoicePdf(companyId, inv.id);
        const fromRegen =
          typeof regen?.pdf_url === 'string' ? regen.pdf_url.trim() : '';
        const res = await getInvoice(companyId, inv.id);
        detail = unwrapInvoicePayload(res) ?? res?.data ?? res ?? inv;
        const fromDetail =
          typeof detail?.pdf_url === 'string' ? detail.pdf_url.trim() : '';
        pdfUrl = fromRegen || fromDetail;
        setInv({ ...detail, ...(pdfUrl ? { pdf_url: pdfUrl } : {}) });
      } else if (!pdfUrl && typeof detail?.pdf_url === 'string') {
        pdfUrl = detail.pdf_url.trim();
      }

      if (!pdfUrl) {
        setError('Aucun PDF disponible.');
        return;
      }

      const fixedUrl = ensurePdfUrlWorksInDev(pdfUrl);
      const opened = openPdfUrlWithPrintDialog(fixedUrl, { printSides });
      if (!opened) {
        setError(
          'PDF prêt, mais impossible d’ouvrir la fenêtre d’impression (pop-ups bloquées). Autorisez les pop-ups pour ce site ou ouvrez le PDF depuis la liste des factures.'
        );
      }
      onUpdated?.();
    } catch (e) {
      setError(getApiErrorMessage(e, 'PDF non disponible'));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.panel} onClick={(e) => e.stopPropagation()}>
        <div className={styles.head}>
          <div className={styles.headText}>
            <h2 className={styles.title}>Brouillon</h2>
            <p className={styles.subtitle}>
              {inv?.invoice_number || initialInvoice?.invoice_number || '—'}
            </p>
          </div>
          <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>
        <div className={styles.body}>
          {loading && <p className={styles.muted}>Chargement…</p>}
          {error && <p className={styles.err}>{error}</p>}

          {!loading && inv && (
            <>
              <div className={styles.summaryBar}>
                <span
                  className={isDraft ? styles.badgeDraft : styles.badgeStatus}
                >
                  {statusLabel(inv.status)}
                </span>
                <div className={styles.summaryTotal}>
                  <span className={styles.summaryLabel}>TTC</span>
                  <span className={styles.summaryAmount}>
                    {draftTotalTtc.toFixed(2)} CHF
                  </span>
                </div>
              </div>

              <div className={styles.tableScroll}>
                <table className={styles.table}>
                <colgroup>
                  <col className={styles.colDesc} />
                  <col className={styles.colHt} />
                  <col className={styles.colNoteCol} />
                  <col className={styles.colActions} />
                </colgroup>
                <thead>
                  <tr>
                    <th className={styles.colDescCell}>Description</th>
                    <th className={styles.colHtCell}>HT</th>
                    <th className={styles.colNote}>Note</th>
                    <th className={styles.colActionsCell} scope="col">
                      <span className={styles.srOnly}>Action</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {lines.map((line) => {
                    const remiseNeg = isRemiseLine(line);
                    const customPos = isCustom(line.type) && !remiseNeg;
                    return (
                    <tr key={lineKey(line)}>
                      <td className={styles.colDescCell}>
                        <div>{line.description || line.type}</div>
                        <div className={styles.muted}>#{line.id}</div>
                        {remiseNeg && (
                          <span className={styles.remiseHint} title="Utilisez le bouton Retirer la remise.">
                            Remise globale
                          </span>
                        )}
                      </td>
                      <td className={styles.colHtCell}>
                        {isDraft && (isRide(line.type) || customPos) ? (
                          <div className={styles.cellEditRow}>
                            <input
                              className={styles.inputSm}
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
                            />
                            <div className={styles.rowActions}>
                              <button
                                type="button"
                                className={styles.btnIconOk}
                                disabled={saving}
                                title="Valider le montant"
                                onClick={() => handleSaveAmount(line)}
                              >
                                <FiCheck size={14} />
                              </button>
                            </div>
                          </div>
                        ) : (
                          formatCurrencyCHF(line.line_total)
                        )}
                      </td>
                      <td className={styles.colNote}>
                        {isDraft && (line.type === 'ride' || customPos) && !remiseNeg ? (
                          <div className={styles.cellEditRow}>
                            <textarea
                              className={styles.textarea}
                              rows={1}
                              defaultValue={line.adjustment_note || ''}
                              onChange={(e) =>
                                setLocalNotes((prev) => ({
                                  ...prev,
                                  [line.id]: e.target.value,
                                }))
                              }
                            />
                            <div className={styles.rowActions}>
                              <button
                                type="button"
                                className={styles.btnIconOk}
                                disabled={saving}
                                title="Enregistrer la note"
                                onClick={() => handleSaveNote(line)}
                              >
                                <FiCheck size={14} />
                              </button>
                            </div>
                          </div>
                        ) : (
                          <span className={styles.muted}>
                            {line.adjustment_note || '—'}
                          </span>
                        )}
                      </td>
                      <td className={styles.colActionsCell}>
                        <div className={styles.colActionsCellInner}>
                          {isDraft && isRide(line.type) && line.reservation_id && (
                            <button
                              type="button"
                              className={`${styles.btnIcon} ${styles.danger}`}
                              disabled={saving}
                              title="Exclure ce transport"
                              onClick={() => handleRemoveLine(line)}
                            >
                              <FiTrash2 size={14} />
                            </button>
                          )}
                          {isDraft && customPos && (
                            <button
                              type="button"
                              className={`${styles.btnIcon} ${styles.danger}`}
                              disabled={saving}
                              title="Retirer cette ligne"
                              onClick={() => handleRemoveCustomOrExtraLine(line)}
                            >
                              <FiTrash2 size={14} />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );})}
                </tbody>
                </table>
              </div>

              {isDraft && (
                <div className={styles.section}>
                  <h3 className={styles.sectionTitle}>Remise globale</h3>
                  <div className={`${styles.formRow} ${styles.formRowHarmonized}`}>
                    <label className={styles.srOnly} htmlFor="draft-gd-pct">Pourcentage</label>
                    <input
                      id="draft-gd-pct"
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
                      disabled={saving}
                      onClick={handleApplyDiscount}
                    >
                      Appliquer
                    </button>
                    <button
                      type="button"
                      className={styles.btnMuted}
                      disabled={saving}
                      onClick={handleRemoveRemise}
                    >
                      Retirer
                    </button>
                  </div>
                </div>
              )}

              {isDraft && (
                <div
                  className={styles.sectionCompact}
                  title="Prix unitaire × quantité, ou taux × durée (unité) selon le mode."
                >
                  <div className={styles.sectionHeaderRow}>
                    <h3 className={styles.sectionTitleInline}>Ligne suppl. HT</h3>
                    <div
                      className={styles.modeSegSm}
                      role="group"
                      aria-label="Type de facturation"
                    >
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
                  <div className={styles.extraFormCompact}>
                    <input
                      className={styles.inputLibelle}
                      type="text"
                      placeholder="Libellé"
                      value={customLineDesc}
                      onChange={(e) => setCustomLineDesc(e.target.value)}
                    />

                    {customLineMode === EXTRA_LINE_MODE.time ? (
                      <div className={styles.compactToolbar}>
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
                          onClick={handleAddCustomLine}
                        >
                          <FiPlus size={18} />
                        </button>
                      </div>
                    ) : (
                      <div className={styles.compactToolbar}>
                        <input
                          id="x-pu"
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
                          id="x-qt"
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
                          onClick={handleAddCustomLine}
                        >
                          <FiPlus size={18} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className={styles.footer}>
                <div className={styles.footerGroup}>
                <button
                  type="button"
                  className={styles.btn}
                  disabled={saving}
                  title="Régénérer le PDF si besoin, puis imprimer (recto par défaut)"
                  onClick={() => handlePdfPrint(INVOICE_PRINT_SIDES.SIMPLEX)}
                >
                  <FiRefreshCw size={14} /> PDF
                </button>
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
      </div>
    </div>
  );
};

export default InvoiceDraftEditModal;
