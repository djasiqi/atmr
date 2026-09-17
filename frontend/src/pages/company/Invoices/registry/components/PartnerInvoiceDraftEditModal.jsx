import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import { INVOICE_CATALOG, resolveInvoiceResource } from '../../../../../utils/invoiceCatalog';
import {
  forceRegeneratePartnerInvoicePdf,
  getPartnerInvoice,
  updatePartnerInvoice,
} from '../../../../../services/invoiceService';
import styles from './PartnerInvoiceDraftEditModal.module.css';

function toDateInput(value) {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 10);
  return d.toISOString().slice(0, 10);
}

function PartnerInvoiceDraftEditModal({
  open,
  initialInvoice,
  companyId,
  onClose,
  onUpdated,
}) {
  const [portalTarget, setPortalTarget] = useState(null);
  const [invoice, setInvoice] = useState(null);
  const [lines, setLines] = useState([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const resource = useMemo(
    () => resolveInvoiceResource(initialInvoice, companyId),
    [initialInvoice, companyId]
  );

  useEffect(() => {
    if (typeof document === 'undefined' || !document.body) return undefined;
    const el = document.createElement('div');
    el.setAttribute('data-portal', 'partner-invoice-draft-edit');
    document.body.appendChild(el);
    setPortalTarget(el);
    return () => {
      setPortalTarget(null);
      if (el.parentNode) el.parentNode.removeChild(el);
    };
  }, []);

  useEffect(() => {
    if (!open || resource.type !== INVOICE_CATALOG.PARTNER || !resource.id) return undefined;
    let cancelled = false;
    setError(null);
    getPartnerInvoice(resource.companyId, resource.id, { cacheBust: true })
      .then((detail) => {
        if (cancelled) return;
        setInvoice(detail);
        setLines((detail.lines || []).map((line) => ({ ...line })));
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err?.message || 'Impossible de charger la facture partenaire.');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [open, resource.type, resource.id, resource.companyId]);

  const updateLine = (index, field, value) => {
    setLines((prev) =>
      prev.map((line, i) => {
        if (i !== index) return line;
        const next = { ...line, [field]: value };
        if (field === 'quantity' || field === 'unit_price') {
          const qty = Number(field === 'quantity' ? value : next.quantity) || 0;
          const price = Number(field === 'unit_price' ? value : next.unit_price) || 0;
          next.amount = Math.round(qty * price * 100) / 100;
        }
        return next;
      })
    );
  };

  const handleSave = async ({ regenerate = false } = {}) => {
    if (!invoice) return;
    setSaving(true);
    setError(null);
    try {
      const saved = await updatePartnerInvoice(resource.companyId, resource.id, {
        notes: invoice.notes,
        recipient_name: invoice.recipient_name,
        recipient_address: invoice.recipient_address,
        recipient_contact: invoice.recipient_contact,
        period_year: Number(invoice.period_year),
        period_month: Number(invoice.period_month),
        issued_at: invoice.issued_at,
        due_date: invoice.due_date,
        lines: lines.map((line) => ({
          id: line.id,
          description: line.description,
          quantity: Number(line.quantity),
          unit_price: Number(line.unit_price),
          amount: Number(line.amount),
          note: line.note,
          service_date: line.service_date,
          departure: line.departure,
          arrival: line.arrival,
        })),
      });
      setInvoice(saved);
      setLines((saved.lines || []).map((line) => ({ ...line })));
      if (regenerate) {
        await forceRegeneratePartnerInvoicePdf(resource.companyId, resource.id);
      }
      onUpdated?.(saved);
    } catch (err) {
      setError(err?.message || 'Enregistrement partenaire impossible.');
    } finally {
      setSaving(false);
    }
  };

  if (!open || !initialInvoice || !portalTarget) return null;
  if (resource.type !== INVOICE_CATALOG.PARTNER) return null;

  return createPortal(
    <div className={styles.overlay} onClick={onClose} role="presentation">
      <div
        className={styles.panel}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Édition de la facture partenaire"
      >
        <div className={styles.head}>
          <div>
            <h2 className={styles.title}>Éditer le brouillon partenaire</h2>
            <p className={styles.subtitle}>
              {invoice?.invoice_number || initialInvoice.invoice_number}
            </p>
          </div>
          <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>
        <div className={styles.body}>
          {error ? <p className={styles.err}>{error}</p> : null}
          {!invoice ? (
            <p>Chargement de la facture partenaire…</p>
          ) : (
            <>
              <div className={styles.grid}>
                <label className={styles.label}>
                  Destinataire
                  <input
                    className={styles.input}
                    value={invoice.recipient_name || ''}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, recipient_name: e.target.value }))
                    }
                  />
                </label>
                <label className={styles.label}>
                  Contact
                  <input
                    className={styles.input}
                    value={invoice.recipient_contact || ''}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, recipient_contact: e.target.value }))
                    }
                  />
                </label>
                <label className={styles.label}>
                  Adresse de facturation
                  <textarea
                    className={styles.textarea}
                    value={invoice.recipient_address || ''}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, recipient_address: e.target.value }))
                    }
                  />
                </label>
                <label className={styles.label}>
                  Notes
                  <textarea
                    className={styles.textarea}
                    value={invoice.notes || ''}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, notes: e.target.value }))
                    }
                  />
                </label>
                <label className={styles.label}>
                  Date d’émission
                  <input
                    type="date"
                    className={styles.input}
                    value={toDateInput(invoice.issued_at)}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, issued_at: e.target.value }))
                    }
                  />
                </label>
                <label className={styles.label}>
                  Échéance
                  <input
                    type="date"
                    className={styles.input}
                    value={toDateInput(invoice.due_date)}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, due_date: e.target.value }))
                    }
                  />
                </label>
                <label className={styles.label}>
                  Année
                  <input
                    type="number"
                    className={styles.input}
                    value={invoice.period_year || ''}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, period_year: e.target.value }))
                    }
                  />
                </label>
                <label className={styles.label}>
                  Mois
                  <input
                    type="number"
                    min="1"
                    max="12"
                    className={styles.input}
                    value={invoice.period_month || ''}
                    onChange={(e) =>
                      setInvoice((prev) => ({ ...prev, period_month: e.target.value }))
                    }
                  />
                </label>
              </div>

              <div className={styles.tableWrap}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Libellé</th>
                      <th>Qté</th>
                      <th>Prix</th>
                      <th>Montant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lines.map((line, index) => (
                      <tr key={line.id || `tmp-${index}`}>
                        <td>
                          <input
                            className={styles.lineInput}
                            value={line.description || ''}
                            onChange={(e) => updateLine(index, 'description', e.target.value)}
                          />
                        </td>
                        <td>
                          <input
                            type="number"
                            step="0.05"
                            className={styles.lineInput}
                            value={line.quantity ?? 1}
                            onChange={(e) => updateLine(index, 'quantity', e.target.value)}
                          />
                        </td>
                        <td>
                          <input
                            type="number"
                            step="0.05"
                            className={styles.lineInput}
                            value={line.unit_price ?? 0}
                            onChange={(e) => updateLine(index, 'unit_price', e.target.value)}
                          />
                        </td>
                        <td>
                          <input
                            type="number"
                            step="0.05"
                            className={styles.lineInput}
                            value={line.amount ?? 0}
                            onChange={(e) => updateLine(index, 'amount', e.target.value)}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className={styles.totals}>
                <span>Total: {Number(invoice.total_amount || 0).toFixed(2)} CHF</span>
              </div>
              <div className={styles.footer}>
                <button type="button" className={styles.btn} onClick={onClose} disabled={saving}>
                  Fermer
                </button>
                <button
                  type="button"
                  className={styles.btn}
                  onClick={() => handleSave({ regenerate: true })}
                  disabled={saving}
                >
                  Enregistrer et régénérer le PDF
                </button>
                <button
                  type="button"
                  className={styles.btnPrimary}
                  onClick={() => handleSave()}
                  disabled={saving}
                >
                  Enregistrer
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>,
    portalTarget
  );
}

export default PartnerInvoiceDraftEditModal;
