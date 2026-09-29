import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link, useParams } from 'react-router-dom';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import apiClient from '../../../utils/apiClient';
import { preloadInvoicePdfPrint } from '../../../utils/invoicePdfPrint';
import { getApiErrorMessage } from '../../../utils/apiErrorMessage';
import styles from './PortalReceivables.module.css';

function formatDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('fr-CH', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
  } catch {
    return String(iso);
  }
}

function formatMoney(amount, currency = 'CHF') {
  const n = Number(amount);
  if (Number.isNaN(n)) return '—';
  return `${n.toFixed(2)} ${currency}`;
}

function statusLabel(row) {
  if (row.status === 'disputed') return 'Contestée — en cours d’examen';
  if (row.status === 'cancelled') return 'Annulée';
  if (row.status === 'paid' || Number(row.balance_due) <= 0) return 'Payée';
  if (row.is_overdue) return 'Échue';
  if (row.status === 'partially_paid') return 'Partiellement payée';
  if (row.status === 'sent') return 'Reçue';
  return 'Reçue';
}

function holdLabel(row) {
  if (row.status === 'disputed') return null;
  if (row.hold_effect === 'carrier_blocked') {
    return 'Nouvelles prestations avec ce transporteur suspendues';
  }
  return null;
}

function isPaidInvoice(row) {
  if (!row || row.status === 'cancelled' || row.status === 'disputed') return false;
  return row.status === 'paid' || Number(row.balance_due) <= 0;
}

function statusClassName(row, styles) {
  if (row.status === 'cancelled') return styles.statusCancelled;
  if (isPaidInvoice(row)) return styles.statusPaid;
  if (row.is_overdue) return styles.statusOverdue;
  return styles.statusOpen;
}

function filenameFromDisposition(header, fallback) {
  const match = /filename="?([^";]+)"?/i.exec(String(header || ''));
  const name = match?.[1]?.trim();
  return name || fallback;
}

async function readBlobError(error, fallback) {
  const data = error?.response?.data;
  if (data instanceof Blob) {
    try {
      const parsed = JSON.parse(await data.text());
      if (parsed?.message) return String(parsed.message);
    } catch {
      /* réponse non JSON */
    }
  }
  return getApiErrorMessage(error, fallback);
}

function InvoiceSheet({ invoiceId }) {
  const stackRef = useRef(null);
  const [status, setStatus] = useState('loading');
  const [message, setMessage] = useState('');

  useEffect(() => {
    const stack = stackRef.current;
    if (!invoiceId || !stack) return undefined;
    let cancelled = false;
    let pdfDoc = null;
    setStatus('loading');
    setMessage('');
    stack.replaceChildren();

    (async () => {
      const res = await apiClient.get(
        `/clients/me/portal-receivables/invoices/${invoiceId}/pdf`,
        { responseType: 'arraybuffer' }
      );
      const source = new Uint8Array(res.data);
      const pdfjs = await preloadInvoicePdfPrint();
      pdfDoc = await pdfjs.getDocument({ data: source.slice() }).promise;
      if (cancelled) return;
      const width = Math.min(stack.clientWidth || 720, 720);
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum += 1) {
        const page = await pdfDoc.getPage(pageNum);
        const base = page.getViewport({ scale: 1 });
        const viewport = page.getViewport({ scale: (width / base.width) * ratio });
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${viewport.height / ratio}px`;
        canvas.className = styles.sheet;
        const ctx = canvas.getContext('2d', { alpha: false });
        if (!ctx) throw new Error('Rendu impossible.');
        await page.render({ canvasContext: ctx, viewport }).promise;
        if (cancelled) return;
        stack.appendChild(canvas);
      }
      if (!cancelled) setStatus('ready');
    })().catch(async (err) => {
      if (cancelled) return;
      setMessage(await readBlobError(err, 'Impossible d’afficher la facture.'));
      setStatus('error');
    });

    return () => {
      cancelled = true;
      pdfDoc?.destroy?.();
      stack.replaceChildren();
    };
  }, [invoiceId]);

  return (
    <div className={styles.sheetDesk}>
      {status === 'loading' ? <p className={styles.muted}>Ouverture de la facture…</p> : null}
      {status === 'error' ? <p className={styles.error}>{message}</p> : null}
      <div ref={stackRef} className={styles.sheetStack} />
    </div>
  );
}

export default function PortalReceivablesPage() {
  const { public_id: publicId } = useParams();
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [disputeId, setDisputeId] = useState(null);
  const [reason, setReason] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [downloadingId, setDownloadingId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await apiClient.get('/clients/me/portal-receivables');
      const list = Array.isArray(res.data?.data) ? res.data.data : [];
      list.sort((a, b) => String(b.issued_at || '').localeCompare(String(a.issued_at || '')));
      setRows(list);
    } catch (err) {
      setError(getApiErrorMessage(err, 'Impossible de charger les factures.'));
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    setSelectedId((current) => {
      if (current && rows.some((row) => row.receivable_id === current)) return current;
      return null;
    });
  }, [rows]);

  useEffect(() => {
    if (!selectedId) return undefined;
    const onKey = (event) => {
      if (event.key === 'Escape') setSelectedId(null);
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', onKey);
    };
  }, [selectedId]);

  const selected = rows.find((row) => row.receivable_id === selectedId) || null;

  const downloadPdf = async (row) => {
    if (!row?.invoice_id || downloadingId) return;
    setDownloadingId(row.invoice_id);
    setError('');
    try {
      const res = await apiClient.get(
        `/clients/me/portal-receivables/invoices/${row.invoice_id}/pdf`,
        { responseType: 'blob' }
      );
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      const fallback = `${row.external_invoice_number || 'facture'}.pdf`;
      link.href = url;
      link.download = filenameFromDisposition(res.headers?.['content-disposition'], fallback);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(await readBlobError(err, 'Impossible de télécharger le PDF.'));
    } finally {
      setDownloadingId(null);
    }
  };

  const currency = rows[0]?.currency || 'CHF';
  const totalInvoiced = rows.reduce((sum, row) => sum + Number(row.total_amount || 0), 0);
  const totalPaid = rows.reduce((sum, row) => sum + Number(row.amount_paid || 0), 0);
  const balanceDue = rows.reduce((sum, row) => sum + Number(row.balance_due || 0), 0);
  const hasOverdue = rows.some((row) => row.is_overdue || holdLabel(row));

  const submitDispute = async (receivableId) => {
    const text = reason.trim();
    if (!text) {
      setError('Indiquez un motif de contestation.');
      return;
    }
    setSubmitting(true);
    setError('');
    try {
      await apiClient.post(`/clients/me/portal-receivables/${receivableId}/dispute`, {
        reason: text,
      });
      setDisputeId(null);
      setReason('');
      await load();
    } catch (err) {
      setError(getApiErrorMessage(err, 'Contestation impossible.'));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={styles.page}>
      <HeaderDashboard />
      <main className={styles.main}>
        <div className={styles.inner}>
          <header className={styles.pageHead}>
            <div>
              <p className={styles.eyebrow}>Compte client</p>
              <h1 className={styles.title}>Factures reçues</h1>
            </div>
            {publicId ? (
              <Link
                className={styles.back}
                to={`/reservations/${encodeURIComponent(publicId)}`}
              >
                Mes courses
              </Link>
            ) : null}
          </header>
          {hasOverdue ? (
            <p className={styles.lead}>
              Une facture échue suspend uniquement les nouvelles courses avec ce transporteur.
            </p>
          ) : null}
          {error ? <p className={styles.error}>{error}</p> : null}
          {loading ? <p className={styles.muted}>Chargement…</p> : null}
          {!loading && rows.length === 0 ? (
            <p className={styles.muted}>Aucune facture pour le moment.</p>
          ) : null}
          {!loading && rows.length > 0 ? (
            <dl className={styles.metrics}>
              <div>
                <dt>Factures</dt>
                <dd>{rows.length}</dd>
              </div>
              <div>
                <dt>Total reçu</dt>
                <dd>{formatMoney(totalInvoiced, currency)}</dd>
              </div>
              <div>
                <dt>Payé</dt>
                <dd>{formatMoney(totalPaid, currency)}</dd>
              </div>
              <div>
                <dt>Reste à payer</dt>
                <dd>{formatMoney(balanceDue, currency)}</dd>
              </div>
            </dl>
          ) : null}
          {!loading && rows.length > 0 ? (
            <div className={styles.ledgerWrap}>
              <table className={styles.ledger}>
                <thead>
                  <tr>
                    <th scope="col">Émission</th>
                    <th scope="col">Référence</th>
                    <th scope="col">Transporteur</th>
                    <th scope="col">Échéance</th>
                    <th scope="col">Statut</th>
                    <th scope="col" className={styles.num}>Total</th>
                    <th scope="col" className={styles.num}>Payé</th>
                    <th scope="col" className={styles.num}>Solde</th>
                    <th scope="col" className={styles.actionsHead}>
                      <span className={styles.srOnly}>Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => {
                    const isOpen = row.receivable_id === selectedId;
                    const canOpen = Boolean(row.pdf_available && row.invoice_id);
                    const paid = isPaidInvoice(row);
                    const rowClass = [paid ? styles.paidRow : '', isOpen ? styles.activeRow : '']
                      .filter(Boolean)
                      .join(' ');
                    return (
                      <tr key={row.receivable_id} className={rowClass || undefined}>
                        <td>{formatDate(row.issued_at)}</td>
                        <td className={styles.ref}>{row.external_invoice_number || '—'}</td>
                        <td>{row.creditor_company_name}</td>
                        <td>{formatDate(row.due_date)}</td>
                        <td>
                          <span className={statusClassName(row, styles)}>{statusLabel(row)}</span>
                        </td>
                        <td className={styles.num}>{formatMoney(row.total_amount, row.currency)}</td>
                        <td className={styles.num}>{formatMoney(row.amount_paid, row.currency)}</td>
                        <td className={`${styles.num} ${paid ? styles.paidBalance : ''}`}>
                          {formatMoney(row.balance_due, row.currency)}
                        </td>
                        <td className={styles.actionsCell}>
                          {canOpen ? (
                            <button
                              type="button"
                              className={styles.textBtn}
                              aria-expanded={isOpen}
                              onClick={() =>
                                setSelectedId(isOpen ? null : row.receivable_id)
                              }
                            >
                              {isOpen ? 'Fermer' : 'Voir'}
                            </button>
                          ) : (
                            <span className={styles.noPdf}>PDF indisponible</span>
                          )}
                          {canOpen ? (
                            <button
                              type="button"
                              className={styles.textBtn}
                              disabled={downloadingId === row.invoice_id}
                              onClick={() => downloadPdf(row)}
                            >
                              {downloadingId === row.invoice_id ? '…' : 'PDF'}
                            </button>
                          ) : null}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
          {!loading && selected?.invoice_id
            ? createPortal(
            <div
              className={styles.modalBackdrop}
              role="presentation"
              onClick={() => setSelectedId(null)}
            >
            <section
              className={styles.modal}
              role="dialog"
              aria-modal="true"
              aria-label={`Facture ${selected.external_invoice_number || ''}`}
              onClick={(event) => event.stopPropagation()}
            >
              <button
                type="button"
                className={styles.modalClose}
                aria-label="Fermer"
                onClick={() => setSelectedId(null)}
              >
                Fermer
              </button>
              {holdLabel(selected) ? (
                <p className={styles.holdNote}>{holdLabel(selected)}</p>
              ) : null}
              <InvoiceSheet invoiceId={selected.invoice_id} />
              {selected.can_dispute ? (
                disputeId === selected.receivable_id ? (
                  <div className={styles.disputeBox}>
                    <label htmlFor={`dispute-${selected.receivable_id}`}>
                      Motif de contestation
                    </label>
                    <textarea
                      id={`dispute-${selected.receivable_id}`}
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      rows={3}
                    />
                    <div className={styles.actions}>
                      <button
                        type="button"
                        className={styles.primary}
                        disabled={submitting}
                        onClick={() => submitDispute(selected.receivable_id)}
                      >
                        Envoyer
                      </button>
                      <button
                        type="button"
                        className={styles.ghost}
                        disabled={submitting}
                        onClick={() => {
                          setDisputeId(null);
                          setReason('');
                        }}
                      >
                        Annuler
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className={styles.ghost}
                    onClick={() => {
                      setDisputeId(selected.receivable_id);
                      setReason('');
                      setError('');
                    }}
                  >
                    Contester
                  </button>
                )
              ) : null}
            </section>
            </div>,
            document.body
          ) : null}
        </div>
      </main>
      <Footer />
    </div>
  );
}
