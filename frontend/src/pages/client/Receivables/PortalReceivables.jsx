import React, { useCallback, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import apiClient from '../../../utils/apiClient';
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
  return 'Émise';
}

function holdLabel(row) {
  if (row.status === 'disputed') return null;
  if (row.hold_effect === 'carrier_blocked') {
    return 'Nouvelles prestations avec ce transporteur suspendues';
  }
  return null;
}

export default function PortalReceivablesPage() {
  const { public_id: publicId } = useParams();
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [disputeId, setDisputeId] = useState(null);
  const [reason, setReason] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await apiClient.get('/clients/me/portal-receivables');
      setRows(Array.isArray(res.data?.data) ? res.data.data : []);
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
          <p className={styles.eyebrow}>Compte client</p>
          <h1 className={styles.title}>Factures transporteurs</h1>
          <p className={styles.lead}>
            Factures émises par les transporteurs pour vos courses. Une facture
            échue suspend uniquement les nouvelles prestations avec ce
            transporteur — pas votre compte Lirie.
          </p>
          {publicId ? (
            <p className={styles.navBack}>
              <Link to={`/reservations/${encodeURIComponent(publicId)}`}>
                ← Mes courses
              </Link>
            </p>
          ) : null}
          {error ? <p className={styles.error}>{error}</p> : null}
          {loading ? <p className={styles.muted}>Chargement…</p> : null}
          {!loading && rows.length === 0 ? (
            <p className={styles.muted}>Aucune facture pour le moment.</p>
          ) : null}
          <ul className={styles.list}>
            {rows.map((row) => {
              const hold = holdLabel(row);
              return (
                <li key={row.receivable_id} className={styles.card}>
                  <div className={styles.cardHead}>
                    <strong>{row.creditor_company_name}</strong>
                    <span>{statusLabel(row)}</span>
                  </div>
                  <dl className={styles.meta}>
                    <div>
                      <dt>Référence</dt>
                      <dd>{row.external_invoice_number}</dd>
                    </div>
                    <div>
                      <dt>Émission</dt>
                      <dd>{formatDate(row.issued_at)}</dd>
                    </div>
                    <div>
                      <dt>Échéance</dt>
                      <dd>{formatDate(row.due_date)}</dd>
                    </div>
                    <div>
                      <dt>Total</dt>
                      <dd>{formatMoney(row.total_amount, row.currency)}</dd>
                    </div>
                    <div>
                      <dt>Payé</dt>
                      <dd>{formatMoney(row.amount_paid, row.currency)}</dd>
                    </div>
                    <div>
                      <dt>Solde</dt>
                      <dd>{formatMoney(row.balance_due, row.currency)}</dd>
                    </div>
                  </dl>
                  {hold ? <p className={styles.holdNote}>{hold}</p> : null}
                  {row.can_dispute ? (
                    disputeId === row.receivable_id ? (
                      <div className={styles.disputeBox}>
                        <label htmlFor={`dispute-${row.receivable_id}`}>
                          Motif de contestation
                        </label>
                        <textarea
                          id={`dispute-${row.receivable_id}`}
                          value={reason}
                          onChange={(e) => setReason(e.target.value)}
                          rows={3}
                        />
                        <div className={styles.actions}>
                          <button
                            type="button"
                            className={styles.primary}
                            disabled={submitting}
                            onClick={() => submitDispute(row.receivable_id)}
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
                          setDisputeId(row.receivable_id);
                          setReason('');
                          setError('');
                        }}
                      >
                        Contester
                      </button>
                    )
                  ) : null}
                </li>
              );
            })}
          </ul>
        </div>
      </main>
      <Footer />
    </div>
  );
}
