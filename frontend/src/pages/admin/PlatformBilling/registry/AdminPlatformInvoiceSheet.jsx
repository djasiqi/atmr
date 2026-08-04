import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { FiX } from 'react-icons/fi';
import {
  fetchPlatformBillingInvoice,
  fetchPlatformIssuedInvoice,
  issuePlatformBillingInvoice,
  sendPlatformIssuedInvoice,
  payPlatformIssuedInvoice,
  reversePlatformIssuedPayment,
  updatePlatformIssuedDueDate,
  cancelPlatformIssuedInvoice,
  createPlatformIssuedCreditNote,
  downloadPlatformIssuedInvoicePdf,
  validatePlatformBillingInvoice,
} from '../../../../services/adminService';
import AdminActionDialog from '../../components/AdminActionDialog';
import AdminPlatformInvoiceEditor from './AdminPlatformInvoiceEditor';
import {
  STATUS_LABELS,
  statusBadgeClass,
  fmtMoney,
  fmtDate,
  fmtDateTime,
  displayInvoiceLineLabel,
} from '../issuedInvoiceUi';
import {
  OPERATIONAL_STATUS_LABELS,
  operationalBadgeClass,
} from '../dossierInvoiceUi';
import styles from './AdminPlatformInvoiceSheet.module.css';

const PAYMENT_METHODS = [
  { value: 'bank_transfer', label: 'Virement bancaire' },
  { value: 'qr_bill', label: 'QR-facture' },
  { value: 'cash', label: 'Espèces' },
  { value: 'other', label: 'Autre' },
];

const TABS = [
  { id: 'overview', label: 'Résumé' },
  { id: 'detail', label: 'Lignes' },
  { id: 'payments', label: 'Paiements' },
  { id: 'history', label: 'Historique' },
];

const timelineTypeClass = (type, styleMap) => {
  switch (type) {
    case 'CANCELLED':
    case 'PAYMENT_REVERSAL':
      return styleMap.tlDanger;
    case 'CREDITED':
    case 'REPLACED_BY':
    case 'REPLACES':
    case 'CREDIT_OF':
      return styleMap.tlWarn;
    case 'PAYMENT':
    case 'SENT':
      return styleMap.tlOk;
    case 'DUE_CHANGED':
    case 'DUE_SET':
      return styleMap.tlMuted;
    default:
      return undefined;
  }
};

const apiErrorMessage = (e) =>
  e?.response?.data?.message ||
  e?.response?.data?.error ||
  e?.message ||
  'Erreur';

const todayIso = () => new Date().toISOString().slice(0, 10);

/** Date formulaire (YYYY-MM-DD) → ISO avec l’heure locale actuelle ce jour-là. */
const paidAtIsoFromDateInput = (dateStr) => {
  const now = new Date();
  if (!dateStr) return now.toISOString();
  const parts = String(dateStr).split('-').map(Number);
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) {
    return now.toISOString();
  }
  const [y, m, d] = parts;
  const local = new Date(
    y,
    m - 1,
    d,
    now.getHours(),
    now.getMinutes(),
    now.getSeconds()
  );
  return local.toISOString();
};

const AdminPlatformInvoiceSheet = ({
  issuedId: issuedIdProp,
  statementId: statementIdProp,
  dossierKey,
  dossierRow,
  onClose,
  onChanged,
  capabilities,
}) => {
  const {
    canBillingSend,
    canBillingPayment,
    canBillingDueDate,
    canBillingCancel,
    canBillingCredit,
  } = capabilities || {};

  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState('overview');
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);
  const [busy, setBusy] = useState(false);
  const [actionDialog, setActionDialog] = useState(null);
  const [paymentForm, setPaymentForm] = useState(null);
  const [dueDateForm, setDueDateForm] = useState(null);
  const [showEditor, setShowEditor] = useState(false);
  /** Garde l’id facture si le parent perd la ligne (filtre liste). */
  const [lockedIssuedId, setLockedIssuedId] = useState(null);
  const [lockedStatementId, setLockedStatementId] = useState(null);

  useEffect(() => {
    const nextIssued = issuedIdProp || dossierRow?.primary_invoice_id || null;
    const nextStatement = statementIdProp || dossierRow?.statement_id || null;
    if (nextIssued) setLockedIssuedId(nextIssued);
    if (nextStatement) setLockedStatementId(nextStatement);
  }, [
    issuedIdProp,
    statementIdProp,
    dossierRow?.primary_invoice_id,
    dossierRow?.statement_id,
  ]);

  useEffect(() => {
    setLockedIssuedId(null);
    setLockedStatementId(null);
  }, [dossierKey]);

  const issuedId =
    issuedIdProp || dossierRow?.primary_invoice_id || lockedIssuedId || null;
  const statementId =
    statementIdProp || dossierRow?.statement_id || lockedStatementId || null;
  const opStatus = dossierRow?.operational_status;
  const allowed = new Set(dossierRow?.allowed_actions || []);

  const detailQuery = useQuery({
    queryKey: ['admin', 'platform-issued-invoice', issuedId],
    queryFn: () => fetchPlatformIssuedInvoice(issuedId),
    enabled: Boolean(issuedId),
  });

  const statementQuery = useQuery({
    queryKey: ['admin', 'platform-billing-invoice', statementId],
    queryFn: () => fetchPlatformBillingInvoice(statementId),
    enabled: Boolean(statementId) && !issuedId,
  });

  const detail = detailQuery.data;
  const statement = statementQuery.data;

  const refreshAll = async () => {
    if (issuedId) {
      await queryClient.invalidateQueries({
        queryKey: ['admin', 'platform-issued-invoice', issuedId],
      });
    }
    if (statementId) {
      await queryClient.invalidateQueries({
        queryKey: ['admin', 'platform-billing-invoice', statementId],
      });
    }
    onChanged?.();
  };

  const flags = useMemo(() => {
    const uiStatus = detail?.ui_status;
    const isIssuedNotSent = uiStatus === 'ISSUED' && !detail?.sent_at;
    const isActiveSent = ['SENT', 'OVERDUE', 'PARTIALLY_PAID'].includes(uiStatus);
    const amountPaidZero = Number(detail?.amount_paid || 0) === 0;
    return {
      uiStatus,
      isIssuedNotSent,
      isActiveSent,
      isReadOnly: ['CANCELLED', 'CREDITED'].includes(uiStatus),
      canCreditNote: isActiveSent && amountPaidZero,
    };
  }, [detail]);

  const runAction = async (fn, { successMessage } = {}) => {
    setBusy(true);
    setError(null);
    setInfo(null);
    try {
      await fn();
      if (successMessage) setInfo(successMessage);
      await refreshAll();
    } catch (e) {
      setError(apiErrorMessage(e));
      throw e;
    } finally {
      setBusy(false);
    }
  };

  const handleSend = () => {
    setActionDialog({
      title: 'Marquer la facture comme envoyée',
      description: `Confirmer l'envoi de ${detail?.invoice_number || 'cette facture'} ?`,
      confirmationLabel: 'Marquer comme envoyée',
      onConfirm: async () => {
        await runAction(() => sendPlatformIssuedInvoice(issuedId), {
          successMessage: 'Facture marquée comme envoyée.',
        });
        setActionDialog(null);
      },
    });
  };

  const handleCancel = () => {
    setActionDialog({
      title: 'Annuler la facture',
      description: 'Réservé aux factures non envoyées et sans paiement.',
      confirmationLabel: 'Annuler la facture',
      danger: true,
      onConfirm: async () => {
        await runAction(() => cancelPlatformIssuedInvoice(issuedId), {
          successMessage: 'Facture annulée.',
        });
        setActionDialog(null);
      },
    });
  };

  const handleCreditNote = () => {
    setActionDialog({
      title: 'Émettre une note de crédit',
      description: `Créer un avoir total pour ${detail?.invoice_number || ''} ?`,
      confirmationLabel: 'Émettre l’avoir',
      danger: true,
      reason: { required: true, label: "Motif de l'avoir", minLength: 3 },
      onConfirm: async ({ reason }) => {
        await runAction(() => createPlatformIssuedCreditNote(issuedId, { reason }), {
          successMessage: 'Note de crédit émise.',
        });
        setActionDialog(null);
      },
    });
  };

  const submitPayment = async (e) => {
    e.preventDefault();
    const amount = Number(String(paymentForm.amount).replace(',', '.'));
    if (!Number.isFinite(amount) || amount <= 0) {
      setError('Montant invalide.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const idempotencyKey =
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : `pay-${issuedId}-${Date.now()}`;
      await payPlatformIssuedInvoice(issuedId, {
        amount,
        paid_at: paidAtIsoFromDateInput(paymentForm.paid_at),
        method: paymentForm.method || undefined,
        reference: paymentForm.reference?.trim() || undefined,
        notes: paymentForm.notes?.trim() || undefined,
        idempotency_key: idempotencyKey,
      });
      setInfo('Paiement enregistré.');
      setPaymentForm(null);
      await refreshAll();
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const submitDueDate = async (e) => {
    e.preventDefault();
    if (!dueDateForm.due_at || (dueDateForm.reason || '').trim().length < 3) {
      setError('Date et motif (3 car. min.) requis.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await updatePlatformIssuedDueDate(issuedId, {
        due_at: dueDateForm.due_at,
        reason: dueDateForm.reason.trim(),
      });
      setInfo('Échéance mise à jour.');
      setDueDateForm(null);
      await refreshAll();
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const titleNumber =
    detail?.invoice_number ||
    (issuedId ? `Facture #${issuedId}` : null) ||
    dossierKey ||
    'Dossier';
  const companyLabel =
    detail?.company_name || dossierRow?.company_name || statement?.company_name || '—';
  const periodLabel = dossierRow?.period_label ? ` · ${dossierRow.period_label}` : '';

  const amountDisplay = fmtMoney(
    detail?.total_amount ?? dossierRow?.amount ?? statement?.total_amount
  );
  const statusBadge = opStatus ? (
    <span className={`${styles.badge} ${operationalBadgeClass(opStatus, styles)}`}>
      {OPERATIONAL_STATUS_LABELS[opStatus] || opStatus}
    </span>
  ) : detail?.ui_status ? (
    <span className={`${styles.badge} ${statusBadgeClass(detail.ui_status, styles)}`}>
      {STATUS_LABELS[detail.ui_status] || detail.ui_status}
    </span>
  ) : null;

  const openPaymentForm = () => {
    setActiveTab('payments');
    setPaymentForm({
      amount: detail?.balance_due || '',
      paid_at: todayIso(),
      method: 'bank_transfer',
      reference: '',
      notes: '',
    });
  };

  const canEdit =
    issuedId &&
    (allowed.has('EDIT_INVOICE') || allowed.has('CORRECT_INVOICE')) &&
    Number(detail?.amount_paid || 0) <= 0 &&
    !flags.isReadOnly;
  const primaryKey = dossierRow?.primary_action || null;
  const primaryIsWorkflow = ['MARK_SENT', 'ISSUE', 'REVIEW', 'RECORD_PAYMENT'].includes(
    primaryKey
  );

  const primaryBtn = (() => {
    if (primaryKey === 'MARK_SENT' && issuedId) {
      return (
        <button type="button" className={styles.btnPrimary} disabled={busy} onClick={handleSend}>
          Envoyer
        </button>
      );
    }
    if (primaryKey === 'ISSUE' && statementId) {
      return (
        <button
          type="button"
          className={styles.btnPrimary}
          disabled={busy}
          onClick={() =>
            runAction(() => issuePlatformBillingInvoice(statementId), {
              successMessage: 'Facture émise.',
            })
          }
        >
          Émettre
        </button>
      );
    }
    if (primaryKey === 'REVIEW' && statementId) {
      return (
        <button
          type="button"
          className={styles.btnPrimary}
          disabled={busy}
          onClick={() =>
            runAction(() => validatePlatformBillingInvoice(statementId), {
              successMessage: 'Relevé validé.',
            })
          }
        >
          Valider
        </button>
      );
    }
    if (primaryKey === 'RECORD_PAYMENT' && issuedId) {
      return (
        <button
          type="button"
          className={styles.btnPrimary}
          disabled={busy}
          onClick={openPaymentForm}
        >
          Paiement
        </button>
      );
    }
    if (canEdit) {
      return (
        <button
          type="button"
          className={styles.btnPrimary}
          disabled={busy}
          onClick={() => setShowEditor(true)}
        >
          {allowed.has('CORRECT_INVOICE') && !allowed.has('EDIT_INVOICE')
            ? 'Corriger'
            : 'Éditer'}
        </button>
      );
    }
    return null;
  })();

  const actionBar = (
    <div className={styles.actionBar}>
      <div className={styles.actionBarStart}>
        {issuedId && (canBillingCancel || allowed.has('CANCEL')) && flags.isIssuedNotSent && (
          <button
            type="button"
            className={styles.btnDangerGhost}
            disabled={busy}
            onClick={handleCancel}
          >
            Annuler
          </button>
        )}
        {issuedId && (canBillingCredit || allowed.has('CREDIT')) && flags.canCreditNote && (
          <button type="button" className={styles.btnGhost} disabled={busy} onClick={handleCreditNote}>
            Avoir
          </button>
        )}
      </div>
      <div className={styles.actionBarEnd}>
        {issuedId && (
          <button
            type="button"
            className={styles.btn}
            disabled={busy}
            onClick={() =>
              downloadPlatformIssuedInvoicePdf(issuedId).catch((e) =>
                setError(apiErrorMessage(e))
              )
            }
          >
            PDF
          </button>
        )}
        {canEdit && primaryIsWorkflow && (
          <button
            type="button"
            className={styles.btn}
            disabled={busy}
            onClick={() => setShowEditor(true)}
          >
            {allowed.has('CORRECT_INVOICE') && !allowed.has('EDIT_INVOICE')
              ? 'Corriger'
              : 'Éditer'}
          </button>
        )}
        {issuedId && (canBillingDueDate || allowed.has('CHANGE_DUE_DATE')) && !flags.isReadOnly && (
          <button
            type="button"
            className={styles.btn}
            disabled={busy}
            onClick={() =>
              setDueDateForm({
                due_at: detail?.due_at ? detail.due_at.slice(0, 10) : todayIso(),
                reason: '',
              })
            }
          >
            Échéance
          </button>
        )}
        {issuedId &&
          Number(detail?.amount_paid || 0) > 0 &&
          (allowed.has('VIEW_PAYMENTS') || canBillingPayment) && (
            <button
              type="button"
              className={styles.btn}
              disabled={busy}
              onClick={() => setActiveTab('payments')}
            >
              Paiements
            </button>
          )}
        {primaryBtn}
      </div>
    </div>
  );

  const dueLabel = detail?.due_at
    ? fmtDate(detail.due_at)
    : dossierRow?.due_at
      ? fmtDate(dossierRow.due_at)
      : null;

  return createPortal(
    <div className={styles.overlay} role="presentation" onClick={onClose}>
      <aside
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="platform-invoice-sheet-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className={styles.header}>
          <div className={styles.headerText}>
            <p className={styles.eyebrow}>
              {detail?.document_type === 'CREDIT_NOTE' ? 'Avoir' : 'Facture'}
            </p>
            <h2 id="platform-invoice-sheet-title" className={styles.title}>
              {titleNumber}
            </h2>
            <div className={styles.metaRow}>
              {statusBadge}
              <span className={styles.meta}>
                {companyLabel}
                {periodLabel}
              </span>
            </div>
            <p className={styles.amountLine}>
              <strong>{amountDisplay}</strong>
              {dueLabel ? <span className={styles.amountHint}>Échéance {dueLabel}</span> : null}
            </p>
          </div>
          <button type="button" className={styles.closeBtn} aria-label="Fermer" onClick={onClose}>
            <FiX size={18} aria-hidden />
          </button>
        </header>

        {error ? (
          <div className={`${styles.banner} ${styles.bannerError}`} role="alert">
            {error}
          </div>
        ) : null}
        {info ? (
          <div className={`${styles.banner} ${styles.bannerOk}`} role="status">
            {info}
          </div>
        ) : null}

        <div className={styles.body}>
          {!issuedId && statement && (
            <section className={styles.section}>
              <h3 className={styles.sectionTitle}>Relevé</h3>
              <dl className={styles.kv}>
                <div>
                  <dt>Statut</dt>
                  <dd>{statement.statement_status || '—'}</dd>
                </div>
                <div>
                  <dt>Total</dt>
                  <dd>{fmtMoney(statement.total_amount)}</dd>
                </div>
                <div>
                  <dt>Portefeuille</dt>
                  <dd>{statement.own_portfolio_count ?? 0}</dd>
                </div>
                <div>
                  <dt>Marketplace</dt>
                  <dd>{statement.lirie_transport_count ?? 0}</dd>
                </div>
              </dl>
              <ul className={styles.lineList}>
                {(statement.lines || []).map((line) => (
                  <li key={line.id}>
                    <span>{line.label}</span>
                    <strong>{fmtMoney(line.amount)}</strong>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {!issuedId && !statement && statementQuery.isLoading && (
            <p className={styles.loading}>Chargement…</p>
          )}

          {!issuedId && !statement && !statementQuery.isLoading && (
            <p className={styles.emptyHint}>
              Aucun relevé. Utilisez « Calculer » depuis le registre.
            </p>
          )}

          {issuedId && detailQuery.isLoading && <p className={styles.loading}>Chargement…</p>}

          {issuedId && detail && (
            <>
              <nav className={styles.tabs} aria-label="Sections de la facture">
                {TABS.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={activeTab === tab.id ? styles.tabActive : styles.tab}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>

              <div className={styles.tabContent}>
                {activeTab === 'overview' && (
                  <section className={styles.section}>
                    <h3 className={styles.sectionTitle}>Situation</h3>
                    <dl className={styles.kv}>
                      <div>
                        <dt>Payé</dt>
                        <dd>{fmtMoney(detail.amount_paid)}</dd>
                      </div>
                      <div>
                        <dt>Solde restant</dt>
                        <dd>{fmtMoney(detail.balance_due)}</dd>
                      </div>
                      <div>
                        <dt>Émise le</dt>
                        <dd>{fmtDate(detail.issued_at)}</dd>
                      </div>
                      <div>
                        <dt>Envoyée le</dt>
                        <dd>{detail.sent_at ? fmtDate(detail.sent_at) : 'Non envoyée'}</dd>
                      </div>
                      {detail.paid_at && (
                        <div>
                          <dt>Payée le</dt>
                          <dd>{fmtDate(detail.paid_at)}</dd>
                        </div>
                      )}
                    </dl>
                  </section>
                )}

                {activeTab === 'detail' && (
                  <section className={styles.section}>
                    <h3 className={styles.sectionTitle}>Lignes de facture</h3>
                    <ul className={styles.lineList}>
                      {(detail.lines || []).length === 0 ? (
                        <li className={styles.emptyHint}>Aucune ligne</li>
                      ) : (
                        (detail.lines || []).map((line, idx) => {
                          const amt = Number(line.amount);
                          const isDiscount =
                            line.line_type === 'DISCOUNT' ||
                            (Number.isFinite(amt) && amt < 0);
                          const unitHint =
                            line.calculation_mode === 'UNIT_PRICE' &&
                            line.quantity != null &&
                            line.unit_amount != null
                              ? `${line.quantity} × ${fmtMoney(line.unit_amount)}`
                              : null;
                          return (
                            <li
                              key={line.id || idx}
                              className={isDiscount ? styles.lineDiscount : undefined}
                            >
                              <div className={styles.lineMain}>
                                <span className={styles.lineLabel}>
                                  {displayInvoiceLineLabel(line)}
                                </span>
                                {unitHint ? (
                                  <span className={styles.lineHint}>{unitHint}</span>
                                ) : null}
                              </div>
                              <strong
                                className={
                                  isDiscount ? styles.lineAmountNeg : undefined
                                }
                              >
                                {fmtMoney(line.amount)}
                              </strong>
                            </li>
                          );
                        })
                      )}
                    </ul>
                    <dl className={styles.lineTotals}>
                      <div>
                        <dt>Sous-total HT</dt>
                        <dd>{fmtMoney(detail.subtotal_amount)}</dd>
                      </div>
                      <div>
                        <dt>
                          TVA
                          {detail.tax_rate != null
                            ? ` (${Number(detail.tax_rate)} %)`
                            : ''}
                        </dt>
                        <dd>{fmtMoney(detail.tax_amount)}</dd>
                      </div>
                      <div className={styles.lineTotalsGrand}>
                        <dt>Total TTC</dt>
                        <dd>{fmtMoney(detail.total_amount)}</dd>
                      </div>
                    </dl>
                  </section>
                )}

                {activeTab === 'payments' && (
                  <section className={styles.section}>
                    <h3 className={styles.sectionTitle}>Paiements</h3>
                    {(detail.payments || []).length === 0 && !paymentForm && (
                      <p className={styles.emptyHint}>Aucun paiement enregistré.</p>
                    )}
                    {(detail.payments || []).map((p) => (
                      <div key={p.id} className={styles.paymentRow}>
                        <span>
                          {fmtMoney(p.amount)} · {fmtDate(p.paid_at)} · {p.method || '—'}
                        </span>
                        {(canBillingPayment || allowed.has('REVERSE_PAYMENT')) &&
                          p.entry_type !== 'REVERSAL' && (
                            <button
                              type="button"
                              className={styles.btnGhost}
                              disabled={busy}
                              onClick={() =>
                                setActionDialog({
                                  title: 'Contre-passer',
                                  description: `Contre-passer ${fmtMoney(p.amount)} ?`,
                                  confirmationLabel: 'Contre-passer',
                                  danger: true,
                                  reason: {
                                    required: true,
                                    label: 'Motif',
                                    minLength: 3,
                                  },
                                  onConfirm: async ({ reason }) => {
                                    await runAction(
                                      () =>
                                        reversePlatformIssuedPayment(issuedId, p.id, {
                                          reason,
                                        }),
                                      { successMessage: 'Paiement contre-passé.' }
                                    );
                                    setActionDialog(null);
                                  },
                                })
                              }
                            >
                              Contre-passer
                            </button>
                          )}
                      </div>
                    ))}
                    {paymentForm && (
                      <form className={styles.form} onSubmit={submitPayment}>
                        <h3 className={styles.formTitle}>Nouveau paiement</h3>
                        <label>
                          Montant
                          <input
                            value={paymentForm.amount}
                            onChange={(e) =>
                              setPaymentForm((f) => ({ ...f, amount: e.target.value }))
                            }
                          />
                        </label>
                        <label>
                          Date
                          <input
                            type="date"
                            value={paymentForm.paid_at}
                            onChange={(e) =>
                              setPaymentForm((f) => ({ ...f, paid_at: e.target.value }))
                            }
                          />
                        </label>
                        <label>
                          Méthode
                          <select
                            value={paymentForm.method}
                            onChange={(e) =>
                              setPaymentForm((f) => ({ ...f, method: e.target.value }))
                            }
                          >
                            {PAYMENT_METHODS.map((m) => (
                              <option key={m.value} value={m.value}>
                                {m.label}
                              </option>
                            ))}
                          </select>
                        </label>
                        <div className={styles.formActions}>
                          <button type="submit" className={styles.btnPrimary} disabled={busy}>
                            Enregistrer
                          </button>
                          <button
                            type="button"
                            className={styles.btn}
                            onClick={() => setPaymentForm(null)}
                          >
                            Fermer
                          </button>
                        </div>
                      </form>
                    )}
                  </section>
                )}

                {activeTab === 'history' && (
                  <section className={styles.section}>
                    <h3 className={styles.sectionTitle}>Historique</h3>
                    {(detail.timeline || []).length === 0 ? (
                      <p className={styles.emptyHint}>Aucun événement enregistré.</p>
                    ) : (
                      <ol className={styles.timeline}>
                        {(detail.timeline || []).map((ev, idx) => (
                          <li
                            key={`${ev.type}-${ev.at}-${idx}`}
                            className={[
                              styles.timelineItem,
                              timelineTypeClass(ev.type, styles),
                            ]
                              .filter(Boolean)
                              .join(' ')}
                          >
                            <time className={styles.timelineTime} dateTime={ev.at || undefined}>
                              {fmtDateTime(ev.at)}
                            </time>
                            <div className={styles.timelineBody}>
                              <span className={styles.timelineLabel}>{ev.label}</span>
                              {ev.detail ? (
                                <span className={styles.timelineDetail}>{ev.detail}</span>
                              ) : null}
                            </div>
                          </li>
                        ))}
                      </ol>
                    )}
                  </section>
                )}
              </div>
            </>
          )}

          {dueDateForm && (
            <form className={styles.form} onSubmit={submitDueDate}>
              <h3 className={styles.formTitle}>Modifier l’échéance</h3>
              <label>
                Date
                <input
                  type="date"
                  value={dueDateForm.due_at}
                  onChange={(e) => setDueDateForm((f) => ({ ...f, due_at: e.target.value }))}
                />
              </label>
              <label>
                Motif
                <input
                  value={dueDateForm.reason}
                  onChange={(e) => setDueDateForm((f) => ({ ...f, reason: e.target.value }))}
                />
              </label>
              <div className={styles.formActions}>
                <button type="submit" className={styles.btnPrimary} disabled={busy}>
                  Enregistrer
                </button>
                <button type="button" className={styles.btn} onClick={() => setDueDateForm(null)}>
                  Fermer
                </button>
              </div>
            </form>
          )}
        </div>

        {actionBar}

        {showEditor && issuedId && (
          <AdminPlatformInvoiceEditor
            issuedId={issuedId}
            onClose={() => setShowEditor(false)}
            onReplaced={(newId) => {
              setShowEditor(false);
              onChanged?.({ replacedIssuedId: newId });
            }}
          />
        )}

        {actionDialog && (
          <AdminActionDialog
            open
            title={actionDialog.title}
            description={actionDialog.description}
            confirmationLabel={actionDialog.confirmationLabel || 'Confirmer'}
            danger={actionDialog.danger}
            reason={actionDialog.reason}
            onConfirm={actionDialog.onConfirm}
            onClose={() => setActionDialog(null)}
            loading={busy}
          />
        )}
      </aside>
    </div>,
    document.body
  );
};

export default AdminPlatformInvoiceSheet;
