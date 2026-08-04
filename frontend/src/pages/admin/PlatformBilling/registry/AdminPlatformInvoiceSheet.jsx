import React, { useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchPlatformIssuedInvoice,
  sendPlatformIssuedInvoice,
  payPlatformIssuedInvoice,
  reversePlatformIssuedPayment,
  updatePlatformIssuedDueDate,
  cancelPlatformIssuedInvoice,
  createPlatformIssuedCreditNote,
  downloadPlatformIssuedInvoicePdf,
} from '../../../../services/adminService';
import AdminActionDialog from '../../components/AdminActionDialog';
import { STATUS_LABELS, statusBadgeClass, fmtMoney, fmtDate } from '../issuedInvoiceUi';
import styles from './AdminPlatformInvoiceSheet.module.css';

const PAYMENT_METHODS = [
  { value: 'bank_transfer', label: 'Virement bancaire' },
  { value: 'qr_bill', label: 'QR-facture' },
  { value: 'cash', label: 'Espèces' },
  { value: 'other', label: 'Autre' },
];

const TABS = [
  { id: 'overview', label: 'Aperçu' },
  { id: 'detail', label: 'Détail' },
  { id: 'payments', label: 'Paiements' },
  { id: 'history', label: 'Historique' },
];

const apiErrorMessage = (e) =>
  e?.response?.data?.message ||
  e?.response?.data?.error ||
  e?.message ||
  'Erreur';

const todayIso = () => new Date().toISOString().slice(0, 10);

const AdminPlatformInvoiceSheet = ({ issuedId, onClose, onChanged, capabilities }) => {
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

  const detailQuery = useQuery({
    queryKey: ['admin', 'platform-issued-invoice', issuedId],
    queryFn: () => fetchPlatformIssuedInvoice(issuedId),
    enabled: Boolean(issuedId),
  });

  const detail = detailQuery.data;

  const refreshAll = async () => {
    await queryClient.invalidateQueries({
      queryKey: ['admin', 'platform-issued-invoice', issuedId],
    });
    onChanged?.();
  };

  const flags = useMemo(() => {
    const uiStatus = detail?.ui_status;
    const isIssuedNotSent = uiStatus === 'ISSUED' && !detail?.sent_at;
    const isActiveSent = ['SENT', 'OVERDUE', 'PARTIALLY_PAID'].includes(uiStatus);
    const isPaid = uiStatus === 'PAID';
    const isReadOnly = ['CANCELLED', 'CREDITED'].includes(uiStatus);
    const amountPaidZero = Number(detail?.amount_paid || 0) === 0;
    return {
      uiStatus,
      isIssuedNotSent,
      isActiveSent,
      isPaid,
      isReadOnly,
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
      description: `Confirmer l'envoi de la facture ${detail?.invoice_number || ''} à l'entreprise ?`,
      confirmationLabel: 'Marquer envoyée',
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
      description:
        'Annuler cette facture ? Cette action est réservée aux factures non envoyées et sans paiement.',
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
      description: `Créer un avoir total pour la facture ${detail?.invoice_number || ''} ? Cette action est irréversible.`,
      confirmationLabel: 'Émettre l’avoir',
      danger: true,
      reason: { required: true, label: "Motif de l'avoir", minLength: 3 },
      onConfirm: async ({ reason }) => {
        await runAction(
          () => createPlatformIssuedCreditNote(issuedId, { reason }),
          { successMessage: 'Note de crédit émise.' }
        );
        setActionDialog(null);
      },
    });
  };

  const handleReversePayment = (payment) => {
    setActionDialog({
      title: 'Contre-passer le paiement',
      description: `Annuler l'écriture de paiement de ${fmtMoney(payment.amount)} du ${fmtDate(payment.paid_at)} ?`,
      confirmationLabel: 'Contre-passer',
      danger: true,
      reason: { required: true, label: 'Motif de la contre-passation', minLength: 3 },
      onConfirm: async ({ reason }) => {
        await runAction(
          () => reversePlatformIssuedPayment(issuedId, payment.id, { reason }),
          { successMessage: 'Paiement contre-passé.' }
        );
        setActionDialog(null);
      },
    });
  };

  const handleDownloadPdf = async () => {
    setError(null);
    try {
      await downloadPlatformIssuedInvoicePdf(issuedId);
    } catch (e) {
      setError(apiErrorMessage(e) || 'Téléchargement impossible');
    }
  };

  const openPaymentForm = () => {
    setPaymentForm({
      amount: detail?.balance_due && Number(detail.balance_due) > 0 ? detail.balance_due : '',
      paid_at: todayIso(),
      method: 'bank_transfer',
      reference: '',
      notes: '',
    });
  };

  const submitPayment = async (e) => {
    e.preventDefault();
    const amount = Number(String(paymentForm.amount).replace(',', '.'));
    if (!Number.isFinite(amount) || amount <= 0) {
      setError('Indiquez un montant valide.');
      return;
    }
    setBusy(true);
    setError(null);
    setInfo(null);
    try {
      const idempotencyKey =
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : `pay-${issuedId}-${Date.now()}`;
      await payPlatformIssuedInvoice(issuedId, {
        amount,
        paid_at: paymentForm.paid_at || undefined,
        method: paymentForm.method || undefined,
        reference: paymentForm.reference?.trim() || undefined,
        notes: paymentForm.notes?.trim() || undefined,
        idempotency_key: idempotencyKey,
      });
      setInfo('Paiement enregistré.');
      setPaymentForm(null);
      await refreshAll();
    } catch (e2) {
      setError(apiErrorMessage(e2));
    } finally {
      setBusy(false);
    }
  };

  const openDueDateForm = () => {
    setDueDateForm({
      due_at: detail?.due_at ? detail.due_at.slice(0, 10) : todayIso(),
      reason: '',
    });
  };

  const submitDueDate = async (e) => {
    e.preventDefault();
    if (!dueDateForm.due_at) {
      setError("Indiquez une date d'échéance.");
      return;
    }
    if (!dueDateForm.reason || dueDateForm.reason.trim().length < 3) {
      setError('Le motif est obligatoire (3 caractères minimum).');
      return;
    }
    setBusy(true);
    setError(null);
    setInfo(null);
    try {
      await updatePlatformIssuedDueDate(issuedId, {
        due_at: dueDateForm.due_at,
        reason: dueDateForm.reason.trim(),
      });
      setInfo('Échéance mise à jour.');
      setDueDateForm(null);
      await refreshAll();
    } catch (e2) {
      setError(apiErrorMessage(e2));
    } finally {
      setBusy(false);
    }
  };

  const isCreditNoteDoc = detail?.document_type === 'CREDIT_NOTE';

  return (
    <div className={styles.overlay} role="presentation" onClick={onClose}>
      <div
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="platform-invoice-sheet-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <div>
            <h2 id="platform-invoice-sheet-title" className={styles.title}>
              {isCreditNoteDoc ? 'Avoir ' : 'Facture '}
              {detail?.invoice_number || `#${issuedId}`}
            </h2>
            <p className={styles.subtitle}>
              {detail?.company_name || '—'}
              {detail?.ui_status ? (
                <span
                  className={`${styles.badge} ${statusBadgeClass(detail.ui_status, styles)}`}
                >
                  {STATUS_LABELS[detail.ui_status] || detail.ui_status}
                </span>
              ) : null}
            </p>
          </div>
          <button type="button" className={styles.closeBtn} aria-label="Fermer" onClick={onClose}>
            ×
          </button>
        </div>

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

        {detailQuery.isLoading ? (
          <p className={styles.loading}>Chargement…</p>
        ) : detailQuery.isError ? (
          <div className={`${styles.banner} ${styles.bannerError}`} role="alert">
            {apiErrorMessage(detailQuery.error)}
          </div>
        ) : detail ? (
          <>
            <nav className={styles.tabs}>
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
              {activeTab === 'overview' ? (
                <>
                  <div className={styles.summaryGrid}>
                    <div className={styles.summaryItem}>
                      <span>Entreprise</span>
                      <strong>{detail.company_name || '—'}</strong>
                    </div>
                    <div className={styles.summaryItem}>
                      <span>Période</span>
                      <strong>
                        {detail.billing_year && detail.billing_month
                          ? `${String(detail.billing_month).padStart(2, '0')}.${detail.billing_year}`
                          : '—'}
                      </strong>
                    </div>
                    <div className={styles.summaryItem}>
                      <span>Émission</span>
                      <strong>{fmtDate(detail.issued_at)}</strong>
                    </div>
                    <div className={styles.summaryItem}>
                      <span>Échéance</span>
                      <strong>{fmtDate(detail.due_at)}</strong>
                    </div>
                    <div className={styles.summaryItem}>
                      <span>Envoi</span>
                      <strong>{detail.sent_at ? fmtDate(detail.sent_at) : '—'}</strong>
                    </div>
                    <div className={styles.summaryItem}>
                      <span>Total TTC</span>
                      <strong>{fmtMoney(detail.total_amount)}</strong>
                    </div>
                    <div className={styles.summaryItem}>
                      <span>Payé</span>
                      <strong>{fmtMoney(detail.amount_paid)}</strong>
                    </div>
                    <div className={`${styles.summaryItem} ${styles.summaryTotal}`}>
                      <span>Solde dû</span>
                      <strong>{fmtMoney(detail.balance_due)}</strong>
                    </div>
                  </div>

                  {detail.ui_status === 'OVERDUE' && detail.payment_state === 'PARTIAL' ? (
                    <p className={styles.notice}>
                      Partiellement payée — solde {fmtMoney(detail.balance_due)}.
                    </p>
                  ) : null}

                  <div className={styles.actions}>
                    {flags.isIssuedNotSent ? (
                      <>
                        <button
                          type="button"
                          className={`${styles.btn} ${styles.btnPrimary}`}
                          disabled={busy || !canBillingSend}
                          title={!canBillingSend ? 'Capacité admin.billing.send requise' : undefined}
                          onClick={handleSend}
                        >
                          Marquer envoyée
                        </button>
                        <button
                          type="button"
                          className={styles.btn}
                          disabled={busy || !canBillingDueDate}
                          title={
                            !canBillingDueDate ? 'Capacité admin.billing.due_date requise' : undefined
                          }
                          onClick={openDueDateForm}
                        >
                          Modifier l’échéance
                        </button>
                        <button
                          type="button"
                          className={`${styles.btn} ${styles.btnDanger}`}
                          disabled={busy || !canBillingCancel}
                          title={
                            !canBillingCancel ? 'Capacité admin.billing.cancel requise' : undefined
                          }
                          onClick={handleCancel}
                        >
                          Annuler
                        </button>
                      </>
                    ) : null}

                    {flags.isActiveSent ? (
                      <>
                        <button
                          type="button"
                          className={`${styles.btn} ${styles.btnPrimary}`}
                          disabled={busy || !canBillingPayment}
                          title={
                            !canBillingPayment ? 'Capacité admin.billing.payment requise' : undefined
                          }
                          onClick={openPaymentForm}
                        >
                          Enregistrer un paiement
                        </button>
                        <button
                          type="button"
                          className={styles.btn}
                          disabled={busy || !canBillingDueDate}
                          title={
                            !canBillingDueDate ? 'Capacité admin.billing.due_date requise' : undefined
                          }
                          onClick={openDueDateForm}
                        >
                          Prolonger l’échéance
                        </button>
                        {flags.canCreditNote ? (
                          <button
                            type="button"
                            className={`${styles.btn} ${styles.btnDanger}`}
                            disabled={busy || !canBillingCredit}
                            title={
                              !canBillingCredit ? 'Capacité admin.billing.credit requise' : undefined
                            }
                            onClick={handleCreditNote}
                          >
                            Émettre un avoir
                          </button>
                        ) : null}
                      </>
                    ) : null}

                    {detail.pdf_storage_key ? (
                      <button
                        type="button"
                        className={styles.btn}
                        disabled={busy}
                        onClick={handleDownloadPdf}
                      >
                        Télécharger PDF
                      </button>
                    ) : null}
                  </div>

                  {flags.isReadOnly ? (
                    <p className={styles.subtitleMuted}>
                      {detail.ui_status === 'CANCELLED'
                        ? 'Facture annulée — lecture seule.'
                        : 'Facture créditée — lecture seule.'}
                    </p>
                  ) : null}
                  {flags.isPaid ? (
                    <p className={styles.subtitleMuted}>
                      Facture intégralement payée — aucun avoir possible.
                    </p>
                  ) : null}
                </>
              ) : null}

              {activeTab === 'detail' ? (
                <>
                  <div className={styles.tableWrap}>
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>Libellé</th>
                          <th className={styles.num}>Qté</th>
                          <th className={styles.num}>P.U.</th>
                          <th className={styles.num}>Montant</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(detail.statement_lines || []).length === 0 ? (
                          <tr>
                            <td colSpan={4} className={styles.emptyCell}>
                              Aucune ligne de détail disponible.
                            </td>
                          </tr>
                        ) : (
                          detail.statement_lines.map((ln) => (
                            <tr key={ln.id}>
                              <td>{ln.label || ln.line_type || '—'}</td>
                              <td className={styles.num}>{ln.quantity ?? '—'}</td>
                              <td className={styles.num}>{fmtMoney(ln.unit_amount)}</td>
                              <td className={styles.num}>{fmtMoney(ln.amount)}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                  {detail.qr_reference ? (
                    <p className={styles.subtitleMuted}>
                      Référence QR : <span className={styles.mono}>{detail.qr_reference}</span> ·
                      Montant QR : {fmtMoney(detail.qr_amount)}
                    </p>
                  ) : null}
                </>
              ) : null}

              {activeTab === 'payments' ? (
                <>
                  {flags.isActiveSent ? (
                    <div className={styles.actions}>
                      <button
                        type="button"
                        className={`${styles.btn} ${styles.btnPrimary}`}
                        disabled={busy || !canBillingPayment}
                        title={
                          !canBillingPayment ? 'Capacité admin.billing.payment requise' : undefined
                        }
                        onClick={openPaymentForm}
                      >
                        Enregistrer un paiement
                      </button>
                    </div>
                  ) : null}
                  <div className={styles.tableWrap}>
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Type</th>
                          <th className={styles.num}>Montant</th>
                          <th>Méthode</th>
                          <th>Référence</th>
                          <th>Notes</th>
                          <th>
                            <span className={styles.srOnly}>Actions</span>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {(detail.payments || []).length === 0 ? (
                          <tr>
                            <td colSpan={7} className={styles.emptyCell}>
                              Aucun paiement enregistré.
                            </td>
                          </tr>
                        ) : (
                          detail.payments.map((p) => (
                            <tr key={p.id}>
                              <td>{fmtDate(p.paid_at)}</td>
                              <td>{p.entry_type === 'REVERSAL' ? 'Contre-écriture' : 'Paiement'}</td>
                              <td className={styles.num}>{fmtMoney(p.amount)}</td>
                              <td>
                                {PAYMENT_METHODS.find((m) => m.value === p.method)?.label ||
                                  p.method ||
                                  '—'}
                              </td>
                              <td>{p.reference || '—'}</td>
                              <td>{p.notes || p.reversal_reason || '—'}</td>
                              <td className={styles.cellAction}>
                                {p.entry_type === 'PAYMENT' && !p.reverses_payment_id ? (
                                  <button
                                    type="button"
                                    className={styles.linkBtnDanger}
                                    disabled={busy || !canBillingPayment}
                                    onClick={() => handleReversePayment(p)}
                                  >
                                    Contre-passer
                                  </button>
                                ) : null}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : null}

              {activeTab === 'history' ? (
                <>
                  <h4 className={styles.sectionLabel}>Historique des échéances</h4>
                  <div className={styles.tableWrap}>
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Ancienne échéance</th>
                          <th>Nouvelle échéance</th>
                          <th>Motif</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(detail.due_date_changes || []).length === 0 ? (
                          <tr>
                            <td colSpan={4} className={styles.emptyCell}>
                              Aucun changement d’échéance.
                            </td>
                          </tr>
                        ) : (
                          detail.due_date_changes.map((c) => (
                            <tr key={c.id}>
                              <td>{fmtDate(c.created_at)}</td>
                              <td>{fmtDate(c.old_due_at)}</td>
                              <td>{fmtDate(c.new_due_at)}</td>
                              <td>{c.reason || '—'}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>

                  {detail.dunning_case ? (
                    <div className={styles.noticeBox}>
                      Dossier de relance {detail.dunning_case.status} ouvert le{' '}
                      {fmtDate(detail.dunning_case.opened_at)}.
                    </div>
                  ) : null}
                  {(detail.dunning_holds || []).length > 0 ? (
                    <div className={styles.noticeBox}>
                      {detail.dunning_holds.length} retenue(s) de relance active(s).
                    </div>
                  ) : null}
                </>
              ) : null}
            </div>
          </>
        ) : null}

        {paymentForm ? (
          <div
            className={styles.formOverlay}
            role="presentation"
            onClick={() => !busy && setPaymentForm(null)}
          >
            <form
              className={styles.formModal}
              onClick={(e) => e.stopPropagation()}
              onSubmit={submitPayment}
            >
              <h3 className={styles.formTitle}>Enregistrer un paiement</h3>
              <label className={styles.formField}>
                Montant (CHF)
                <input
                  type="text"
                  value={paymentForm.amount}
                  onChange={(e) =>
                    setPaymentForm((f) => ({ ...f, amount: e.target.value }))
                  }
                  placeholder="ex. 150.00"
                  required
                  autoFocus
                />
              </label>
              <label className={styles.formField}>
                Date du paiement
                <input
                  type="date"
                  value={paymentForm.paid_at}
                  onChange={(e) =>
                    setPaymentForm((f) => ({ ...f, paid_at: e.target.value }))
                  }
                />
              </label>
              <label className={styles.formField}>
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
              <label className={styles.formField}>
                Référence
                <input
                  type="text"
                  value={paymentForm.reference}
                  onChange={(e) =>
                    setPaymentForm((f) => ({ ...f, reference: e.target.value }))
                  }
                  placeholder="ex. référence QR / virement"
                />
              </label>
              <label className={styles.formField}>
                Notes
                <textarea
                  rows={2}
                  value={paymentForm.notes}
                  onChange={(e) =>
                    setPaymentForm((f) => ({ ...f, notes: e.target.value }))
                  }
                />
              </label>
              <div className={styles.formActions}>
                <button
                  type="button"
                  className={styles.btn}
                  disabled={busy}
                  onClick={() => setPaymentForm(null)}
                >
                  Annuler
                </button>
                <button type="submit" className={`${styles.btn} ${styles.btnPrimary}`} disabled={busy}>
                  {busy ? 'Enregistrement…' : 'Enregistrer'}
                </button>
              </div>
            </form>
          </div>
        ) : null}

        {dueDateForm ? (
          <div
            className={styles.formOverlay}
            role="presentation"
            onClick={() => !busy && setDueDateForm(null)}
          >
            <form
              className={styles.formModal}
              onClick={(e) => e.stopPropagation()}
              onSubmit={submitDueDate}
            >
              <h3 className={styles.formTitle}>Modifier l’échéance</h3>
              <label className={styles.formField}>
                Nouvelle échéance
                <input
                  type="date"
                  value={dueDateForm.due_at}
                  onChange={(e) =>
                    setDueDateForm((f) => ({ ...f, due_at: e.target.value }))
                  }
                  required
                  autoFocus
                />
              </label>
              <label className={styles.formField}>
                Motif *
                <textarea
                  rows={3}
                  value={dueDateForm.reason}
                  onChange={(e) =>
                    setDueDateForm((f) => ({ ...f, reason: e.target.value }))
                  }
                  required
                />
              </label>
              <div className={styles.formActions}>
                <button
                  type="button"
                  className={styles.btn}
                  disabled={busy}
                  onClick={() => setDueDateForm(null)}
                >
                  Annuler
                </button>
                <button type="submit" className={`${styles.btn} ${styles.btnPrimary}`} disabled={busy}>
                  {busy ? 'Enregistrement…' : 'Enregistrer'}
                </button>
              </div>
            </form>
          </div>
        ) : null}

        {actionDialog ? (
          <AdminActionDialog
            open
            title={actionDialog.title}
            description={actionDialog.description}
            confirmationLabel={actionDialog.confirmationLabel}
            danger={Boolean(actionDialog.danger)}
            reason={actionDialog.reason}
            onConfirm={actionDialog.onConfirm}
            onClose={() => setActionDialog(null)}
          />
        ) : null}
      </div>
    </div>
  );
};

export default AdminPlatformInvoiceSheet;
