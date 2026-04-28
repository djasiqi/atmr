import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  createPlatformBillingPeriod,
  downloadPlatformBillingPeriodExport,
  fetchPlatformBillingInvoice,
  fetchPlatformBillingPeriodInvoices,
  fetchPlatformBillingPeriods,
  lockPlatformBillingPeriod,
  recalculatePlatformBillingPeriod,
} from '../../../services/adminService';
import shell from '../adminShell.module.css';
import styles from './AdminPlatformBilling.module.css';

const fmtMoney = (n) => {
  if (n == null || Number.isNaN(Number(n))) return '—';
  return `${Number(n).toLocaleString('fr-CH', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} CHF`;
};

const statusLabel = (s) => {
  if (s === 'locked') return 'Figé';
  if (s === 'draft') return 'Brouillon';
  return s || '—';
};

const AdminPlatformBilling = () => {
  const { public_id: adminId } = useParams();
  const base = `/dashboard/admin/${adminId}/billing`;

  const [periods, setPeriods] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);

  const [year, setYear] = useState(new Date().getFullYear());
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [creating, setCreating] = useState(false);

  const [selectedId, setSelectedId] = useState(null);
  const [invoicesData, setInvoicesData] = useState(null);
  const [invoicesLoading, setInvoicesLoading] = useState(false);

  const [actionLoading, setActionLoading] = useState(false);

  const [modalInvoice, setModalInvoice] = useState(null);
  const [modalLoading, setModalLoading] = useState(false);

  const loadPeriods = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPlatformBillingPeriods();
      const list = data?.periods || [];
      setPeriods(list);
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement des périodes');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPeriods();
  }, [loadPeriods]);

  const selectedPeriod = useMemo(
    () => periods.find((p) => p.id === selectedId) || null,
    [periods, selectedId]
  );

  const loadInvoices = useCallback(async (periodId) => {
    if (!periodId) {
      setInvoicesData(null);
      return;
    }
    setInvoicesLoading(true);
    try {
      const data = await fetchPlatformBillingPeriodInvoices(periodId);
      setInvoicesData(data);
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement des relevés');
      setInvoicesData(null);
    } finally {
      setInvoicesLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedId) loadInvoices(selectedId);
  }, [selectedId, loadInvoices]);

  const onCreatePeriod = async () => {
    setCreating(true);
    setInfo(null);
    setError(null);
    try {
      await createPlatformBillingPeriod(Number(year), Number(month));
      setInfo(`Période ${year}-${String(month).padStart(2, '0')} créée ou déjà ouverte.`);
      await loadPeriods();
    } catch (e) {
      setError(
        e?.response?.data?.message ||
          (typeof e?.response?.data === 'object' && e?.response?.data?.error) ||
          e?.message ||
          'Erreur création période'
      );
    } finally {
      setCreating(false);
    }
  };

  const onRecalculate = async () => {
    if (!selectedId) return;
    setActionLoading(true);
    setInfo(null);
    setError(null);
    try {
      const out = await recalculatePlatformBillingPeriod(selectedId);
      setInfo(
        `Recalcul terminé — relevés générés : ${out?.invoices_generated ?? '—'} (période #${selectedId}).`
      );
      await loadPeriods();
      await loadInvoices(selectedId);
    } catch (e) {
      const msg =
        e?.response?.data?.error ||
        e?.response?.data?.message ||
        e?.message ||
        'Erreur recalcul';
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } finally {
      setActionLoading(false);
    }
  };

  const onLock = async () => {
    if (!selectedId) return;
    if (
      !window.confirm(
        'Verrouiller cette période ? Le recalcul ne sera plus possible tant que la période reste figée.'
      )
    ) {
      return;
    }
    setActionLoading(true);
    setInfo(null);
    setError(null);
    try {
      await lockPlatformBillingPeriod(selectedId);
      setInfo('Période verrouillée.');
      await loadPeriods();
      await loadInvoices(selectedId);
    } catch (e) {
      const msg =
        e?.response?.data?.error ||
        e?.response?.data?.message ||
        e?.message ||
        'Erreur verrouillage';
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } finally {
      setActionLoading(false);
    }
  };

  const onExport = async () => {
    if (!selectedId) return;
    setError(null);
    try {
      await downloadPlatformBillingPeriodExport(selectedId);
      setInfo('Export CSV téléchargé.');
    } catch (e) {
      setError(e?.message || 'Erreur export');
    }
  };

  const openInvoiceModal = async (invoiceId) => {
    setModalInvoice(null);
    setModalLoading(true);
    try {
      const data = await fetchPlatformBillingInvoice(invoiceId);
      setModalInvoice(data);
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur détail relevé');
    } finally {
      setModalLoading(false);
    }
  };

  return (
    <main className={shell.content}>
      <section className={styles.hero}>
        <h1>Facturation plateforme LIRIE</h1>
        <p className={styles.subtitle}>
          Relevés LIRIE vers les entreprises transporteuses (abonnement, commission institution,
          support). Distinct du pilotage analytique sous « Factures ».
        </p>
        <p className={styles.pilotageLink}>
          <Link to={`${base}/pilotage`}>Voir le pilotage billing (analytique)</Link>
        </p>
      </section>

      {error ? (
        <div className={`${styles.msg} ${styles.msgError}`} role="alert">
          {error}
        </div>
      ) : null}
      {info ? (
        <div className={`${styles.msg} ${styles.msgOk}`} role="status">
          {info}
        </div>
      ) : null}

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>Nouvelle période</h2>
        <div className={styles.createRow}>
          <label>
            Année
            <input
              type="number"
              min="2020"
              max="2100"
              value={year}
              onChange={(e) => setYear(e.target.value)}
            />
          </label>
          <label>
            Mois (1–12)
            <input
              type="number"
              min="1"
              max="12"
              value={month}
              onChange={(e) => setMonth(e.target.value)}
            />
          </label>
          <button
            type="button"
            className={`${styles.btn} ${styles.btnPrimary}`}
            disabled={creating}
            onClick={onCreatePeriod}
          >
            {creating ? '…' : 'Créer ou ouvrir'}
          </button>
        </div>
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>Périodes</h2>
        {loading ? (
          <p className={styles.loading}>Chargement…</p>
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Période</th>
                  <th>Statut</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {periods.length === 0 ? (
                  <tr>
                    <td colSpan={4}>
                      Aucune période. Créez-en une ci-dessus ou via l’API / CLI.
                    </td>
                  </tr>
                ) : (
                  periods.map((p) => (
                    <tr
                      key={p.id}
                      className={selectedId === p.id ? styles.selected : ''}
                    >
                      <td>{p.id}</td>
                      <td>
                        {p.billing_year}-{String(p.billing_month).padStart(2, '0')}
                      </td>
                      <td>
                        <span
                          className={`${styles.badge} ${
                            p.status === 'locked' ? styles.badgeLocked : styles.badgeDraft
                          }`}
                        >
                          {statusLabel(p.status)}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          className={styles.btn}
                          onClick={() => {
                            setSelectedId(p.id);
                            setInfo(null);
                            setError(null);
                          }}
                        >
                          Sélectionner
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selectedPeriod ? (
        <section className={styles.detailPanel}>
          <h2 className={styles.sectionTitle}>
            Période {selectedPeriod.billing_year}-
            {String(selectedPeriod.billing_month).padStart(2, '0')} (#{selectedPeriod.id})
          </h2>
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.btn}
              disabled={actionLoading || selectedPeriod.status === 'locked'}
              onClick={onRecalculate}
            >
              Recalculer brouillons
            </button>
            <button
              type="button"
              className={`${styles.btn} ${styles.btnDanger}`}
              disabled={actionLoading || selectedPeriod.status === 'locked'}
              onClick={onLock}
            >
              Verrouiller
            </button>
            <button type="button" className={styles.btn} disabled={actionLoading} onClick={onExport}>
              Télécharger CSV
            </button>
          </div>

          <h3 className={styles.sectionTitle} style={{ marginTop: '1.25rem' }}>
            Relevés par entreprise
          </h3>
          {invoicesLoading ? (
            <p>Chargement des relevés…</p>
          ) : (
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>ID relevé</th>
                    <th>Entreprise</th>
                    <th>Total</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {(invoicesData?.invoices || []).length === 0 ? (
                    <tr>
                      <td colSpan={4}>Aucun relevé (billing non activé ou rien à facturer).</td>
                    </tr>
                  ) : (
                    (invoicesData?.invoices || []).map((inv) => (
                      <tr key={inv.id}>
                        <td>{inv.id}</td>
                        <td>{inv.company_id}</td>
                        <td className={styles.lineAmount}>{fmtMoney(inv.total_amount)}</td>
                        <td>
                          <button
                            type="button"
                            className={styles.btn}
                            onClick={() => openInvoiceModal(inv.id)}
                          >
                            Lignes
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ) : null}

      {modalInvoice || modalLoading ? (
        <div
          className={styles.modalOverlay}
          role="presentation"
          onClick={() => !modalLoading && setModalInvoice(null)}
        >
          <div
            className={styles.modal}
            role="dialog"
            aria-modal="true"
            aria-labelledby="pb-invoice-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              className={styles.modalClose}
              aria-label="Fermer"
              onClick={() => setModalInvoice(null)}
            >
              ×
            </button>
            <h3 id="pb-invoice-title">Relevé #{modalInvoice?.id}</h3>
            {modalLoading ? (
              <p>Chargement…</p>
            ) : modalInvoice ? (
              <>
                <p>
                  Entreprise {modalInvoice.company_id} — Total {fmtMoney(modalInvoice.total_amount)}
                </p>
                <div className={styles.tableWrap}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Type</th>
                        <th>Libellé</th>
                        <th>Montant</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(modalInvoice.lines || []).map((ln) => (
                        <tr key={ln.id}>
                          <td className={styles.mono}>{ln.line_type}</td>
                          <td>{ln.label || '—'}</td>
                          <td className={styles.lineAmount}>{fmtMoney(ln.amount)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : null}
          </div>
        </div>
      ) : null}
    </main>
  );
};

export default AdminPlatformBilling;
