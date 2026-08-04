import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  createPlatformBillingPeriod,
  createPlatformSupportEntry,
  deletePlatformSupportEntry,
  downloadPlatformBillingPeriodExport,
  downloadPlatformIssuedInvoicePdf,
  fetchPlatformBillingCompaniesConfig,
  fetchPlatformBillingInvoice,
  fetchPlatformBillingPeriodInvoices,
  fetchPlatformBillingPeriods,
  fetchPlatformSupportEntries,
  issuePlatformBillingInvoice,
  lockPlatformBillingPeriod,
  recalculatePlatformBillingPeriod,
  reopenPlatformBillingInvoice,
  updatePlatformSupportEntry,
  validatePlatformBillingInvoice,
} from '../../../services/adminService';
import AdminActionDialog from '../components/AdminActionDialog';
import { useAdminCapabilities } from '../../../hooks/useAdminCapabilities';
import { STATUS_LABELS, statusBadgeClass, fmtDate } from './issuedInvoiceUi';
import styles from './AdminPlatformBilling.module.css';

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

const SUPPORT_CATEGORIES = [
  { value: 'support', label: 'Support' },
  { value: 'training', label: 'Formation' },
  { value: 'configuration', label: 'Configuration' },
  { value: 'other', label: 'Autre' },
];

const fmtMoney = (n) => {
  if (n == null || n === '') return '—';
  return `${String(n)} CHF`;
};

/** Quantité / heures sans zéros inutiles (1.50 → 1.5). */
const fmtQty = (n) => {
  if (n == null || n === '') return null;
  const v = Number(String(n).replace(',', '.'));
  if (!Number.isFinite(v)) return null;
  return String(v.toFixed(4)).replace(/\.?0+$/, '');
};

/** Prix unitaire (CHF) ou taux (%) selon la ligne. */
const fmtUnit = (ln) => {
  const isCommission = String(ln?.line_type || '')
    .toLowerCase()
    .includes('commission');
  const pct = fmtQty(ln?.unit_rate_percent);
  const unit = ln?.unit_amount;
  if (isCommission && pct != null) {
    return unit != null && unit !== ''
      ? `${pct} % · ${fmtMoney(unit)}`
      : `${pct} %`;
  }
  if (unit != null && unit !== '') return fmtMoney(unit);
  if (pct != null) return `${pct} %`;
  return '—';
};

const supportHoursOf = (e) => {
  const fromHours = fmtQty(e?.duration_hours);
  if (fromHours != null) return fromHours;
  if (e?.duration_minutes != null) return fmtQty(Number(e.duration_minutes) / 60);
  return null;
};

const statementLabel = (s) => {
  const v = String(s || '').toUpperCase();
  if (v === 'NEEDS_REVIEW') return 'À contrôler';
  if (v === 'VALIDATED' || v === 'LOCKED') return 'Prête';
  if (v === 'CALCULATED') return 'Calculée';
  if (v === 'DRAFT') return 'Brouillon';
  return s || '—';
};

const statementBadge = (s) => {
  const label = statementLabel(s);
  if (label === 'Prête') return styles.badgeReady;
  if (label === 'Calculée') return styles.badgeCalc;
  if (label === 'À contrôler') return styles.badgeReview;
  return styles.badgeMuted;
};

/** True si le mois calendaire Europe/Zurich est terminé. */
const billingPeriodHasEnded = (year, month) => {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Europe/Zurich',
    year: 'numeric',
    month: '2-digit',
  }).formatToParts(new Date());
  const cy = Number(parts.find((p) => p.type === 'year')?.value);
  const cm = Number(parts.find((p) => p.type === 'month')?.value);
  if (!cy || !cm) return false;
  return Number(year) < cy || (Number(year) === cy && Number(month) < cm);
};

const apiErrorMessage = (e) =>
  e?.response?.data?.message ||
  e?.response?.data?.error ||
  e?.message ||
  'Erreur';

const AdminPlatformBilling = () => {
  const location = useLocation();
  const { canBillingLock, canBillingIssue, canBillingValidate } =
    useAdminCapabilities();
  const focusFromOverview = location.state || {};
  const openedFocusRef = useRef(null);
  const didAutoSelectRef = useRef(false);

  const now = new Date();
  const [year, setYear] = useState(now.getFullYear());
  const [month, setMonth] = useState(now.getMonth() + 1);

  const [periods, setPeriods] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);

  const [selectedId, setSelectedId] = useState(focusFromOverview.periodId || null);
  const [invoicesData, setInvoicesData] = useState(null);
  const [invoicesLoading, setInvoicesLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [companyNames, setCompanyNames] = useState({});
  const [companySupportRates, setCompanySupportRates] = useState({});

  const [modalInvoice, setModalInvoice] = useState(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [supportEntries, setSupportEntries] = useState([]);
  const [supportHours, setSupportHours] = useState('');
  const [supportDesc, setSupportDesc] = useState('');
  const [supportCategory, setSupportCategory] = useState('support');
  const [supportRate, setSupportRate] = useState('');
  const [supportSaving, setSupportSaving] = useState(false);
  const [editingSupportId, setEditingSupportId] = useState(null);
  const [actionDialog, setActionDialog] = useState(null);

  const loadPeriods = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, cfgRes] = await Promise.all([
        fetchPlatformBillingPeriods(),
        fetchPlatformBillingCompaniesConfig({}).catch(() => null),
      ]);
      const list = data?.periods || [];
      setPeriods(list);
      const names = {};
      const rates = {};
      (cfgRes?.items || []).forEach((item) => {
        if (item?.company_id != null) {
          names[item.company_id] = item.company_name || `Entreprise #${item.company_id}`;
          if (item.config?.support_hourly_rate_default != null) {
            rates[item.company_id] = String(item.config.support_hourly_rate_default);
          }
        }
      });
      setCompanyNames(names);
      setCompanySupportRates(rates);
      return list;
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement des périodes');
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPeriods();
  }, [loadPeriods]);

  const resolveCompanyName = (inv) =>
    inv?.company_name ||
    companyNames[inv?.company_id] ||
    (inv?.company_id != null ? `Entreprise #${inv.company_id}` : '—');

  useEffect(() => {
    if (focusFromOverview.periodId) {
      setSelectedId(focusFromOverview.periodId);
    }
  }, [focusFromOverview.periodId]);

  // Auto-sélection initiale uniquement (évite de forcer le mois choisi par l’utilisateur)
  useEffect(() => {
    if (didAutoSelectRef.current || selectedId || !periods.length) return;
    didAutoSelectRef.current = true;
    if (focusFromOverview.periodId) return;
    const match = periods.find(
      (p) => p.billing_year === Number(year) && p.billing_month === Number(month)
    );
    if (match) {
      setSelectedId(match.id);
      return;
    }
    const sorted = [...periods].sort((a, b) => {
      if (a.billing_year !== b.billing_year) return b.billing_year - a.billing_year;
      return b.billing_month - a.billing_month;
    });
    setSelectedId(sorted[0].id);
    setYear(sorted[0].billing_year);
    setMonth(sorted[0].billing_month);
  }, [periods, selectedId, year, month, focusFromOverview.periodId]);

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

  const onSelectMonth = (y, m) => {
    setYear(y);
    setMonth(m);
    const match = periods.find(
      (p) => p.billing_year === Number(y) && p.billing_month === Number(m)
    );
    setSelectedId(match?.id || null);
    setInfo(null);
    setError(null);
  };

  const onOpenOrCreate = async () => {
    setCreating(true);
    setInfo(null);
    setError(null);
    try {
      const res = await createPlatformBillingPeriod(Number(year), Number(month));
      const p = res?.period || res;
      const list = await loadPeriods();
      const id = p?.id || list.find(
        (x) => x.billing_year === Number(year) && x.billing_month === Number(month)
      )?.id;
      if (id) setSelectedId(id);
      setInfo(`Période ${year}-${String(month).padStart(2, '0')} prête.`);
    } catch (e) {
      setError(
        e?.response?.data?.message ||
          e?.response?.data?.error ||
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
      setInfo(`Recalcul terminé — ${out?.invoices_generated ?? '—'} relevé(s).`);
      await loadPeriods();
      await loadInvoices(selectedId);
    } catch (e) {
      setError(
        e?.response?.data?.error || e?.response?.data?.message || e?.message || 'Erreur recalcul'
      );
    } finally {
      setActionLoading(false);
    }
  };

  const onLock = () => {
    if (!selectedId) return;
    setActionDialog({
      title: 'Verrouiller la période',
      description: 'Verrouiller cette période ? Le recalcul ne sera plus possible.',
      confirmationLabel: 'Verrouiller',
      danger: true,
      onConfirm: async () => {
        setActionLoading(true);
        setInfo(null);
        setError(null);
        try {
          await lockPlatformBillingPeriod(selectedId);
          setInfo('Période verrouillée.');
          await loadPeriods();
          await loadInvoices(selectedId);
          setActionDialog(null);
        } finally {
          setActionLoading(false);
        }
      },
    });
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

  const openInvoiceModal = useCallback(async (invoiceId) => {
    setModalInvoice(null);
    setSupportEntries([]);
    setSupportHours('');
    setSupportDesc('');
    setSupportCategory('support');
    setSupportRate('');
    setEditingSupportId(null);
    setModalLoading(true);
    try {
      const data = await fetchPlatformBillingInvoice(invoiceId);
      setModalInvoice(data);
      if (data?.company_id) {
        const defaultRate = companySupportRates[data.company_id];
        if (defaultRate) setSupportRate(defaultRate);
        try {
          const se = await fetchPlatformSupportEntries(data.company_id);
          const periodId = data.period_id;
          const all = se?.entries || [];
          setSupportEntries(
            periodId != null
              ? all.filter(
                  (e) =>
                    e.billing_period_id == null ||
                    Number(e.billing_period_id) === Number(periodId)
                )
              : all
          );
        } catch {
          setSupportEntries([]);
        }
      }
    } catch (e) {
      const status = e?.response?.status;
      const msg =
        e?.response?.data?.message ||
        e?.response?.data?.error ||
        e?.message ||
        'Erreur détail relevé';
      setError(
        status === 404
          ? 'Ce relevé a été recalculé (nouvel identifiant). Rouvrez-le depuis la liste.'
          : msg
      );
      setModalInvoice(null);
    } finally {
      setModalLoading(false);
    }
  }, [companySupportRates]);

  const statementEditable =
    modalInvoice &&
    !modalInvoice.issued_invoice &&
    modalInvoice.statement_status !== 'VALIDATED' &&
    modalInvoice.statement_status !== 'LOCKED';

  /**
   * Après recalcul, le relevé est régénéré (nouvel id).
   * On recharge la liste période puis on rouvre le relevé de la même entreprise.
   */
  const reloadModalAfterRecalculate = async (companyId) => {
    const periodId = modalInvoice?.period_id || selectedId;
    if (!periodId || companyId == null) {
      setModalInvoice(null);
      return;
    }
    const data = await fetchPlatformBillingPeriodInvoices(periodId);
    setInvoicesData(data);
    const next = (data?.invoices || []).find(
      (inv) => Number(inv.company_id) === Number(companyId)
    );
    if (next?.id) {
      await openInvoiceModal(next.id);
    } else {
      setModalInvoice(null);
      setSupportEntries([]);
      setInfo('Relevé recalculé — rouvrez la ligne entreprise dans la liste.');
    }
  };

  const resetSupportForm = () => {
    setSupportHours('');
    setSupportDesc('');
    setSupportCategory('support');
    setEditingSupportId(null);
    const defaultRate = modalInvoice?.company_id
      ? companySupportRates[modalInvoice.company_id]
      : '';
    setSupportRate(defaultRate || '');
  };

  const startEditSupport = (entry) => {
    setEditingSupportId(entry.id);
    setSupportHours(supportHoursOf(entry) || '');
    setSupportCategory(entry.category || 'support');
    setSupportDesc(entry.description || '');
    setSupportRate(
      entry.hourly_rate_snapshot != null
        ? String(entry.hourly_rate_snapshot)
        : companySupportRates[modalInvoice?.company_id] || ''
    );
  };

  const onDeleteSupport = (entry) => {
    if (!entry?.id) return;
    setActionDialog({
      title: 'Supprimer l’entrée support',
      description: 'Supprimer cette entrée support ?',
      confirmationLabel: 'Supprimer',
      danger: true,
      onConfirm: async () => {
        const companyId = modalInvoice?.company_id;
        setSupportSaving(true);
        setError(null);
        setInfo(null);
        try {
          const res = await deletePlatformSupportEntry(entry.id, {
            recalculate_period: true,
            billing_period_id: modalInvoice?.period_id || selectedId || undefined,
          });
          if (res?.recalculate_error) {
            setInfo(`Entrée supprimée — recalcul : ${res.recalculate_error}`);
          } else {
            setInfo('Entrée support supprimée.');
          }
          resetSupportForm();
          await reloadModalAfterRecalculate(companyId);
          setActionDialog(null);
        } finally {
          setSupportSaving(false);
        }
      },
    });
  };

  const onAddSupportHours = async () => {
    if (!modalInvoice?.company_id) return;
    const companyId = modalInvoice.company_id;
    const hours = Number(String(supportHours).replace(',', '.'));
    if (!Number.isFinite(hours) || hours <= 0) {
      setError('Indiquez un nombre d’heures valide (ex. 1.5).');
      return;
    }
    if (supportCategory === 'other' && !supportDesc.trim()) {
      setError('Précisez la description pour la catégorie « Autre ».');
      return;
    }
    setSupportSaving(true);
    setError(null);
    setInfo(null);
    try {
      const payload = {
        company_id: companyId,
        duration_hours: hours,
        category: supportCategory,
        description:
          supportCategory === 'other' ? supportDesc.trim() : null,
        billing_period_id: modalInvoice.period_id || selectedId || undefined,
        auto_validate: true,
        recalculate_period: true,
      };
      if (supportRate.trim()) {
        payload.hourly_rate_snapshot = supportRate.trim().replace(',', '.');
      }
      const res = editingSupportId
        ? await updatePlatformSupportEntry(editingSupportId, payload)
        : await createPlatformSupportEntry(payload);
      if (res?.recalculate_error) {
        setInfo(
          `${editingSupportId ? 'Correction enregistrée' : 'Heures enregistrées'} — recalcul : ${res.recalculate_error}`
        );
      } else {
        setInfo(
          editingSupportId
            ? `Support corrigé : ${res?.entry?.duration_hours || hours} h — ${
                res?.entry?.amount || '—'
              } CHF.`
            : `Support ajouté : ${res?.entry?.duration_hours || hours} h — ${
                res?.entry?.amount || '—'
              } CHF.`
        );
      }
      resetSupportForm();
      await reloadModalAfterRecalculate(companyId);
    } catch (e) {
      setError(
        e?.response?.data?.error ||
          e?.response?.data?.message ||
          e?.message ||
          'Impossible d’enregistrer les heures'
      );
    } finally {
      setSupportSaving(false);
    }
  };

  useEffect(() => {
    const focusId = focusFromOverview.focusInvoiceId;
    if (!focusId || !invoicesData?.invoices?.length) return;
    if (openedFocusRef.current === focusId) return;
    if (invoicesData.invoices.some((inv) => inv.id === focusId)) {
      openedFocusRef.current = focusId;
      openInvoiceModal(focusId);
    }
  }, [focusFromOverview.focusInvoiceId, invoicesData, openInvoiceModal]);

  const periodLabel = selectedPeriod
    ? `${MONTHS_FR[selectedPeriod.billing_month - 1]} ${selectedPeriod.billing_year}`
    : `${MONTHS_FR[Number(month) - 1]} ${year}`;

  const invoices = invoicesData?.invoices || [];

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <label className={styles.field}>
            Année
            <input
              type="number"
              min="2020"
              max="2100"
              value={year}
              onChange={(e) => onSelectMonth(Number(e.target.value), month)}
            />
          </label>
          <label className={styles.field}>
            Mois
            <select
              value={month}
              onChange={(e) => onSelectMonth(year, Number(e.target.value))}
            >
              {MONTHS_FR.map((label, i) => (
                <option key={label} value={i + 1}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <div className={styles.periodMeta}>
            <span>{periodLabel}</span>
            {!selectedPeriod ? (
              <span className={`${styles.chip} ${styles.chipMissing}`}>Non créée</span>
            ) : selectedPeriod.status === 'locked' ? (
              <span className={`${styles.chip} ${styles.chipLocked}`}>Verrouillée</span>
            ) : (
              <span className={`${styles.chip} ${styles.chipOpen}`}>Ouverte</span>
            )}
          </div>
        </div>
        <div className={styles.toolbarRight}>
          {!selectedPeriod ? (
            <button
              type="button"
              className={`${styles.btn} ${styles.btnPrimary}`}
              disabled={creating || loading}
              onClick={onOpenOrCreate}
            >
              {creating ? 'Ouverture…' : 'Ouvrir la période'}
            </button>
          ) : (
            <>
              <button
                type="button"
                className={styles.btn}
                disabled={actionLoading || selectedPeriod.status === 'locked'}
                onClick={onRecalculate}
              >
                Recalculer
              </button>
              <button
                type="button"
                className={styles.btn}
                disabled={actionLoading}
                onClick={onExport}
              >
                Export CSV
              </button>
              <button
                type="button"
                className={`${styles.btn} ${styles.btnDanger}`}
                disabled={
                  actionLoading ||
                  selectedPeriod.status === 'locked' ||
                  !canBillingLock
                }
                onClick={onLock}
                title={
                  !canBillingLock
                    ? 'Capacité admin.billing.lock requise'
                    : undefined
                }
              >
                Verrouiller
              </button>
            </>
          )}
        </div>
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

      <section className={styles.panel}>
        <div className={styles.panelHead}>
          <h2 className={styles.panelTitle}>Relevés par entreprise</h2>
          <span className={styles.hint}>
            {selectedPeriod
              ? `${invoices.length} relevé${invoices.length > 1 ? 's' : ''}`
              : 'Ouvrez une période pour afficher les relevés'}
          </span>
        </div>

        {loading || invoicesLoading ? (
          <p className={styles.loading}>Chargement…</p>
        ) : !selectedPeriod ? (
          <div className={styles.empty}>
            Aucune période pour {MONTHS_FR[Number(month) - 1]} {year}.
            <br />
            Cliquez sur « Ouvrir la période », puis recalculez depuis la vue d&apos;ensemble
            ou ici.
          </div>
        ) : invoices.length === 0 ? (
          <div className={styles.empty}>
            Aucun relevé pour {periodLabel}. Cliquez sur « Recalculer » après avoir configuré
            les entreprises.
          </div>
        ) : (
          <div className={styles.tableWrap}>
            <table className={`${styles.table} ${styles.tableReleves}`}>
              <colgroup>
                <col className={styles.colCompany} />
                <col className={styles.colCount} />
                <col className={styles.colCount} />
                <col className={styles.colState} />
                <col className={styles.colAmount} />
                <col className={styles.colInvoice} />
                <col className={styles.colAction} />
              </colgroup>
              <thead>
                <tr>
                  <th scope="col">Entreprise</th>
                  <th scope="col" className={styles.colHead}>
                    <span className={styles.thMain}>Portefeuille</span>
                    <span className={styles.thSub}>nb courses (abo)</span>
                  </th>
                  <th scope="col" className={styles.colHead}>
                    <span className={styles.thMain}>Marketplace</span>
                    <span className={styles.thSub}>nb transports LIRIE</span>
                  </th>
                  <th scope="col" className={styles.colHeadCenter}>
                    État
                  </th>
                  <th scope="col" className={styles.colHead}>
                    <span className={styles.thMain}>Montant</span>
                    <span className={styles.thSub}>total TTC</span>
                  </th>
                  <th scope="col" className={styles.colHead}>
                    <span className={styles.thMain}>Facture LIRIE</span>
                    <span className={styles.thSub}>n°, échéance, solde</span>
                  </th>
                  <th scope="col" className={styles.colHeadAction}>
                    <span className={styles.srOnly}>Action</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((inv) => {
                  const issued = inv.issued_invoice;
                  return (
                  <tr key={inv.id}>
                    <td className={styles.companyName}>{resolveCompanyName(inv)}</td>
                    <td className={styles.cellCount}>
                      <span className={styles.countValue}>
                        {inv.own_portfolio_count != null ? inv.own_portfolio_count : '—'}
                      </span>
                      <span className={styles.countUnit}>courses</span>
                    </td>
                    <td className={styles.cellCount}>
                      <span className={styles.countValue}>
                        {inv.lirie_transport_count != null
                          ? inv.lirie_transport_count
                          : '—'}
                      </span>
                      <span className={styles.countUnit}>transports</span>
                    </td>
                    <td className={styles.cellState}>
                      <span
                        className={`${styles.badge} ${statementBadge(inv.statement_status)}`}
                      >
                        {statementLabel(inv.statement_status)}
                      </span>
                    </td>
                    <td className={styles.cellAmount}>{fmtMoney(inv.total_amount)}</td>
                    <td className={styles.cellInvoice}>
                      {issued ? (
                        <div className={styles.invoiceCell}>
                          <span className={styles.invoiceNumberRow}>
                            <span className={styles.mono}>{issued.invoice_number}</span>
                            <span
                              className={`${styles.badge} ${statusBadgeClass(issued.ui_status, styles)}`}
                            >
                              {STATUS_LABELS[issued.ui_status] || issued.ui_status}
                            </span>
                          </span>
                          <span className={styles.invoiceMeta}>
                            Échéance : {fmtDate(issued.due_at)}
                          </span>
                          <span className={styles.invoiceMeta}>
                            Payé : {fmtMoney(issued.amount_paid)} · Solde :{' '}
                            {fmtMoney(issued.balance_due)}
                          </span>
                        </div>
                      ) : (
                        <span className={styles.invoiceMeta}>Non émise</span>
                      )}
                    </td>
                    <td className={styles.cellAction}>
                      <button
                        type="button"
                        className={styles.rowAction}
                        onClick={() => openInvoiceModal(inv.id)}
                      >
                        Ouvrir
                      </button>
                      {issued ? (
                        <Link
                          to={`../factures?issued_id=${issued.id}`}
                          className={styles.rowAction}
                        >
                          Voir la facture
                        </Link>
                      ) : null}
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

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
            <div className={styles.modalHeader}>
              <div>
                <h3 id="pb-invoice-title" className={styles.modalTitle}>
                  {modalInvoice ? resolveCompanyName(modalInvoice) : 'Relevé'}
                </h3>
                {modalInvoice ? (
                  <p className={styles.subtitle}>
                    Relevé #{modalInvoice.id} ·{' '}
                    <span
                      className={`${styles.badge} ${statementBadge(
                        modalInvoice.statement_status
                      )}`}
                    >
                      {statementLabel(modalInvoice.statement_status)}
                    </span>
                  </p>
                ) : null}
              </div>
              <button
                type="button"
                className={styles.modalClose}
                aria-label="Fermer"
                onClick={() => setModalInvoice(null)}
              >
                ×
              </button>
            </div>

            {modalLoading ? (
              <p className={styles.loading}>Chargement…</p>
            ) : modalInvoice ? (
              <>
                {(() => {
                  const supportHoursTotal = supportEntries.reduce((acc, e) => {
                    const h = Number(supportHoursOf(e));
                    return acc + (Number.isFinite(h) ? h : 0);
                  }, 0);
                  const supportHoursLabel = fmtQty(supportHoursTotal);
                  return (
                    <>
                <ol className={styles.steps} aria-label="Parcours facture">
                  <li className={styles.step}>Valider</li>
                  <li className={styles.step}>Générer PDF/QR</li>
                  <li className={styles.step}>Télécharger</li>
                </ol>

                <div className={styles.summaryGrid}>
                  <div className={styles.summaryItem}>
                    <span>Portefeuille (abo)</span>
                    <strong className={styles.summaryStrong}>
                      {fmtMoney(modalInvoice.subscription_amount)}
                    </strong>
                    <em className={styles.summaryMeta}>
                      {modalInvoice.own_portfolio_count ?? 0} course
                      {Number(modalInvoice.own_portfolio_count) === 1 ? '' : 's'}
                    </em>
                  </div>
                  <div className={styles.summaryItem}>
                    <span>Commission LIRIE</span>
                    <strong className={styles.summaryStrong}>
                      {fmtMoney(modalInvoice.commission_amount)}
                    </strong>
                    <em className={styles.summaryMeta}>
                      {modalInvoice.lirie_transport_count ?? 0} transport
                      {Number(modalInvoice.lirie_transport_count) === 1 ? '' : 's'}
                    </em>
                  </div>
                  <div className={styles.summaryItem}>
                    <span>Support</span>
                    <strong className={styles.summaryStrong}>
                      {fmtMoney(modalInvoice.support_amount)}
                    </strong>
                    <em className={styles.summaryMeta}>
                      {supportHoursLabel != null
                        ? `${supportHoursLabel} h déjà saisies`
                        : 'aucune heure saisie'}
                    </em>
                  </div>
                  <div className={`${styles.summaryItem} ${styles.summaryTotal}`}>
                    <span>Total TTC</span>
                    <strong className={styles.summaryStrong}>
                      {fmtMoney(modalInvoice.total_amount)}
                    </strong>
                    <em className={styles.summaryMeta}>
                      {Number(modalInvoice.tax_rate) === 0
                        ? 'TVA non applicable (franchise)'
                        : `TVA ${modalInvoice.tax_rate} %`}
                    </em>
                  </div>
                </div>

                <div className={styles.actions}>
                  <button
                    type="button"
                    className={styles.btn}
                    disabled={
                      modalInvoice.statement_status !== 'CALCULATED' ||
                      !canBillingValidate ||
                      !billingPeriodHasEnded(
                        selectedPeriod?.billing_year,
                        selectedPeriod?.billing_month
                      )
                    }
                    title={
                      !canBillingValidate
                        ? 'Capacité admin.billing.validate requise'
                        : modalInvoice.statement_status === 'NEEDS_REVIEW'
                          ? 'Corrigez les données sources puis recalculez avant de valider'
                          : !billingPeriodHasEnded(
                                selectedPeriod?.billing_year,
                                selectedPeriod?.billing_month
                              )
                            ? 'Le mois n’est pas terminé — validation finale interdite'
                            : undefined
                    }
                    onClick={() => {
                      if (modalInvoice.statement_status === 'NEEDS_REVIEW') {
                        setError(
                          'Ce relevé contient des éléments non résolus. Corrigez les données sources puis recalculez.'
                        );
                        return;
                      }
                      setActionDialog({
                        title: 'Valider le relevé',
                        description: `Confirmer la validation du relevé ${resolveCompanyName(modalInvoice)} pour ${periodLabel} ?`,
                        confirmationLabel: 'Valider le relevé',
                        onConfirm: async () => {
                          try {
                            await validatePlatformBillingInvoice(modalInvoice.id);
                            setInfo(
                              'Relevé validé. Clôturez la période lorsque tous les relevés sont validés, puis émettez la facture.'
                            );
                            openInvoiceModal(modalInvoice.id);
                            if (selectedId) loadInvoices(selectedId);
                            setActionDialog(null);
                          } catch (e) {
                            setError(apiErrorMessage(e));
                          }
                        },
                      });
                    }}
                  >
                    Valider
                  </button>
                  {(() => {
                    const periodLocked = selectedPeriod?.status === 'locked';
                    const statementLocked =
                      modalInvoice.statement_status === 'LOCKED';
                    const monthEnded = billingPeriodHasEnded(
                      selectedPeriod?.billing_year,
                      selectedPeriod?.billing_month
                    );
                    const alreadyIssued = Boolean(
                      modalInvoice.issued_invoice &&
                        modalInvoice.issued_invoice.status !== 'cancelled'
                    );
                    const canIssue =
                      statementLocked &&
                      periodLocked &&
                      monthEnded &&
                      canBillingIssue &&
                      !alreadyIssued;
                    let nextStep = null;
                    if (alreadyIssued) {
                      nextStep = null;
                    } else if (!monthEnded) {
                      nextStep =
                        'Le mois est encore en cours. Calcul provisoire uniquement — pas d’émission.';
                    } else if (!statementLocked || !periodLocked) {
                      nextStep =
                        'Validez tous les relevés, puis clôturez la période avant d’émettre les factures.';
                    } else if (!canBillingIssue) {
                      nextStep = 'Capacité admin.billing.issue requise.';
                    }
                    return (
                      <>
                        <button
                          type="button"
                          className={`${styles.btn} ${styles.btnPrimary}`}
                          disabled={!canIssue}
                          title={
                            !canBillingIssue
                              ? 'Capacité admin.billing.issue requise'
                              : undefined
                          }
                          onClick={() => {
                            setActionDialog({
                              title: 'Émettre la facture',
                              description: [
                                `Entreprise : ${resolveCompanyName(modalInvoice)}`,
                                `Période : ${periodLabel}`,
                                `Montant : ${fmtMoney(modalInvoice.total_amount)}`,
                                'Cette action attribue un numéro définitif et crée une facture légale.',
                              ].join('\n'),
                              confirmationLabel: 'Émettre la facture',
                              onConfirm: async () => {
                                try {
                                  const res = await issuePlatformBillingInvoice(
                                    modalInvoice.id
                                  );
                                  setInfo(
                                    `Facture ${res?.issued_invoice?.invoice_number || ''} émise.`
                                  );
                                  openInvoiceModal(modalInvoice.id);
                                  if (selectedId) loadInvoices(selectedId);
                                  setActionDialog(null);
                                } catch (e) {
                                  setError(apiErrorMessage(e));
                                }
                              },
                            });
                          }}
                        >
                          Émettre la facture
                        </button>
                        {nextStep ? (
                          <p className={styles.subtitle} style={{ flexBasis: '100%' }}>
                            {nextStep}
                          </p>
                        ) : null}
                      </>
                    );
                  })()}
                  {modalInvoice.issued_invoice?.id ? (
                    <button
                      type="button"
                      className={`${styles.btn} ${styles.btnPrimary}`}
                      onClick={async () => {
                        try {
                          await downloadPlatformIssuedInvoicePdf(
                            modalInvoice.issued_invoice.id
                          );
                          setInfo(
                            `PDF : ${modalInvoice.issued_invoice.invoice_number || ''}`
                          );
                        } catch (e) {
                          setError(
                            apiErrorMessage(e) || 'Téléchargement impossible'
                          );
                        }
                      }}
                    >
                      Télécharger PDF
                    </button>
                  ) : null}
                </div>

                {modalInvoice.issued_invoice ? (
                  <p className={styles.subtitle}>
                    Facture <strong>{modalInvoice.issued_invoice.invoice_number}</strong> —{' '}
                    <span
                      className={`${styles.badge} ${statusBadgeClass(
                        modalInvoice.issued_invoice.ui_status,
                        styles
                      )}`}
                    >
                      {STATUS_LABELS[modalInvoice.issued_invoice.ui_status] ||
                        modalInvoice.issued_invoice.ui_status}
                    </span>
                  </p>
                ) : null}

                <div className={styles.modalSection}>
                  <h4 className={styles.sectionLabel}>Détail du relevé</h4>
                  <div className={styles.tableWrap}>
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>Libellé</th>
                          <th className={styles.num}>Qté</th>
                          <th className={styles.num}>P.U. / taux</th>
                          <th className={styles.num}>Montant</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(modalInvoice.lines || []).map((ln) => (
                          <tr key={ln.id}>
                            <td>{ln.label || ln.line_type || '—'}</td>
                            <td className={styles.num}>{fmtQty(ln.quantity) ?? '—'}</td>
                            <td className={styles.num}>{fmtUnit(ln)}</td>
                            <td className={styles.num}>{fmtMoney(ln.amount)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className={styles.supportBox}>
                  <h4 className={styles.supportTitle}>Support plateforme</h4>
                  <p className={styles.subtitle}>
                    Les heures déjà saisies sont incluses dans le total ci-dessus. Tarif
                    contrat :{' '}
                    {companySupportRates[modalInvoice.company_id]
                      ? `${companySupportRates[modalInvoice.company_id]} CHF/h`
                      : 'à renseigner'}
                    .
                  </p>

                  {supportEntries.length > 0 ? (
                    <div className={styles.tableWrap} style={{ marginTop: '0.55rem' }}>
                      <table className={styles.table}>
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Description</th>
                            <th className={styles.num}>Heures</th>
                            <th className={styles.num}>Tarif</th>
                            <th className={styles.num}>Montant</th>
                            {statementEditable ? <th className={styles.num}> </th> : null}
                          </tr>
                        </thead>
                        <tbody>
                          {supportEntries.slice(0, 8).map((e) => {
                            const h = supportHoursOf(e);
                            const isEditing = editingSupportId === e.id;
                            return (
                              <tr
                                key={e.id}
                                className={isEditing ? styles.supportRowEditing : undefined}
                              >
                                <td className={styles.mono}>
                                  {e.occurred_at
                                    ? new Date(e.occurred_at).toLocaleDateString('fr-CH')
                                    : '—'}
                                </td>
                                <td>
                                  {SUPPORT_CATEGORIES.find((c) => c.value === e.category)
                                    ?.label || e.category || '—'}
                                  {e.category === 'other' && e.description
                                    ? ` — ${e.description}`
                                    : ''}
                                </td>
                                <td className={styles.num}>{h != null ? `${h} h` : '—'}</td>
                                <td className={styles.num}>
                                  {fmtMoney(e.hourly_rate_snapshot)}
                                </td>
                                <td className={styles.num}>{fmtMoney(e.amount)}</td>
                                {statementEditable ? (
                                  <td className={styles.supportRowActions}>
                                    <button
                                      type="button"
                                      className={styles.linkBtn}
                                      disabled={supportSaving}
                                      onClick={() => startEditSupport(e)}
                                    >
                                      Corriger
                                    </button>
                                    <button
                                      type="button"
                                      className={styles.linkBtnDanger}
                                      disabled={supportSaving}
                                      onClick={() => onDeleteSupport(e)}
                                    >
                                      Supprimer
                                    </button>
                                  </td>
                                ) : null}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className={styles.subtitle}>Aucune heure enregistrée pour l’instant.</p>
                  )}

                  {statementEditable ? (
                    <>
                      <p className={styles.supportAddLabel}>
                        {editingSupportId
                          ? 'Corriger l’entrée sélectionnée'
                          : 'Ajouter des heures'}
                      </p>
                      <div className={styles.supportForm}>
                        <label>
                          Heures
                          <input
                            type="number"
                            min="0.25"
                            step="0.25"
                            value={supportHours}
                            onChange={(e) => setSupportHours(e.target.value)}
                            placeholder="ex. 1.5"
                          />
                        </label>
                        <label>
                          Tarif CHF/h
                          <input
                            type="text"
                            value={supportRate}
                            onChange={(e) => setSupportRate(e.target.value)}
                            placeholder="ex. 45"
                          />
                        </label>
                        <label>
                          Description
                          <select
                            value={supportCategory}
                            onChange={(e) => {
                              setSupportCategory(e.target.value);
                              if (e.target.value !== 'other') setSupportDesc('');
                            }}
                          >
                            {SUPPORT_CATEGORIES.map((c) => (
                              <option key={c.value} value={c.value}>
                                {c.label}
                              </option>
                            ))}
                          </select>
                        </label>
                        {supportCategory === 'other' ? (
                          <label className={styles.supportDesc}>
                            Précision
                            <input
                              type="text"
                              value={supportDesc}
                              onChange={(e) => setSupportDesc(e.target.value)}
                              placeholder="Précisez…"
                              required
                            />
                          </label>
                        ) : null}
                        <button
                          type="button"
                          className={`${styles.btn} ${styles.btnPrimary}`}
                          disabled={
                            supportSaving ||
                            !supportHours ||
                            (supportCategory === 'other' && !supportDesc.trim())
                          }
                          onClick={onAddSupportHours}
                        >
                          {supportSaving
                            ? '…'
                            : editingSupportId
                              ? 'Enregistrer'
                              : 'Ajouter'}
                        </button>
                        {editingSupportId ? (
                          <button
                            type="button"
                            className={styles.btn}
                            disabled={supportSaving}
                            onClick={resetSupportForm}
                          >
                            Annuler
                          </button>
                        ) : null}
                      </div>
                    </>
                  ) : (
                    <div style={{ marginTop: '0.55rem' }}>
                      <p className={styles.subtitle}>
                        Saisie verrouillée. Réouvrez pour corriger (annule la facture non
                        envoyée).
                      </p>
                      <button
                        type="button"
                        className={styles.btn}
                        onClick={() => {
                          setActionDialog({
                            title: 'Réouvrir le relevé',
                            description:
                              'Réouvrir ce relevé ? La facture non envoyée sera annulée.',
                            confirmationLabel: 'Réouvrir',
                            danger: true,
                            onConfirm: async () => {
                              const res = await reopenPlatformBillingInvoice(
                                modalInvoice.id
                              );
                              setInfo(
                                `Relevé réouvert — ${
                                  res?.recalculate?.invoices_generated ?? '—'
                                } régénéré(s).`
                              );
                              setModalInvoice(null);
                              if (selectedId) await loadInvoices(selectedId);
                              setActionDialog(null);
                            },
                          });
                        }}
                      >
                        Réouvrir
                      </button>
                    </div>
                  )}
                </div>
                    </>
                  );
                })()}
              </>
            ) : null}
          </div>
        </div>
      ) : null}

      {actionDialog ? (
        <AdminActionDialog
          open
          title={actionDialog.title}
          description={actionDialog.description}
          confirmationLabel={actionDialog.confirmationLabel}
          danger={Boolean(actionDialog.danger)}
          onConfirm={actionDialog.onConfirm}
          onClose={() => setActionDialog(null)}
        />
      ) : null}
    </div>
  );
};

export default AdminPlatformBilling;
