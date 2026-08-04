import React, { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import {
  fetchPlatformIssuedInvoices,
  exportPlatformIssuedInvoices,
} from '../../../../services/adminService';
import { useAdminCapabilities } from '../../../../hooks/useAdminCapabilities';
import { STATUS_LABELS, statusBadgeClass, fmtMoney, fmtDate } from '../issuedInvoiceUi';
import AdminPlatformInvoiceSheet from './AdminPlatformInvoiceSheet';
import styles from './AdminPlatformInvoicesRegistry.module.css';

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

const STATUS_OPTIONS = [
  { value: '', label: 'Tous les statuts' },
  { value: 'ISSUED', label: STATUS_LABELS.ISSUED },
  { value: 'SENT', label: STATUS_LABELS.SENT },
  { value: 'PARTIALLY_PAID', label: STATUS_LABELS.PARTIALLY_PAID },
  { value: 'PAID', label: STATUS_LABELS.PAID },
  { value: 'OVERDUE', label: STATUS_LABELS.OVERDUE },
  { value: 'CANCELLED', label: STATUS_LABELS.CANCELLED },
  { value: 'CREDITED', label: STATUS_LABELS.CREDITED },
];

const apiErrorMessage = (e) =>
  e?.response?.data?.message ||
  e?.response?.data?.error ||
  e?.message ||
  'Erreur';

const buildApiParams = (filters) => {
  const params = {
    page: filters.page,
    per_page: filters.per_page,
    sort_by: 'issued_at',
    sort_order: 'desc',
  };
  if (filters.q) params.q = filters.q;
  if (filters.year) params.year = filters.year;
  if (filters.month) params.month = filters.month;
  if (filters.with_balance) params.with_balance = 1;
  if (filters.overdue_only) params.overdue_only = 1;
  if (filters.status === 'PARTIALLY_PAID') {
    params.payment_state = 'PARTIAL';
  } else if (filters.status === 'OVERDUE') {
    params.overdue_only = 1;
  } else if (filters.status) {
    params.status = filters.status;
  }
  return params;
};

const DEFAULT_FILTERS = {
  q: '',
  status: '',
  year: '',
  month: '',
  with_balance: false,
  overdue_only: false,
  per_page: 20,
  page: 1,
};

const AdminPlatformInvoicesRegistry = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    canBillingSend,
    canBillingPayment,
    canBillingDueDate,
    canBillingCancel,
    canBillingCredit,
  } = useAdminCapabilities();

  const [filters, setFilters] = useState(() => ({
    q: searchParams.get('q') || DEFAULT_FILTERS.q,
    status: searchParams.get('status') || DEFAULT_FILTERS.status,
    year: searchParams.get('year') || DEFAULT_FILTERS.year,
    month: searchParams.get('month') || DEFAULT_FILTERS.month,
    with_balance: searchParams.get('with_balance') === '1',
    overdue_only: searchParams.get('overdue_only') === '1',
    per_page: Number(searchParams.get('per_page')) || DEFAULT_FILTERS.per_page,
    page: Number(searchParams.get('page')) || DEFAULT_FILTERS.page,
  }));
  const [selectedId, setSelectedId] = useState(() => {
    const raw = searchParams.get('issued_id');
    const n = Number(raw);
    return raw && Number.isFinite(n) ? n : null;
  });
  const [exportError, setExportError] = useState(null);
  const [exporting, setExporting] = useState(false);

  // Filtres + drawer -> URL (source de vérité = état local)
  useEffect(() => {
    const p = new URLSearchParams();
    if (filters.q) p.set('q', filters.q);
    if (filters.status) p.set('status', filters.status);
    if (filters.year) p.set('year', String(filters.year));
    if (filters.month) p.set('month', String(filters.month));
    if (filters.with_balance) p.set('with_balance', '1');
    if (filters.overdue_only) p.set('overdue_only', '1');
    if (filters.per_page !== DEFAULT_FILTERS.per_page) {
      p.set('per_page', String(filters.per_page));
    }
    if (filters.page !== DEFAULT_FILTERS.page) p.set('page', String(filters.page));
    if (selectedId) p.set('issued_id', String(selectedId));
    setSearchParams(p, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, selectedId]);

  const apiParams = useMemo(() => buildApiParams(filters), [filters]);

  const {
    data,
    isLoading,
    isFetching,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['admin', 'platform-issued-invoices', apiParams],
    queryFn: () => fetchPlatformIssuedInvoices(apiParams),
    placeholderData: keepPreviousData,
    staleTime: 20_000,
  });

  const items = data?.items || [];
  const pagination = data?.pagination || {};
  const stats = data?.stats || {};

  const updateFilters = (patch) => {
    setFilters((prev) => ({
      ...prev,
      ...patch,
      page: 'page' in patch ? patch.page : 1,
    }));
  };

  const onResetFilters = () => setFilters({ ...DEFAULT_FILTERS });

  const openDetail = (id) => setSelectedId(id);
  const closeDetail = () => setSelectedId(null);

  const onExport = async () => {
    setExporting(true);
    setExportError(null);
    try {
      await exportPlatformIssuedInvoices(apiParams);
    } catch (e) {
      setExportError(apiErrorMessage(e));
    } finally {
      setExporting(false);
    }
  };

  const listErrorMessage = isError ? apiErrorMessage(error) : null;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Factures LIRIE aux transporteurs</h1>
          <p className={styles.subtitle}>
            Registre des factures légales émises — suivi des échéances, paiements et avoirs.
          </p>
        </div>
        <div className={styles.headerActions}>
          <Link to="../releves" className={styles.linkBtn}>
            Voir les relevés
          </Link>
          <button
            type="button"
            className={styles.btn}
            disabled={exporting}
            onClick={onExport}
          >
            {exporting ? 'Export…' : 'Export CSV'}
          </button>
        </div>
      </header>

      {exportError ? (
        <div className={`${styles.banner} ${styles.bannerError}`} role="alert">
          {exportError}
        </div>
      ) : null}

      <section className={styles.kpis}>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel}>Total facturé</span>
          <strong className={styles.kpiValue}>{fmtMoney(stats.total_invoiced)}</strong>
        </div>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel}>Payé</span>
          <strong className={styles.kpiValue}>{fmtMoney(stats.total_paid)}</strong>
        </div>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel}>Solde dû</span>
          <strong className={styles.kpiValue}>{fmtMoney(stats.total_balance)}</strong>
        </div>
        <div className={`${styles.kpiCard} ${styles.kpiCardWarn}`}>
          <span className={styles.kpiLabel}>En retard</span>
          <strong className={styles.kpiValue}>
            {stats.overdue_count || 0}
            <span className={styles.kpiSub}> · {fmtMoney(stats.overdue_amount)}</span>
          </strong>
        </div>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel}>Net facturé</span>
          <strong className={styles.kpiValue}>{fmtMoney(stats.net_invoiced)}</strong>
        </div>
      </section>

      <section className={styles.filters}>
        <label className={styles.field}>
          Recherche
          <input
            type="text"
            value={filters.q}
            placeholder="N° facture, entreprise…"
            onChange={(e) => updateFilters({ q: e.target.value })}
          />
        </label>
        <label className={styles.field}>
          Statut
          <select
            value={filters.status}
            onChange={(e) => updateFilters({ status: e.target.value })}
          >
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.field}>
          Année
          <input
            type="number"
            min="2020"
            max="2100"
            value={filters.year}
            placeholder="Année"
            onChange={(e) => updateFilters({ year: e.target.value })}
          />
        </label>
        <label className={styles.field}>
          Mois
          <select
            value={filters.month}
            onChange={(e) => updateFilters({ month: e.target.value })}
          >
            <option value="">Tous</option>
            {MONTHS_FR.map((label, i) => (
              <option key={label} value={i + 1}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.checkboxField}>
          <input
            type="checkbox"
            checked={filters.with_balance}
            onChange={(e) => updateFilters({ with_balance: e.target.checked })}
          />
          Avec solde
        </label>
        <label className={styles.checkboxField}>
          <input
            type="checkbox"
            checked={filters.overdue_only}
            onChange={(e) => updateFilters({ overdue_only: e.target.checked })}
          />
          En retard uniquement
        </label>
        <label className={styles.field}>
          Par page
          <select
            value={filters.per_page}
            onChange={(e) => updateFilters({ per_page: Number(e.target.value) })}
          >
            {[10, 20, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className={styles.btnGhost} onClick={onResetFilters}>
          Réinitialiser
        </button>
      </section>

      {listErrorMessage ? (
        <div className={`${styles.banner} ${styles.bannerError}`} role="alert">
          {listErrorMessage}
          <button type="button" className={styles.retryBtn} onClick={() => refetch()}>
            Réessayer
          </button>
        </div>
      ) : null}

      <section className={styles.panel}>
        <div className={styles.panelHead}>
          <h2 className={styles.panelTitle}>Factures</h2>
          <span className={styles.hint}>
            {pagination.total != null
              ? `${pagination.total} facture${pagination.total > 1 ? 's' : ''}`
              : ''}
            {isFetching && !isLoading ? ' · mise à jour…' : ''}
          </span>
        </div>

        {isLoading ? (
          <p className={styles.loading}>Chargement…</p>
        ) : items.length === 0 ? (
          <div className={styles.empty}>Aucune facture ne correspond aux filtres.</div>
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">N°</th>
                  <th scope="col">Entreprise</th>
                  <th scope="col">Période</th>
                  <th scope="col">Émission</th>
                  <th scope="col">Échéance</th>
                  <th scope="col" className={styles.num}>
                    Montant TTC
                  </th>
                  <th scope="col" className={styles.num}>
                    Payé / solde
                  </th>
                  <th scope="col">Statut</th>
                  <th scope="col">
                    <span className={styles.srOnly}>Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {items.map((inv) => {
                  const period =
                    inv.billing_year && inv.billing_month
                      ? `${String(inv.billing_month).padStart(2, '0')}.${inv.billing_year}`
                      : '—';
                  const isCredit = inv.document_type === 'CREDIT_NOTE';
                  const showPartialSubline =
                    inv.ui_status === 'OVERDUE' && inv.payment_state === 'PARTIAL';
                  return (
                    <tr key={inv.id}>
                      <td className={styles.mono}>
                        {isCredit ? 'Avoir ' : ''}
                        {inv.invoice_number || `#${inv.id}`}
                      </td>
                      <td className={styles.companyName}>{inv.company_name || '—'}</td>
                      <td>{period}</td>
                      <td>{fmtDate(inv.issued_at)}</td>
                      <td>{fmtDate(inv.due_at)}</td>
                      <td className={styles.num}>{fmtMoney(inv.total_amount)}</td>
                      <td className={styles.num}>
                        <div className={styles.paymentCell}>
                          <span>Payé : {fmtMoney(inv.amount_paid)}</span>
                          <span
                            className={
                              Number(inv.balance_due) > 0 ? styles.balanceDue : undefined
                            }
                          >
                            Solde : {fmtMoney(inv.balance_due)}
                          </span>
                        </div>
                      </td>
                      <td>
                        <span
                          className={`${styles.badge} ${statusBadgeClass(
                            inv.ui_status,
                            styles
                          )}`}
                        >
                          {STATUS_LABELS[inv.ui_status] || inv.ui_status || '—'}
                        </span>
                        {showPartialSubline ? (
                          <span className={styles.subline}>
                            Partiellement payée — solde {fmtMoney(inv.balance_due)}
                          </span>
                        ) : null}
                      </td>
                      <td className={styles.cellAction}>
                        <button
                          type="button"
                          className={styles.rowAction}
                          onClick={() => openDetail(inv.id)}
                        >
                          Voir
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {pagination.pages > 1 ? (
          <div className={styles.pagination}>
            <span className={styles.paginationInfo}>
              Page {pagination.page} sur {pagination.pages}
            </span>
            <div className={styles.paginationControls}>
              <button
                type="button"
                disabled={pagination.page <= 1}
                onClick={() => updateFilters({ page: pagination.page - 1 })}
              >
                Précédent
              </button>
              <button
                type="button"
                disabled={pagination.page >= pagination.pages}
                onClick={() => updateFilters({ page: pagination.page + 1 })}
              >
                Suivant
              </button>
            </div>
          </div>
        ) : null}
      </section>

      {selectedId != null ? (
        <AdminPlatformInvoiceSheet
          issuedId={selectedId}
          onClose={closeDetail}
          onChanged={refetch}
          capabilities={{
            canBillingSend,
            canBillingPayment,
            canBillingDueDate,
            canBillingCancel,
            canBillingCredit,
          }}
        />
      ) : null}
    </div>
  );
};

export default AdminPlatformInvoicesRegistry;
