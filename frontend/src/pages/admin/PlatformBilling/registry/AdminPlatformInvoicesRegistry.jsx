import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useSearchParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import { toast } from 'sonner';
import {
  createPlatformBillingPeriod,
  exportPlatformBillingDossiers,
  fetchPlatformBillingDossiers,
  fetchPlatformBillingPeriods,
  issuePlatformBillingInvoice,
  issueReadyPlatformBillingPeriod,
  lockPlatformBillingPeriod,
  recalculatePlatformBillingCompany,
  recalculatePlatformBillingPeriod,
  downloadPlatformIssuedInvoicePdf,
  sendPlatformIssuedInvoice,
  validatePlatformBillingInvoice,
} from '../../../../services/adminService';
import {
  ACTION_LABELS,
  OPERATIONAL_STATUS_LABELS,
  fmtDate,
  fmtMoney,
  groupAllowedActions,
  operationalBadgeClass,
} from '../dossierInvoiceUi';
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

const CHIP_FILTERS = [
  { id: 'a_traiter', label: 'À traiter' },
  { id: '', label: 'Toutes' },
  { id: 'A_ENVOYER', label: 'À envoyer' },
  { id: 'OVERDUE', label: 'En retard' },
  { id: 'PAID', label: 'Payées' },
];

const apiErrorMessage = (e) =>
  e?.response?.data?.message ||
  e?.response?.data?.error ||
  e?.message ||
  'Erreur';

const AdminPlatformInvoicesRegistry = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const menuRef = useRef(null);
  const rowMenuRef = useRef(null);
  const [rowMenu, setRowMenu] = useState(null); // { key, top, left, row }
  const [periodMenuOpen, setPeriodMenuOpen] = useState(false);

  const [filters, setFilters] = useState(() => ({
    q: searchParams.get('q') || '',
    chip: searchParams.get('chip') || 'a_traiter',
    year: searchParams.get('year') || '',
    month: searchParams.get('month') || '',
    page: Number(searchParams.get('page')) || 1,
    per_page: Number(searchParams.get('per_page')) || 50,
  }));

  const selectedDossier = searchParams.get('dossier') || null;
  /** Snapshot : le dossier peut sortir du filtre courant (ex. payé hors « à traiter »). */
  const [dossierSnapshot, setDossierSnapshot] = useState(null);

  useEffect(() => {
    const next = new URLSearchParams();
    if (filters.q) next.set('q', filters.q);
    if (filters.chip) next.set('chip', filters.chip);
    if (filters.year) next.set('year', filters.year);
    if (filters.month) next.set('month', filters.month);
    if (filters.page > 1) next.set('page', String(filters.page));
    if (filters.per_page !== 50) next.set('per_page', String(filters.per_page));
    if (selectedDossier) next.set('dossier', selectedDossier);
    setSearchParams(next, { replace: true });
  }, [filters, selectedDossier, setSearchParams]);

  useEffect(() => {
    const onDoc = (e) => {
      const t = e.target;
      if (menuRef.current?.contains(t)) return;
      if (rowMenuRef.current?.contains(t)) return;
      setRowMenu(null);
      setPeriodMenuOpen(false);
    };
    const onScroll = () => setRowMenu(null);
    document.addEventListener('mousedown', onDoc);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  const openRowMenu = (event, row) => {
    event.stopPropagation();
    const rect = event.currentTarget.getBoundingClientRect();
    const menuWidth = 224;
    const left = Math.min(
      Math.max(8, rect.right - menuWidth),
      window.innerWidth - menuWidth - 8
    );
    setRowMenu((prev) =>
      prev?.key === row.dossier_key
        ? null
        : {
            key: row.dossier_key,
            top: rect.bottom + 4,
            left,
            row,
          }
    );
  };

  const apiParams = useMemo(() => {
    const params = {
      page: filters.page,
      per_page: filters.per_page,
    };
    if (filters.q) params.q = filters.q;
    if (filters.year) params.year = Number(filters.year);
    if (filters.month) params.month = Number(filters.month);
    if (filters.chip === 'a_traiter') params.a_traiter = 1;
    else if (filters.chip) params.operational_status = filters.chip;
    return params;
  }, [filters]);

  const dossiersQuery = useQuery({
    queryKey: ['admin', 'platform-dossiers', apiParams],
    queryFn: () => fetchPlatformBillingDossiers(apiParams),
    placeholderData: keepPreviousData,
  });

  const periodsQuery = useQuery({
    queryKey: ['admin', 'platform-billing-periods', { with_activity: 1 }],
    queryFn: () => fetchPlatformBillingPeriods({ with_activity: 1 }),
  });

  const periods = useMemo(() => {
    const raw = periodsQuery.data?.periods || periodsQuery.data || [];
    return Array.isArray(raw) ? raw : [];
  }, [periodsQuery.data]);

  const selectedPeriod = useMemo(() => {
    if (!filters.year || !filters.month) return null;
    return periods.find(
      (p) =>
        Number(p.billing_year) === Number(filters.year) &&
        Number(p.billing_month) === Number(filters.month)
    );
  }, [periods, filters.year, filters.month]);

  const items = useMemo(
    () => dossiersQuery.data?.items || [],
    [dossiersQuery.data?.items]
  );
  const stats = dossiersQuery.data?.stats || {};
  const pagination = dossiersQuery.data?.pagination || {};

  useEffect(() => {
    if (!selectedDossier) {
      setDossierSnapshot(null);
      return;
    }
    const row = items.find((i) => i.dossier_key === selectedDossier);
    if (row) setDossierSnapshot(row);
  }, [items, selectedDossier]);

  const selectedDossierRow =
    items.find((i) => i.dossier_key === selectedDossier) ||
    (dossierSnapshot?.dossier_key === selectedDossier ? dossierSnapshot : null);

  const invalidate = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['admin', 'platform-dossiers'] });
    queryClient.invalidateQueries({ queryKey: ['admin', 'platform-billing-periods'] });
  }, [queryClient]);

  const openDossier = (key) => {
    const next = new URLSearchParams(searchParams);
    if (key) {
      next.set('dossier', key);
      const row = items.find((i) => i.dossier_key === key);
      if (row) setDossierSnapshot(row);
    } else {
      next.delete('dossier');
      setDossierSnapshot(null);
    }
    setSearchParams(next, { replace: true });
  };

  const closeDossier = () => openDossier(null);

  const ensurePeriod = async () => {
    if (!filters.year || !filters.month) {
      throw new Error('Sélectionnez un mois pour cette action');
    }
    if (selectedPeriod?.id) return selectedPeriod.id;
    const created = await createPlatformBillingPeriod(
      Number(filters.year),
      Number(filters.month)
    );
    await periodsQuery.refetch();
    return created.id;
  };

  const runPrimary = async (row) => {
    const action = row.primary_action;
    try {
      if (action === 'VIEW' || action === 'REVIEW' || action === 'VIEW_CREDIT_NOTE') {
        openDossier(row.dossier_key);
        return;
      }
      if (action === 'RECALCULATE_DOSSIER') {
        await recalculatePlatformBillingCompany(row.period_id, row.company_id);
        toast.success('Dossier recalculé');
        invalidate();
        return;
      }
      if (action === 'ISSUE' && row.statement_id) {
        await issuePlatformBillingInvoice(row.statement_id);
        toast.success('Facture émise');
        invalidate();
        openDossier(row.dossier_key);
        return;
      }
      if (action === 'MARK_SENT' && row.primary_invoice_id) {
        await sendPlatformIssuedInvoice(row.primary_invoice_id);
        toast.success('Facture marquée comme envoyée');
        invalidate();
        return;
      }
      if (action === 'RECORD_PAYMENT' || action === 'VIEW_PAYMENTS') {
        openDossier(row.dossier_key);
        return;
      }
      openDossier(row.dossier_key);
    } catch (e) {
      toast.error(apiErrorMessage(e));
    }
  };

  const runSecondary = async (row, action) => {
    setRowMenu(null);
    if (action === row.primary_action) {
      await runPrimary(row);
      return;
    }
    try {
      if (action === 'RECALCULATE_DOSSIER') {
        await recalculatePlatformBillingCompany(row.period_id, row.company_id);
        toast.success('Dossier recalculé');
        invalidate();
        return;
      }
      if (action === 'ISSUE' && row.statement_id) {
        await issuePlatformBillingInvoice(row.statement_id);
        toast.success('Facture émise');
        invalidate();
        return;
      }
      if (action === 'MARK_SENT' && row.primary_invoice_id) {
        await sendPlatformIssuedInvoice(row.primary_invoice_id);
        toast.success('Facture marquée comme envoyée');
        invalidate();
        return;
      }
      if (action === 'DOWNLOAD_PDF' && row.primary_invoice_id) {
        await downloadPlatformIssuedInvoicePdf(row.primary_invoice_id);
        return;
      }
      if (action === 'REVIEW' && row.statement_id) {
        await validatePlatformBillingInvoice(row.statement_id);
        toast.success('Relevé validé');
        invalidate();
        return;
      }
      // Les autres actions se font dans le drawer
      openDossier(row.dossier_key);
    } catch (e) {
      toast.error(apiErrorMessage(e));
    }
  };

  const periodMut = useMutation({
    mutationFn: async (kind) => {
      const periodId = await ensurePeriod();
      if (kind === 'recalculate') return recalculatePlatformBillingPeriod(periodId);
      if (kind === 'lock') return lockPlatformBillingPeriod(periodId);
      if (kind === 'issue-ready') return issueReadyPlatformBillingPeriod(periodId);
      return null;
    },
    onSuccess: (data, kind) => {
      if (kind === 'issue-ready') {
        const n = data?.issued?.length || 0;
        const f = data?.failed?.length || 0;
        toast.success(`Émission : ${n} ok, ${f} échec(s)`);
      } else if (kind === 'lock') toast.success('Période verrouillée');
      else if (kind === 'recalculate') toast.success('Période recalculée');
      invalidate();
      setPeriodMenuOpen(false);
    },
    onError: (e) => toast.error(apiErrorMessage(e)),
  });

  const onExport = async () => {
    try {
      await exportPlatformBillingDossiers(apiParams);
      toast.success('Export CSV téléchargé');
    } catch (e) {
      toast.error(apiErrorMessage(e));
    }
  };

  const periodOptions = useMemo(() => {
    const byKey = new Map();
    periods.forEach((p) => {
      const year = Number(p.billing_year);
      const month = Number(p.billing_month);
      if (!year || !month) return;
      // Uniquement périodes avec activité (relevé ou facture), sauf si
      // l'API a déjà filtré with_activity.
      if (
        p.has_billing_activity === false &&
        Number(p.statement_count || 0) === 0 &&
        Number(p.issued_count || 0) === 0
      ) {
        return;
      }
      byKey.set(`${year}-${month}`, {
        year,
        month,
        status: p.status,
        id: p.id,
        pending: false,
      });
    });
    // Mois civil courant : proposé s'il n'est pas déjà listé (pour l'ouvrir)
    const now = new Date();
    const cy = now.getFullYear();
    const cm = now.getMonth() + 1;
    const currentKey = `${cy}-${cm}`;
    if (!byKey.has(currentKey)) {
      byKey.set(currentKey, {
        year: cy,
        month: cm,
        status: null,
        id: null,
        pending: true,
      });
    }
    return Array.from(byKey.values()).sort((a, b) => {
      if (a.year !== b.year) return b.year - a.year;
      return b.month - a.month;
    });
  }, [periods]);

  const periodSelected = Boolean(filters.year && filters.month);
  const periodStatusLabel = !periodSelected
    ? null
    : selectedPeriod
      ? selectedPeriod.status === 'locked'
        ? 'Verrouillée'
        : 'Ouverte'
      : 'Non créée';

  const formatPeriodOption = (year, month, pending) => {
    const label = MONTHS_FR[month - 1] || String(month);
    const title = `${label.charAt(0).toUpperCase()}${label.slice(1)} ${year}`;
    return pending ? `${title} (à ouvrir)` : title;
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Factures</h1>
        </div>
        <div className={styles.headerActions} ref={menuRef}>
          <button type="button" className={styles.btn} onClick={() => invalidate()}>
            Actualiser
          </button>
          <button type="button" className={styles.btn} onClick={onExport}>
            Exporter CSV
          </button>
          {periodSelected && (
            <div className={styles.menuWrap}>
              <button
                type="button"
                className={styles.btn}
                onClick={() => setPeriodMenuOpen((v) => !v)}
              >
                Actions de période ▾
              </button>
              {periodMenuOpen && (
                <div className={styles.dropdown}>
                  <button
                    type="button"
                    className={styles.dropdownItem}
                    onClick={() => periodMut.mutate('recalculate')}
                    disabled={periodMut.isPending}
                  >
                    Recalculer les dossiers
                  </button>
                  <button
                    type="button"
                    className={styles.dropdownItem}
                    onClick={() => periodMut.mutate('lock')}
                    disabled={periodMut.isPending}
                  >
                    Verrouiller la période
                  </button>
                  <button
                    type="button"
                    className={styles.dropdownItem}
                    onClick={() => periodMut.mutate('issue-ready')}
                    disabled={periodMut.isPending}
                  >
                    Émettre toutes les factures prêtes
                  </button>
                  <button type="button" className={styles.dropdownItem} onClick={onExport}>
                    Exporter les données comptables
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </header>

      <div className={styles.filters}>
        <label className={styles.filterField}>
          <span>Période</span>
          <select
            value={filters.year && filters.month ? `${filters.year}-${filters.month}` : ''}
            onChange={(e) => {
              const v = e.target.value;
              if (!v) {
                setFilters((f) => ({ ...f, year: '', month: '', page: 1 }));
                return;
              }
              const [y, m] = v.split('-');
              setFilters((f) => ({ ...f, year: y, month: m, page: 1 }));
            }}
          >
            <option value="">Toutes les périodes</option>
            {periodOptions.map((p) => (
              <option key={`${p.year}-${p.month}`} value={`${p.year}-${p.month}`}>
                {formatPeriodOption(p.year, p.month, p.pending)}
              </option>
            ))}
          </select>
        </label>
        {periodStatusLabel && (
          <span className={styles.periodChip}>État : {periodStatusLabel}</span>
        )}
        <label className={styles.filterFieldGrow}>
          <span>Recherche</span>
          <input
            type="search"
            placeholder="Rechercher une entreprise ou une facture…"
            value={filters.q}
            onChange={(e) => setFilters((f) => ({ ...f, q: e.target.value, page: 1 }))}
          />
        </label>
      </div>

      <div className={styles.chips}>
        {CHIP_FILTERS.map((c) => (
          <button
            key={c.id || 'all'}
            type="button"
            className={filters.chip === c.id ? styles.chipActive : styles.chip}
            onClick={() => setFilters((f) => ({ ...f, chip: c.id, page: 1 }))}
          >
            {c.label}
          </button>
        ))}
      </div>

      <div className={styles.kpis}>
        <div className={styles.kpi}>
          <span className={styles.kpiLabel}>Dossiers</span>
          <strong>{stats.dossiers_count ?? 0}</strong>
        </div>
        <div className={styles.kpi}>
          <span className={styles.kpiLabel}>À émettre</span>
          <strong>{fmtMoney(stats.a_emettre)}</strong>
        </div>
        <div className={styles.kpi}>
          <span className={styles.kpiLabel}>Facturé net</span>
          <strong>{fmtMoney(stats.facture_net)}</strong>
        </div>
        <div className={styles.kpi}>
          <span className={styles.kpiLabel}>Encaissé</span>
          <strong>{fmtMoney(stats.encaisse)}</strong>
        </div>
        <div className={styles.kpi}>
          <span className={styles.kpiLabel}>Solde ouvert</span>
          <strong>{fmtMoney(stats.solde_ouvert)}</strong>
        </div>
      </div>

      {dossiersQuery.isError && (
        <p className={styles.error}>{apiErrorMessage(dossiersQuery.error)}</p>
      )}

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Entreprise / période</th>
              <th>Composition</th>
              <th>Facture</th>
              <th>Montant</th>
              <th>Échéance</th>
              <th>Paiement</th>
              <th>État</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {dossiersQuery.isLoading && (
              <tr>
                <td colSpan={8}>Chargement…</td>
              </tr>
            )}
            {!dossiersQuery.isLoading && items.length === 0 && (
              <tr>
                <td colSpan={8}>Aucun dossier</td>
              </tr>
            )}
            {items.map((row) => {
              const groups = groupAllowedActions(
                row.allowed_actions || [],
                row.primary_action,
                { rowMenuOnly: true }
              );
              return (
                <tr
                  key={row.dossier_key}
                  className={styles.rowClickable}
                  onClick={() => openDossier(row.dossier_key)}
                >
                  <td>
                    <div className={styles.primaryCell}>
                      {row.company_name}
                      <span className={styles.muted}> · {row.period_label}</span>
                    </div>
                  </td>
                  <td className={styles.muted}>{row.composition?.summary || '—'}</td>
                  <td>{row.invoice_number || 'Non émise'}</td>
                  <td>{fmtMoney(row.amount)}</td>
                  <td>
                    {fmtDate(row.due_at)}
                    {row.operational_status === 'OVERDUE' && row.due_at && (
                      <div className={styles.warn}>En retard</div>
                    )}
                  </td>
                  <td>
                    {row.primary_invoice_id
                      ? `${fmtMoney(row.amount_paid)} / ${fmtMoney(row.amount)}`
                      : '—'}
                    {row.operational_status === 'PARTIALLY_PAID' && (
                      <div className={styles.muted}>
                        Solde : {fmtMoney(row.balance_due)}
                      </div>
                    )}
                  </td>
                  <td>
                    <span
                      className={operationalBadgeClass(
                        row.operational_status,
                        styles
                      )}
                    >
                      {OPERATIONAL_STATUS_LABELS[row.operational_status] ||
                        row.operational_status}
                    </span>
                  </td>
                  <td onClick={(e) => e.stopPropagation()}>
                    <div className={styles.actionCell}>
                      <button
                        type="button"
                        className={styles.btnPrimary}
                        onClick={() => runPrimary(row)}
                      >
                        {ACTION_LABELS[row.primary_action] || row.primary_action}
                      </button>
                      {groups.length > 0 && (
                        <div className={styles.menuWrap}>
                          <button
                            type="button"
                            className={styles.btnGhost}
                            aria-label="Actions secondaires"
                            aria-expanded={rowMenu?.key === row.dossier_key}
                            onClick={(e) => openRowMenu(e, row)}
                          >
                            •••
                          </button>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {rowMenu &&
        createPortal(
          <div
            ref={rowMenuRef}
            className={styles.dropdownFixed}
            style={{ top: rowMenu.top, left: rowMenu.left }}
            role="menu"
          >
            {groupAllowedActions(
              rowMenu.row.allowed_actions || [],
              rowMenu.row.primary_action,
              { rowMenuOnly: true }
            ).map((g) => (
              <div key={g.id} className={styles.dropdownGroup}>
                <div className={styles.dropdownGroupLabel}>{g.label}</div>
                {g.items.map((a) => (
                  <button
                    key={a}
                    type="button"
                    className={styles.dropdownItem}
                    role="menuitem"
                    onClick={() => runSecondary(rowMenu.row, a)}
                  >
                    {ACTION_LABELS[a] || a}
                  </button>
                ))}
              </div>
            ))}
          </div>,
          document.body
        )}

      {pagination.pages > 1 && (
        <div className={styles.pagination}>
          <button
            type="button"
            className={styles.btn}
            disabled={filters.page <= 1}
            onClick={() => setFilters((f) => ({ ...f, page: f.page - 1 }))}
          >
            Précédent
          </button>
          <span>
            Page {pagination.page} / {pagination.pages}
          </span>
          <button
            type="button"
            className={styles.btn}
            disabled={filters.page >= pagination.pages}
            onClick={() => setFilters((f) => ({ ...f, page: f.page + 1 }))}
          >
            Suivant
          </button>
        </div>
      )}

      {selectedDossier && (
        <AdminPlatformInvoiceSheet
          dossierKey={selectedDossier}
          dossierRow={selectedDossierRow}
          issuedId={selectedDossierRow?.primary_invoice_id || null}
          statementId={selectedDossierRow?.statement_id || null}
          onClose={closeDossier}
          onChanged={invalidate}
        />
      )}
    </div>
  );
};

export default AdminPlatformInvoicesRegistry;
