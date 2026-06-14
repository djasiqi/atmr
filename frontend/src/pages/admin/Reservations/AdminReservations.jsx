import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { FiChevronDown, FiDownload, FiRotateCcw, FiSearch, FiTag, FiX } from 'react-icons/fi';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import { fetchAdminBookings, downloadAdminBookingsExport } from '../../../services/adminService';
import rfChipStyles from '../../company/Reservations/components/ReservationFilters.module.css';
import styles from './AdminReservations.module.css';
import shell from '../adminShell.module.css';

/** Même composant que ReservationFilters (entreprise) — styles partagés via rfChipStyles */
function ChipDropdown({ icon, value, options, onChange, activeWhen }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) close();
    };
    const onKey = (e) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, close]);

  const selected = options.find((o) => o.value === value);
  const isActive = activeWhen ? activeWhen(value) : value !== options[0]?.value;

  return (
    <div className={rfChipStyles.chipDrop} ref={ref}>
      <button
        type="button"
        className={`${rfChipStyles.chipBtn} ${isActive ? rfChipStyles.chipBtnActive : ''}`}
        onClick={() => setOpen((p) => !p)}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        {icon}
        <span className={rfChipStyles.chipText}>{selected?.label || '—'}</span>
        <FiChevronDown size={11} className={`${rfChipStyles.chipArrow} ${open ? rfChipStyles.chipArrowOpen : ''}`} />
      </button>
      {open && (
        <div className={rfChipStyles.chipMenu} role="listbox">
          {options.map((o) => (
            <button
              key={`${o.label}-${String(o.value)}`}
              type="button"
              role="option"
              aria-selected={o.value === value}
              className={`${rfChipStyles.chipOption} ${o.value === value ? rfChipStyles.chipOptionActive : ''}`}
              onClick={() => {
                onChange(o.value);
                close();
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const DEFAULT_PER_PAGE = 25;

/** Aligné sur backend/models/enums.py BookingStatus — filtre API `status` (un code ou liste séparée par virgules). */
const BOOKING_STATUS_OPTIONS = [
  { value: 'PENDING', label: 'En attente' },
  { value: 'ACCEPTED', label: 'Acceptée' },
  { value: 'ASSIGNED', label: 'Assignée' },
  { value: 'EN_ROUTE', label: 'En route' },
  { value: 'IN_PROGRESS', label: 'En cours' },
  { value: 'COMPLETED', label: 'Terminée' },
  { value: 'RETURN_COMPLETED', label: 'Retour terminé' },
  { value: 'CANCELED', label: 'Annulée' },
];

const BOOKING_STATUS_SINGLE = new Set(BOOKING_STATUS_OPTIONS.map((o) => o.value));

const statusBadgeClass = (status) => {
  const s = String(status || '').toLowerCase();
  if (s === 'canceled') return styles.statusCancelled;
  if (s === 'completed' || s === 'return_completed') return styles.statusCompleted;
  if (s === 'pending') return styles.statusPending;
  if (s === 'accepted' || s === 'assigned') return styles.statusAssigned;
  if (s === 'en_route' || s === 'in_progress') return styles.statusAccepted;
  return styles.statusDefault;
};

function buildParamsFromSearch(searchParams) {
  const get = (k) => searchParams.get(k) || '';
  const out = {
    page: Math.max(1, parseInt(get('page'), 10) || 1),
    per_page: Math.min(100, Math.max(5, parseInt(get('per_page'), 10) || DEFAULT_PER_PAGE)),
    sort: get('sort') || 'scheduled_time',
    order: get('order') || 'desc',
  };
  const q = get('q').trim();
  if (q) out.q = q;
  const st = get('status').trim();
  if (st) out.status = st;
  ['created_from', 'created_to', 'scheduled_from', 'scheduled_to'].forEach((k) => {
    const v = get(k).trim();
    if (v) out[k] = v;
  });
  const iid = get('institution_id').trim();
  if (iid) out.institution_id = iid;
  const cid = get('company_id').trim();
  if (cid) out.company_id = cid;
  const instQ = get('institution_q').trim();
  if (instQ) out.institution_q = instQ;
  const compQ = get('company_q').trim();
  if (compQ) out.company_q = compQ;
  ['cancelled_only', 'exclude_cancelled', 'with_transfer', 'unassigned', 'incomplete_data', 'needs_investigation'].forEach(
    (k) => {
      const v = searchParams.get(k);
      if (v === 'true' || v === '1') out[k] = true;
      if (v === 'false' || v === '0') out[k] = false;
    }
  );
  return out;
}

const AdminReservations = () => {
  const { public_id: adminId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  const apiParams = useMemo(() => buildParamsFromSearch(searchParams), [searchParams]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchAdminBookings(apiParams);
      setData(payload);
    } catch (err) {
      const message = err?.response?.data?.message || err?.message || 'Erreur inconnue';
      setError(message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [apiParams]);

  React.useEffect(() => {
    load();
  }, [load]);

  const setFacet = (patch) => {
    const next = new URLSearchParams(searchParams);
    Object.entries(patch).forEach(([k, v]) => {
      if (v === undefined || v === null || v === '') next.delete(k);
      else next.set(k, String(v));
    });
    next.set('page', '1');
    setSearchParams(next);
  };

  const resetFilters = () => {
    setSearchParams(new URLSearchParams({ page: '1', per_page: String(DEFAULT_PER_PAGE) }));
  };

  const goPage = (p) => {
    const next = new URLSearchParams(searchParams);
    next.set('page', String(p));
    setSearchParams(next);
  };

  const summary = data?.summary;
  const pagination = data?.pagination;
  const items = data?.items ?? [];

  const statusParam = (searchParams.get('status') || '').trim();
  const statusCustomUrl =
    statusParam &&
    (statusParam.includes(',') || !BOOKING_STATUS_SINGLE.has(statusParam));

  const statusChipOptions = useMemo(() => {
    return [
      { value: '', label: 'Tous les statuts' },
      ...(statusCustomUrl
        ? [{ value: statusParam, label: `Filtre actuel (${statusParam})` }]
        : []),
      ...BOOKING_STATUS_OPTIONS.map(({ value, label }) => ({
        value,
        label: `${label} (${value})`,
      })),
    ];
  }, [statusParam, statusCustomUrl]);

  const qFromUrl = searchParams.get('q') || '';
  const [qDraft, setQDraft] = useState(qFromUrl);
  useEffect(() => {
    setQDraft(qFromUrl);
  }, [qFromUrl]);

  const base = `/dashboard/admin/${adminId}`;

  return (
    <main className={shell.content}>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderLeft}>
          <span className={styles.pageEyebrow}>Supervision plateforme</span>
          <h1 className={styles.pageTitle}>Réservations</h1>
          <p className={styles.pageLead}>
            Recherche, filtres et synthèse côté serveur. Les libellés de statut proviennent de
            l&apos;API.
          </p>
        </div>
      </header>

      <section className={styles.toolbarCard} aria-labelledby="admin-res-filters-title">
        <div className={styles.toolbarHead}>
          <h2 id="admin-res-filters-title" className={styles.toolbarTitle}>
            Filtres
          </h2>
        </div>

        <div className={styles.toolbarRow}>
          <div className={`${styles.field} ${styles.fieldSearch}`}>
            <span>Recherche</span>
            <div className={rfChipStyles.searchWrap}>
              <FiSearch className={rfChipStyles.searchIcon} size={14} aria-hidden />
              <input
                type="text"
                className={rfChipStyles.searchInput}
                placeholder="ID, nom, lieu…"
                value={qDraft}
                onChange={(e) => setQDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    setFacet({ q: qDraft.trim() });
                  }
                }}
                aria-label="Recherche (Entrée pour filtrer)"
              />
              {qDraft ? (
                <button
                  type="button"
                  className={rfChipStyles.clearBtn}
                  onClick={() => {
                    setQDraft('');
                    setFacet({ q: '' });
                  }}
                  title="Effacer"
                  aria-label="Effacer la recherche"
                >
                  <FiX size={12} />
                </button>
              ) : null}
            </div>
          </div>
          <div className={styles.field}>
            <span>Statut</span>
            <ChipDropdown
              icon={<FiTag size={12} aria-hidden />}
              value={statusParam}
              options={statusChipOptions}
              onChange={(v) => setFacet({ status: v })}
              activeWhen={(v) => v !== ''}
            />
          </div>
          <label className={`${styles.field} ${styles.fieldDate}`}>
            <span>Création du</span>
            <InlineDatePicker
              value={searchParams.get('created_from') || ''}
              onChange={(iso) => setFacet({ created_from: iso })}
            />
          </label>
          <label className={`${styles.field} ${styles.fieldDate}`}>
            <span>Création au</span>
            <InlineDatePicker
              value={searchParams.get('created_to') || ''}
              onChange={(iso) => setFacet({ created_to: iso })}
            />
          </label>
        </div>

        <div className={styles.toolbarDivider} aria-hidden />

        <div className={styles.toolbarRow}>
          <label className={`${styles.field} ${styles.fieldDate}`}>
            <span>Transport du</span>
            <InlineDatePicker
              value={searchParams.get('scheduled_from') || ''}
              onChange={(iso) => setFacet({ scheduled_from: iso })}
            />
          </label>
          <label className={`${styles.field} ${styles.fieldDate}`}>
            <span>Transport au</span>
            <InlineDatePicker
              value={searchParams.get('scheduled_to') || ''}
              onChange={(iso) => setFacet({ scheduled_to: iso })}
            />
          </label>
          <label className={styles.field}>
            <span>Institution</span>
            <input
              type="search"
              enterKeyHint="search"
              placeholder="Nom de l'institution…"
              defaultValue={searchParams.get('institution_q') || ''}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  setFacet({
                    institution_q: e.target.value.trim(),
                    institution_id: '',
                  });
                }
              }}
              aria-label="Filtrer par nom d'institution (partie du libellé)"
            />
          </label>
          <label className={styles.field}>
            <span>Entreprise</span>
            <input
              type="search"
              enterKeyHint="search"
              placeholder="Nom de l'entreprise…"
              defaultValue={searchParams.get('company_q') || ''}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  setFacet({
                    company_q: e.target.value.trim(),
                    company_id: '',
                  });
                }
              }}
              aria-label="Filtrer par nom d'entreprise porteuse ou exécutante"
            />
          </label>
        </div>

        <div className={styles.toolbarActions}>
          <button type="button" className={styles.btnReset} onClick={resetFilters}>
            <FiRotateCcw size={15} aria-hidden />
            Réinitialiser
          </button>
          <button
            type="button"
            className={styles.btnExport}
            disabled={exporting}
            onClick={async () => {
              setExporting(true);
              try {
                await downloadAdminBookingsExport(apiParams);
              } catch {
                setError("Export impossible.");
              } finally {
                setExporting(false);
              }
            }}
          >
            <FiDownload size={15} aria-hidden />
            {exporting ? 'Export…' : 'Export CSV'}
          </button>
        </div>
      </section>

      {summary ? (
        <section className={styles.metricsBlock} aria-labelledby="admin-res-metrics-title">
          <h2 id="admin-res-metrics-title" className={styles.metricsBlockTitle}>
            Synthèse filtrée
          </h2>
          <div className={styles.metricsRow} aria-label="Indicateurs sur le jeu filtré">
          <button
            type="button"
            className={styles.metricCard}
            onClick={() => goPage(1)}
            title="Total des résultats pour les filtres actuels"
          >
            <span>Résultats</span>
            <strong>{summary.total}</strong>
          </button>
          <button
            type="button"
            className={styles.metricCard}
            onClick={() => setFacet({ unassigned: 'true' })}
          >
            <span>Non assignées</span>
            <strong>{summary.unassigned}</strong>
          </button>
          <button
            type="button"
            className={styles.metricCard}
            onClick={() => setFacet({ status: 'CANCELED' })}
          >
            <span>Annulées (dans le filtre)</span>
            <strong>{summary.canceled}</strong>
          </button>
          <button
            type="button"
            className={styles.metricCard}
            onClick={() => setFacet({ with_transfer: 'true' })}
          >
            <span>Avec transfert</span>
            <strong>{summary.transferred}</strong>
          </button>
          <button
            type="button"
            className={styles.metricCard}
            onClick={() => setFacet({ incomplete_data: 'true' })}
          >
            <span>Données incomplètes</span>
            <strong>{summary.incomplete_data}</strong>
          </button>
          <button
            type="button"
            className={styles.metricCard}
            onClick={() => setFacet({ needs_investigation: 'true' })}
          >
            <span>À investiguer</span>
            <strong>{summary.needs_investigation}</strong>
          </button>
          </div>
        </section>
      ) : null}

      {loading && (
        <div className={styles.feedbackCard} role="status" aria-live="polite">
          <p className={styles.feedbackText}>Chargement…</p>
        </div>
      )}
      {error && (
        <div className={styles.feedbackCardError} role="alert">
          <p className={styles.error}>{error}</p>
        </div>
      )}

      {!loading && !error && (
        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>ID</th>
                <th>Créée</th>
                <th>Transport</th>
                <th>Client</th>
                <th>Institution</th>
                <th>Entreprise</th>
                <th>Statut</th>
                <th>Transf.</th>
                <th>Montant</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {items.length === 0 ? (
                <tr>
                  <td colSpan="10" className={styles.empty}>
                    Aucune réservation pour ces critères.
                  </td>
                </tr>
              ) : (
                items.map((row) => (
                  <tr key={row.id}>
                    <td>
                      <span className={styles.idBadge}>{row.id}</span>
                    </td>
                    <td>{row.created_at ? new Date(row.created_at).toLocaleString('fr-CH') : '—'}</td>
                    <td>{row.scheduling?.display_datetime || (row.scheduled_at ? new Date(row.scheduled_at).toLocaleString('fr-CH') : '—')}</td>
                    <td>{row.identity?.primary_label || row.client_name || '—'}</td>
                    <td className={styles.locationCell} title={row.institution_name || ''}>
                      {row.institution_name ?? '—'}
                    </td>
                    <td className={styles.locationCell} title={row.current_company_name || ''}>
                      {row.current_company_name ?? '—'}
                    </td>
                    <td>
                      <span className={`${styles.statusBadge} ${statusBadgeClass(row.status)}`}>
                        {row.status_label ?? row.status}
                      </span>
                    </td>
                    <td>{row.has_transfer ? 'Oui' : 'Non'}</td>
                    <td className={styles.amountCell}>
                      {row.amount_chf != null ? `${row.amount_chf} CHF` : '—'}
                    </td>
                    <td>
                      <Link className={styles.detailLink} to={`${base}/reservations/${row.id}`}>
                        Détail
                      </Link>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {pagination && pagination.total_pages > 1 ? (
        <nav className={styles.pagination} aria-label="Pagination">
          <button
            type="button"
            className={styles.paginationBtn}
            disabled={pagination.page <= 1}
            onClick={() => goPage(pagination.page - 1)}
          >
            Précédent
          </button>
          <span className={styles.paginationMeta}>
            Page {pagination.page} / {pagination.total_pages}
            <span className={styles.paginationCount}>({pagination.total_items} résultats)</span>
          </span>
          <button
            type="button"
            className={styles.paginationBtn}
            disabled={pagination.page >= pagination.total_pages}
            onClick={() => goPage(pagination.page + 1)}
          >
            Suivant
          </button>
        </nav>
      ) : null}
    </main>
  );
};

export default AdminReservations;
