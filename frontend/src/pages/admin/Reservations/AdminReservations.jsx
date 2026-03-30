import React, { useCallback, useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { fetchAdminBookings, downloadAdminBookingsExport } from '../../../services/adminService';
import styles from './AdminReservations.module.css';

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

  const base = `/dashboard/admin/${adminId}`;

  return (
    <main className={styles.content}>
      <header className={styles.header}>
        <h1>Réservations — supervision plateforme</h1>
        <p>
          Recherche, filtres et synthèse sur le jeu de résultats serveur. Les libellés de statut
          proviennent de l&apos;API.
        </p>
      </header>

      <section className={styles.toolbar} aria-label="Filtres">
        <div className={styles.toolbarRow}>
          <label className={styles.field}>
            <span>Recherche</span>
            <input
              type="search"
              placeholder="ID, nom, lieu…"
              defaultValue={searchParams.get('q') || ''}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  setFacet({ q: e.target.value.trim() });
                }
              }}
            />
          </label>
          <label className={styles.field}>
            <span>Statut</span>
            <select
              className={styles.fieldSelect}
              value={statusParam}
              onChange={(e) => setFacet({ status: e.target.value })}
              aria-label="Filtrer par statut de réservation"
            >
              <option value="">Tous les statuts</option>
              {statusCustomUrl ? (
                <option value={statusParam}>Filtre actuel ({statusParam})</option>
              ) : null}
              {BOOKING_STATUS_OPTIONS.map(({ value, label }) => (
                <option key={value} value={value}>
                  {label} ({value})
                </option>
              ))}
            </select>
          </label>
          <label className={styles.field}>
            <span>Création du</span>
            <input
              type="date"
              defaultValue={searchParams.get('created_from') || ''}
              onChange={(e) => setFacet({ created_from: e.target.value })}
            />
          </label>
          <label className={styles.field}>
            <span>Création au</span>
            <input
              type="date"
              defaultValue={searchParams.get('created_to') || ''}
              onChange={(e) => setFacet({ created_to: e.target.value })}
            />
          </label>
        </div>
        <div className={styles.toolbarRow}>
          <label className={styles.field}>
            <span>Transport du</span>
            <input
              type="date"
              defaultValue={searchParams.get('scheduled_from') || ''}
              onChange={(e) => setFacet({ scheduled_from: e.target.value })}
            />
          </label>
          <label className={styles.field}>
            <span>Transport au</span>
            <input
              type="date"
              defaultValue={searchParams.get('scheduled_to') || ''}
              onChange={(e) => setFacet({ scheduled_to: e.target.value })}
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
          <button type="button" className={styles.btnGhost} onClick={resetFilters}>
            Réinitialiser les filtres
          </button>
          <button
            type="button"
            className={styles.btnGhost}
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
            {exporting ? 'Export…' : 'Export CSV'}
          </button>
        </div>
      </section>

      {summary ? (
        <section className={styles.metricsRow} aria-label="Synthèse filtrée">
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
        </section>
      ) : null}

      {loading && (
        <div className={styles.feedbackCard}>
          <p>Chargement…</p>
        </div>
      )}
      {error && (
        <div className={styles.feedbackCard}>
          <p className={styles.error}>Erreur : {error}</p>
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
                    <td>{row.scheduled_at ? new Date(row.scheduled_at).toLocaleString('fr-CH') : '—'}</td>
                    <td>{row.client_name ?? '—'}</td>
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
            disabled={pagination.page <= 1}
            onClick={() => goPage(pagination.page - 1)}
          >
            Précédent
          </button>
          <span>
            Page {pagination.page} / {pagination.total_pages} ({pagination.total_items} résultats)
          </span>
          <button
            type="button"
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
