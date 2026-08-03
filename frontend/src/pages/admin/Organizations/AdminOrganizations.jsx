import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { FiSearch } from 'react-icons/fi';
import { fetchPartnerOrganizations } from '../../../services/adminService';
import AdminOrganizationDetailDrawer from './AdminOrganizationDetailDrawer';
import styles from './AdminOrganizations.module.css';
import shell from '../adminShell.module.css';

const TYPE_LABELS = {
  company: 'Entreprise de transport',
  institution: 'Institution',
};

const LIFECYCLE_LABELS = {
  active: 'Active',
  onboarding: 'Onboarding',
  suspended: 'Suspendue',
  archived: 'Archivée',
  draft: 'Brouillon',
};

const formatDate = (value) => {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleDateString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
};

const rowKey = (row) =>
  row.public_id || row.organization_key || `${row.organization_type}:${row.organization_id}`;

const AdminOrganizations = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [items, setItems] = useState([]);
  const [summary, setSummary] = useState({});
  const [pagination, setPagination] = useState({ page: 1, per_page: 50, total: 0, pages: 1 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedOrg, setSelectedOrg] = useState(null);
  const [readMode, setReadMode] = useState('legacy');

  const search = searchParams.get('search') || '';
  const organizationType = searchParams.get('organization_type') || '';
  const lifecycleStatus = searchParams.get('lifecycle_status') || '';
  const includeSynthetic = searchParams.get('include_synthetic') === 'true';
  const page = Math.max(Number(searchParams.get('page') || 1) || 1, 1);

  const updateParams = (patch) => {
    const next = new URLSearchParams(searchParams);
    Object.entries(patch).forEach(([key, value]) => {
      if (value === '' || value === null || value === undefined || value === false) {
        next.delete(key);
      } else {
        next.set(key, String(value));
      }
    });
    setSearchParams(next, { replace: true });
  };

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchPartnerOrganizations({
          page,
          per_page: 50,
          search: search || undefined,
          organization_type: organizationType || undefined,
          lifecycle_status: lifecycleStatus || undefined,
          include_synthetic: includeSynthetic,
        });
        if (cancelled) return;
        setItems(data.items || []);
        setSummary(data.summary || {});
        setPagination(data.pagination || { page: 1, per_page: 50, total: 0, pages: 1 });
        setReadMode(data.read_mode || 'legacy');
      } catch (err) {
        if (cancelled) return;
        setError(err?.response?.data?.message || 'Impossible de charger les organisations.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [page, search, organizationType, lifecycleStatus, includeSynthetic]);

  const metrics = useMemo(() => {
    if (summary.organizations_production != null || summary.active != null) {
      return [
        { label: 'Organisations production', value: summary.organizations_production || 0 },
        { label: 'Actives', value: summary.active || 0 },
        { label: 'En onboarding', value: summary.onboarding || 0 },
        { label: 'Suspendues', value: summary.suspended || 0 },
        { label: 'À vérifier', value: summary.needs_attention || 0 },
      ];
    }
    return [
      { label: 'Organisations configurées', value: summary.configured_organizations || 0 },
      { label: 'Configurations incomplètes', value: summary.incomplete_configurations || 0 },
      { label: 'Accès restreints', value: summary.restricted_companies || 0 },
      { label: 'Démonstrations actives', value: summary.active_demonstrations || 0 },
    ];
  }, [summary]);

  const openRow = (row) => {
    if (row.public_id) {
      navigate(`../organizations/${row.public_id}`);
      return;
    }
    setSelectedOrg(row);
  };

  return (
    <main className={shell.content}>
      <header className={styles.pageHeader}>
        <div>
          <p className={styles.eyebrow}>Partenaires</p>
          <h1 className={styles.title}>Organisations</h1>
          <p className={styles.subtext}>
            Structures LIRIE (entreprises et institutions) — lecture seule. Mode lecture :{' '}
            {readMode}.
          </p>
        </div>
      </header>

      <section className={styles.metricsGrid} aria-label="Indicateurs organisations">
        {metrics.map((m) => (
          <article key={m.label} className={styles.metricCard}>
            <span>{m.label}</span>
            <strong>{m.value}</strong>
          </article>
        ))}
      </section>

      <div className={styles.toolbar}>
        <label className={styles.searchWrap}>
          <FiSearch aria-hidden />
          <input
            type="search"
            value={search}
            placeholder="Rechercher une organisation…"
            onChange={(e) => updateParams({ search: e.target.value, page: 1 })}
          />
        </label>
        <select
          value={organizationType}
          onChange={(e) => updateParams({ organization_type: e.target.value, page: 1 })}
          aria-label="Type d'organisation"
        >
          <option value="">Tous les types</option>
          <option value="company">Entreprises</option>
          <option value="institution">Institutions</option>
        </select>
        <select
          value={lifecycleStatus}
          onChange={(e) => updateParams({ lifecycle_status: e.target.value, page: 1 })}
          aria-label="Statut de cycle de vie"
        >
          <option value="">Tous les statuts</option>
          <option value="active">Active</option>
          <option value="onboarding">Onboarding</option>
          <option value="suspended">Suspendue</option>
        </select>
        <label className={styles.checkLabel}>
          <input
            type="checkbox"
            checked={includeSynthetic}
            onChange={(e) => updateParams({ include_synthetic: e.target.checked, page: 1 })}
          />
          Inclure hors production
        </label>
      </div>

      {loading ? <p className={styles.muted}>Chargement…</p> : null}
      {error ? (
        <p className={styles.error} role="alert">
          {error}
        </p>
      ) : null}

      {!loading && !error ? (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Organisation</th>
                <th>Type</th>
                <th>Statut</th>
                <th>Prestations détectées</th>
                <th>Utilisateurs</th>
                <th>Activité</th>
                <th>Alerte</th>
              </tr>
            </thead>
            <tbody>
              {items.length === 0 ? (
                <tr>
                  <td colSpan={7} className={styles.muted}>
                    Aucune organisation trouvée.
                  </td>
                </tr>
              ) : (
                items.map((row) => (
                  <tr
                    key={rowKey(row)}
                    className={styles.clickableRow}
                    onClick={() => openRow(row)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        openRow(row);
                      }
                    }}
                    tabIndex={0}
                  >
                    <td>{row.name || '—'}</td>
                    <td>{TYPE_LABELS[row.organization_type] || row.organization_type}</td>
                    <td>
                      {LIFECYCLE_LABELS[row.lifecycle_status] ||
                        row.lifecycle_status ||
                        row.configuration_status ||
                        '—'}
                    </td>
                    <td>
                      {row.services_detected_count != null
                        ? `${row.services_detected_count} détectées`
                        : '—'}
                    </td>
                    <td>{row.accounts_count ?? '—'}</td>
                    <td>
                      {row.last_activity_at
                        ? formatDate(row.last_activity_at)
                        : 'Non disponible'}
                    </td>
                    <td>
                      {row.comparison_state === 'missing_in_cp'
                        ? 'Projection manquante'
                        : row.data_origin === 'unknown'
                          ? 'Origine inconnue'
                          : '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className={styles.pagination}>
        <button
          type="button"
          disabled={page <= 1}
          onClick={() => updateParams({ page: page - 1 })}
        >
          Précédent
        </button>
        <span>
          Page {pagination.page} / {pagination.pages} ({pagination.total} au total)
        </span>
        <button
          type="button"
          disabled={page >= (pagination.pages || 1)}
          onClick={() => updateParams({ page: page + 1 })}
        >
          Suivant
        </button>
      </div>

      {selectedOrg ? (
        <AdminOrganizationDetailDrawer
          organization={selectedOrg}
          onClose={() => setSelectedOrg(null)}
        />
      ) : null}
    </main>
  );
};

export default AdminOrganizations;
