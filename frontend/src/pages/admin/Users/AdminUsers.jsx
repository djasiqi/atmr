import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { FiSearch } from 'react-icons/fi';
import {
  fetchUsers,
  fetchControlPlaneAnomalies,
} from '../../../services/adminService';
import AdminAccountManageDrawer from '../Organizations/AdminAccountManageDrawer';
import styles from './AdminUsers.module.css';
import adminShell from '../adminShell.module.css';

const ROLE_LABELS = {
  admin: 'Admin',
  ADMIN: 'Admin',
  client: 'Client',
  CLIENT: 'Client',
  driver: 'Chauffeur',
  DRIVER: 'Chauffeur',
  company: 'Entreprise',
  COMPANY: 'Entreprise',
  institution: 'Institution',
  INSTITUTION: 'Institution',
};

const formatRole = (role) => ROLE_LABELS[role] || role || '—';

const formatDate = (value) => {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleDateString('fr-CH');
};

const organizationLabel = (user) => {
  if (user.company_name) return user.company_name;
  if (user.institution_id) return `Institution #${user.institution_id}`;
  if (String(user.role || '').toUpperCase() === 'COMPANY' && !user.company_id) {
    return 'Aucune (orphelin)';
  }
  if (String(user.role || '').toUpperCase() === 'INSTITUTION' && !user.institution_id) {
    return 'Aucune (orphelin)';
  }
  return '—';
};

/**
 * Comptes et accès — gestion sécurisée (MDP, rôle) + diagnostic.
 */
const AdminUsers = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [tab, setTab] = useState(searchParams.get('tab') || 'all');
  const [users, setUsers] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [anomalyTotal, setAnomalyTotal] = useState(0);
  const [search, setSearch] = useState((searchParams.get('search') || '').trim());
  const [debouncedSearch, setDebouncedSearch] = useState(search);
  const [roleFilter, setRoleFilter] = useState(searchParams.get('role') || '');
  const [companyFilter, setCompanyFilter] = useState(
    searchParams.get('company_id') || ''
  );
  const [includeSynthetic, setIncludeSynthetic] = useState(
    searchParams.get('include_synthetic') === 'true'
  );
  const [page, setPage] = useState(Number(searchParams.get('page') || 1) || 1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalUsers, setTotalUsers] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [integrityAccountId, setIntegrityAccountId] = useState(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search.trim()), 300);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    const next = new URLSearchParams();
    if (tab && tab !== 'all') next.set('tab', tab);
    if (debouncedSearch) next.set('search', debouncedSearch);
    if (roleFilter) next.set('role', roleFilter);
    if (companyFilter) next.set('company_id', companyFilter);
    if (includeSynthetic) next.set('include_synthetic', 'true');
    if (page > 1) next.set('page', String(page));
    setSearchParams(next, { replace: true });
  }, [
    tab,
    debouncedSearch,
    roleFilter,
    companyFilter,
    includeSynthetic,
    page,
    setSearchParams,
  ]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setLoadError(null);
      try {
        if (tab === 'anomalies') {
          const data = await fetchControlPlaneAnomalies({
            page,
            per_page: 50,
            unresolved_only: true,
          });
          if (cancelled) return;
          setAnomalies(data.items || []);
          setAnomalyTotal(data.pagination?.total || 0);
          setTotalPages(data.pagination?.pages || 1);
        } else {
          const data = await fetchUsers({
            page,
            per_page: 50,
            search: debouncedSearch,
            role: roleFilter,
            company_id: companyFilter || undefined,
            include_synthetic: includeSynthetic,
            paginate: true,
          });
          if (cancelled) return;
          setUsers(data.users || []);
          setTotalPages(data.total_pages || 1);
          setTotalUsers(data.total || 0);
          const attention = await fetchControlPlaneAnomalies({
            page: 1,
            per_page: 5,
            unresolved_only: true,
          });
          if (!cancelled) {
            setAnomalies(attention.items || []);
            setAnomalyTotal(attention.pagination?.total || 0);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setLoadError(err?.response?.data?.message || 'Impossible de charger les comptes.');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [
    page,
    debouncedSearch,
    roleFilter,
    companyFilter,
    includeSynthetic,
    tab,
    reloadToken,
  ]);

  const openAccount = (user) => {
    setIntegrityAccountId(user.id);
  };

  return (
    <main className={adminShell.content}>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderLeft}>
          <p className={styles.pageEyebrow}>Partenaires</p>
          <h1 className={styles.pageTitle}>Comptes et accès</h1>
          <p className={styles.pageSubtitle}>
            {totalUsers} compte{totalUsers === 1 ? '' : 's'}
            {includeSynthetic ? '' : ' (hors techniques/démo)'}. {anomalyTotal} anomalie
            {anomalyTotal === 1 ? '' : 's'} à traiter.
          </p>
        </div>
      </header>

      <div className={styles.filters}>
        <button
          type="button"
          className={tab === 'all' ? styles.actionBtn : undefined}
          onClick={() => {
            setTab('all');
            setPage(1);
          }}
        >
          Tous
        </button>
        <button
          type="button"
          className={tab === 'anomalies' ? styles.actionBtn : undefined}
          onClick={() => {
            setTab('anomalies');
            setPage(1);
          }}
        >
          Anomalies ({anomalyTotal})
        </button>
        <label className={styles.searchBox}>
          <FiSearch aria-hidden />
          <input
            type="search"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            placeholder="Rechercher un compte…"
            disabled={tab === 'anomalies'}
          />
        </label>
        <select
          value={roleFilter}
          onChange={(e) => {
            setRoleFilter(e.target.value);
            setPage(1);
          }}
          aria-label="Filtrer par rôle"
          disabled={tab === 'anomalies'}
        >
          <option value="">Tous les rôles</option>
          <option value="admin">Admin</option>
          <option value="company">Entreprise</option>
          <option value="institution">Institution</option>
          <option value="driver">Chauffeur</option>
          <option value="client">Client</option>
        </select>
        <label className={styles.checkboxLabel}>
          <input
            type="checkbox"
            checked={includeSynthetic}
            onChange={(e) => {
              setIncludeSynthetic(e.target.checked);
              setPage(1);
            }}
            disabled={tab === 'anomalies'}
          />
          Inclure comptes techniques et démo
        </label>
      </div>

      {tab === 'all' && anomalies.length > 0 ? (
        <section className={styles.tableContainer} aria-label="À traiter">
          <h2>À traiter</h2>
          <ul>
            {anomalies.slice(0, 5).map((a) => (
              <li key={a.id}>
                {a.user_id ? (
                  <button
                    type="button"
                    className={styles.inlineLink}
                    onClick={() => setIntegrityAccountId(a.user_id)}
                  >
                    [{a.severity}] {a.code} — {a.entity_key}
                  </button>
                ) : (
                  <>
                    [{a.severity}] {a.code} — {a.entity_key}
                  </>
                )}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {loading ? <p>Chargement…</p> : null}
      {loadError ? (
        <p className={styles.errorText} role="alert">
          {loadError}
        </p>
      ) : null}

      {!loading && !loadError && tab === 'anomalies' ? (
        <div className={styles.tableContainer}>
          <table className={styles.userTable}>
            <thead>
              <tr>
                <th>Code</th>
                <th>Sévérité</th>
                <th>Entité</th>
                <th>Dernière détection</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.length === 0 ? (
                <tr>
                  <td colSpan={4}>Aucune anomalie ouverte.</td>
                </tr>
              ) : (
                anomalies.map((a) => (
                  <tr
                    key={a.id}
                    className={a.user_id ? styles.clickableRow : undefined}
                    tabIndex={a.user_id ? 0 : undefined}
                    onClick={() => {
                      if (a.user_id) setIntegrityAccountId(a.user_id);
                    }}
                    onKeyDown={(e) => {
                      if (!a.user_id) return;
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setIntegrityAccountId(a.user_id);
                      }
                    }}
                  >
                    <td>{a.code}</td>
                    <td>{a.severity}</td>
                    <td>
                      {a.entity_type} / {a.entity_key}
                    </td>
                    <td>{formatDate(a.last_seen_at)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      ) : null}

      {!loading && !loadError && tab === 'all' ? (
        <div className={styles.tableContainer}>
          <table className={styles.userTable}>
            <thead>
              <tr>
                <th>Nom</th>
                <th>E-mail</th>
                <th>Rôle actuel</th>
                <th>Organisation liée</th>
                <th>Inscription</th>
              </tr>
            </thead>
            <tbody>
              {users.length === 0 ? (
                <tr>
                  <td colSpan={5}>Aucun compte trouvé.</td>
                </tr>
              ) : (
                users.map((user) => (
                  <tr
                    key={user.id}
                    className={styles.clickableRow}
                    tabIndex={0}
                    onClick={() => openAccount(user)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        openAccount(user);
                      }
                    }}
                  >
                    <td>{user.username || '—'}</td>
                    <td>{user.email || '—'}</td>
                    <td>
                      <span className={styles.roleBadge}>{formatRole(user.role)}</span>
                    </td>
                    <td>{organizationLabel(user)}</td>
                    <td>{formatDate(user.created_at)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className={styles.pagination}>
        <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          Précédent
        </button>
        <span>
          Page {page} / {totalPages}
        </span>
        <button
          type="button"
          disabled={page >= totalPages}
          onClick={() => setPage((p) => p + 1)}
        >
          Suivant
        </button>
      </div>

      <AdminAccountManageDrawer
        isOpen={Boolean(integrityAccountId)}
        accountId={integrityAccountId}
        onClose={() => setIntegrityAccountId(null)}
        onChanged={() => setReloadToken((t) => t + 1)}
      />
    </main>
  );
};

export default AdminUsers;
