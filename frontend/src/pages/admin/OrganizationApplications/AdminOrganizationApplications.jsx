import React, { useCallback, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { fetchOrganizationApplications } from '../../../services/adminOrganizationApplicationService';
import { adminPaths } from '../routing/adminRoutePaths';
import shell from '../adminShell.module.css';
import styles from './AdminOrganizationApplications.module.css';

const KIND_OPTIONS = [
  ['', 'Tous les types'],
  ['company', 'Entreprise'],
  ['institution', 'Institution'],
];

const STATUS_OPTIONS = [
  ['', 'Tous les états'],
  ['email_unverified', 'Adresse à confirmer'],
  ['received', 'Dossier reçu'],
  ['in_review', 'En examen'],
  ['needs_info', 'Complément demandé'],
  ['refused', 'Refusée'],
  ['abandoned', 'Abandonnée'],
];

function formatWhen(value) {
  if (!value) return 'Pas encore reçue';
  return new Date(value).toLocaleString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function badgeClass(item) {
  if (item.incomplete) return `${styles.badge} ${styles.badgeIncomplete}`;
  if (!item.actionable) return `${styles.badge} ${styles.badgeClosed}`;
  return styles.badge;
}

export default function AdminOrganizationApplications() {
  const { public_id: adminId } = useParams();
  const [filters, setFilters] = useState({ q: '', kind: '', status: '' });
  const [applied, setApplied] = useState(filters);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = useCallback(async (nextFilters) => {
    setLoading(true);
    setError('');
    try {
      const data = await fetchOrganizationApplications(nextFilters);
      setItems(data.items || []);
    } catch {
      setError('Impossible de charger les demandes.');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(applied);
  }, [applied, load]);

  const onSubmit = (event) => {
    event.preventDefault();
    setApplied(filters);
  };

  return (
    <main className={shell.content}>
      <header className={styles.pageHeader}>
        <div>
          <h1>Inscriptions professionnelles</h1>
          <p className={styles.lead}>
            Les dossiers sont triés par date de réception. Une adresse non confirmée
            reste un dossier incomplet. Aucune approbation n’ouvre de portail.
          </p>
        </div>
        <button type="button" className={styles.ghost} onClick={() => load(applied)} disabled={loading}>
          Actualiser
        </button>
      </header>

      <form className={styles.toolbar} onSubmit={onSubmit}>
        <label className={styles.field}>
          <span>Recherche</span>
          <input
            value={filters.q}
            onChange={(event) => setFilters((current) => ({ ...current, q: event.target.value }))}
            placeholder="Nom, e-mail ou localité"
          />
        </label>
        <label className={styles.field}>
          <span>Type</span>
          <select
            value={filters.kind}
            onChange={(event) => setFilters((current) => ({ ...current, kind: event.target.value }))}
          >
            {KIND_OPTIONS.map(([value, label]) => (
              <option key={value || 'all-kinds'} value={value}>{label}</option>
            ))}
          </select>
        </label>
        <label className={styles.field}>
          <span>État</span>
          <select
            value={filters.status}
            onChange={(event) => setFilters((current) => ({ ...current, status: event.target.value }))}
          >
            {STATUS_OPTIONS.map(([value, label]) => (
              <option key={value || 'all-statuses'} value={value}>{label}</option>
            ))}
          </select>
        </label>
        <button type="submit" className={styles.button}>Filtrer</button>
      </form>

      {error ? <p className={styles.alert} role="alert">{error}</p> : null}
      {loading ? <p className={styles.loading}>Chargement des demandes…</p> : null}

      {!loading && !error && items.length === 0 ? (
        <div className={styles.empty}>
          <h2>Aucune demande à afficher</h2>
          <p>Les dépôts Entreprise et Institution apparaîtront ici.</p>
        </div>
      ) : null}

      {items.length > 0 ? (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Organisation</th>
                <th>Type</th>
                <th>Représentant</th>
                <th>Réception</th>
                <th>File</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.public_id}>
                  <td data-label="Organisation">
                    <Link to={adminPaths.partnersOrganizationApplication(adminId, item.public_id)}>
                      {item.organization_name}
                    </Link>
                  </td>
                  <td data-label="Type">{item.kind_label}</td>
                  <td data-label="Représentant">
                    {item.applicant_name}
                    <div className={styles.email}>{item.applicant_email}</div>
                  </td>
                  <td data-label="Réception">{formatWhen(item.submitted_at)}</td>
                  <td data-label="File">
                    <span className={badgeClass(item)}>{item.queue_label}</span>
                    <div>{item.incomplete ? 'Dossier incomplet' : item.status_label}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </main>
  );
}
