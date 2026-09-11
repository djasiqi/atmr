import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  fetchAdminContactRequests,
  retryAdminContactNotification,
} from '../../../services/adminContactService';
import styles from '../DemoRequests/AdminDemoRequests.module.css';
import shell from '../adminShell.module.css';

const CATEGORY_LABELS = {
  support: 'Support technique',
  institution: 'Institution / Intégration',
  transport: 'Entreprise de transport',
  demo: 'Démonstration',
  billing: 'Facturation',
  family: 'Famille / Proche aidant',
};

const STATUS_META = {
  sent: { label: 'Envoyée', tone: 'success' },
  failed: { label: 'Échec', tone: 'danger' },
  pending: { label: 'En attente', tone: 'neutral' },
  sending: { label: 'Envoi…', tone: 'info' },
  skipped: { label: 'Ignorée', tone: 'neutral' },
  suppressed_duplicate: { label: 'Doublon', tone: 'warning' },
  suppressed_spam: { label: 'Spam', tone: 'warning' },
};

const formatDateTime = (value) => {
  if (!value) return '—';
  return new Date(value).toLocaleString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const getStatusMeta = (status) =>
  STATUS_META[status] || { label: status || '—', tone: 'neutral' };

const AdminContactRequests = () => {
  const [searchParams] = useSearchParams();
  const statusFilter = searchParams.get('status') || 'all';

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [items, setItems] = useState([]);
  const [failedCount, setFailedCount] = useState(0);
  const [busyId, setBusyId] = useState(null);

  const displayItems = useMemo(() => {
    if (statusFilter === 'failed') {
      return items.filter((item) => item.email_delivery_status === 'failed');
    }
    if (statusFilter === 'new') {
      return items.filter((item) => item.status === 'new');
    }
    return items;
  }, [items, statusFilter]);

  const load = useCallback(
    async (showLoading = true) => {
      if (showLoading) {
        setLoading(true);
        setError('');
      }
      try {
        const data = await fetchAdminContactRequests(statusFilter);
        setItems(data.items || []);
        setFailedCount(data.failedCount || 0);
      } catch {
        if (showLoading) {
          setError('Impossible de charger les demandes de contact.');
        }
      } finally {
        if (showLoading) {
          setLoading(false);
        }
      }
    },
    [statusFilter]
  );

  useEffect(() => {
    load();
  }, [load]);

  const handleRetry = async (item) => {
    setBusyId(item.id);
    try {
      await retryAdminContactNotification(item.id);
      await load(false);
    } catch {
      setError("Le nouvel envoi de la notification interne a échoué.");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <main className={shell.content} data-testid="admin-contact-requests">
      <header className={styles.pageHeader}>
        <div>
          <h1>Demandes de contact</h1>
          <p className={styles.subtext}>
            Toutes les demandes sont enregistrées. La confirmation au demandeur
            ne prouve pas que la notification interne a été livrée à info@lirie.ch.
          </p>
        </div>
        <button
          type="button"
          className={styles.refreshButton}
          onClick={() => load()}
          disabled={loading}
        >
          Actualiser
        </button>
      </header>

      <section className={styles.metricsGrid} aria-label="Synthèse notifications">
        <article className={styles.metricCard}>
          <span>Demandes affichées</span>
          <strong>{displayItems.length}</strong>
        </article>
        <article className={styles.metricCard}>
          <span>Notifications internes en échec</span>
          <strong>{failedCount}</strong>
        </article>
      </section>

      {error ? <p className={styles.error}>{error}</p> : null}
      {loading ? <p className={styles.info}>Chargement des demandes…</p> : null}

      {!loading && displayItems.length === 0 ? (
        <div className={styles.emptyState}>
          <h2>Aucune demande à afficher</h2>
          <p>Les nouvelles demandes du site LIRIE apparaîtront ici.</p>
        </div>
      ) : null}

      {displayItems.length > 0 ? (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Référence</th>
                <th>Type</th>
                <th>Demandeur</th>
                <th>Notification interne</th>
                <th>Confirmation</th>
                <th>Tentatives</th>
                <th>Date</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {displayItems.map((item) => {
                const internal = getStatusMeta(item.email_delivery_status);
                const autoreply = getStatusMeta(item.autoreply_delivery_status);
                return (
                  <tr key={item.id}>
                    <td className={styles.idCell}>{item.trace_id}</td>
                    <td>{CATEGORY_LABELS[item.category] || item.category}</td>
                    <td>
                      <div className={styles.requesterName}>{item.name}</div>
                      <div className={styles.requesterEmail}>{item.email}</div>
                      {item.organization ? <div>{item.organization}</div> : null}
                    </td>
                    <td>
                      <span className={`${styles.badge} ${styles[`tone${internal.tone}`]}`}>
                        {internal.label}
                      </span>
                      {item.notification_last_error ? (
                        <div className={styles.requesterEmail}>{item.notification_last_error}</div>
                      ) : null}
                    </td>
                    <td>
                      <span className={`${styles.badge} ${styles[`tone${autoreply.tone}`]}`}>
                        {autoreply.label}
                      </span>
                    </td>
                    <td>{item.notification_retry_count || 0}</td>
                    <td>{formatDateTime(item.created_at)}</td>
                    <td>
                      {item.email_delivery_status === 'failed' ? (
                        <button
                          type="button"
                          className={styles.refreshButton}
                          onClick={() => handleRetry(item)}
                          disabled={busyId === item.id}
                        >
                          {busyId === item.id ? 'Envoi…' : 'Relancer'}
                        </button>
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </main>
  );
};

export default AdminContactRequests;
