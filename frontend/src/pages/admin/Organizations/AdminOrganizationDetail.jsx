import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { fetchOrganizationDetail } from '../../../services/adminService';
import styles from './AdminOrganizations.module.css';
import shell from '../adminShell.module.css';

const AdminOrganizationDetail = () => {
  const { publicId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const payload = await fetchOrganizationDetail(publicId);
        if (!cancelled) setData(payload);
      } catch (err) {
        if (!cancelled) {
          setError(err?.response?.data?.message || 'Organisation introuvable.');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [publicId]);

  return (
    <main className={shell.content}>
      <p className={styles.eyebrow}>
        <Link to="../organizations">← Organisations</Link>
      </p>
      {loading ? <p className={styles.muted}>Chargement…</p> : null}
      {error ? (
        <p className={styles.error} role="alert">
          {error}
        </p>
      ) : null}
      {data ? (
        <>
          <header className={styles.pageHeader}>
            <div>
              <h1 className={styles.title}>{data.name || 'Organisation'}</h1>
              <p className={styles.subtext}>
                {data.organization_type === 'company'
                  ? 'Entreprise de transport'
                  : 'Institution'}{' '}
                · {data.lifecycle_status} · Configuration à confirmer
              </p>
            </div>
            <button type="button" onClick={() => navigate(-1)}>
              Fermer
            </button>
          </header>

          <section className={styles.metricsGrid}>
            <article className={styles.metricCard}>
              <span>Utilisateurs détectés</span>
              <strong>{data.accounts_count ?? 0}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Prestations détectées</span>
              <strong>{data.services_detected_count ?? 0}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Origine</span>
              <strong>{data.data_origin || 'unknown'}</strong>
            </article>
          </section>

          <section className={styles.tableWrap}>
            <h2>Vue d&apos;ensemble</h2>
            <p>Contact : {data.contact_email || '—'}</p>
            <p>
              Readiness : identité{' '}
              {data.readiness?.identity_ready ? 'ok' : 'à compléter'} · accès{' '}
              {data.readiness?.access_ready ? 'ok' : 'à compléter'} · services non
              confirmés
            </p>
          </section>

          <section className={styles.tableWrap}>
            <h2>Prestations détectées</h2>
            <ul>
              {(data.services_detected || []).map((s) => (
                <li key={s.service_key}>
                  {s.label || s.service_key} ({s.enforcement_mode})
                </li>
              ))}
            </ul>
            {(data.services_detected || []).length === 0 ? (
              <p className={styles.muted}>Aucune prestation détectée.</p>
            ) : null}
          </section>

          <section className={styles.tableWrap}>
            <h2>Utilisateurs détectés</h2>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Nom</th>
                  <th>Email</th>
                  <th>Statut appartenance</th>
                </tr>
              </thead>
              <tbody>
                {(data.users_detected || []).map((u) => (
                  <tr key={u.user_id}>
                    <td>{u.name}</td>
                    <td>{u.email || '—'}</td>
                    <td>{u.membership_status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className={styles.tableWrap}>
            <h2>Anomalies</h2>
            {(data.anomalies || []).length === 0 ? (
              <p className={styles.muted}>Aucune anomalie ouverte.</p>
            ) : (
              <ul>
                {data.anomalies.map((a) => (
                  <li key={`${a.code}-${a.entity_key}`}>
                    [{a.severity}] {a.code}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      ) : null}
    </main>
  );
};

export default AdminOrganizationDetail;
