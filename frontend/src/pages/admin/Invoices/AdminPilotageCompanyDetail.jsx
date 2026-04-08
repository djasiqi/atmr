import React, { useCallback, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { fetchBillingPilotageCompanyDetail } from '../../../services/adminService';
import styles from './AdminInvoices.module.css';

const fmtMoney = (n) => {
  if (n == null || Number.isNaN(Number(n))) return '—';
  return `${Number(n).toLocaleString('fr-CH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} CHF`;
};

const reliabilityFr = (bucket) => {
  const b = String(bucket || '').toLowerCase();
  if (b === 'good') return 'Bon';
  if (b === 'medium') return 'Moyen';
  if (b === 'low') return 'Faible';
  return '—';
};

const AdminPilotageCompanyDetail = () => {
  const { public_id: adminId, companyId } = useParams();
  const [createdFrom, setCreatedFrom] = useState('');
  const [createdTo, setCreatedTo] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const base = { page, per_page: 40 };
      if (createdFrom) base.created_from = createdFrom;
      if (createdTo) base.created_to = createdTo;
      const res = await fetchBillingPilotageCompanyDetail(companyId, base);
      setData(res);
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [companyId, createdFrom, createdTo, page]);

  useEffect(() => {
    load();
  }, [load]);

  const summary = data?.summary;
  const co = data?.company;

  return (
    <div>
      <nav className={styles.breadcrumb}>
        <Link to={`/dashboard/admin/${adminId}/billing/pilotage`}>← Pilotage facturable</Link>
      </nav>

      <section className={styles.hero}>
        <h1>{co?.name || 'Entreprise'}</h1>
        <p className={styles.subtitle}>
          Compte d’exploitation plateforme — lecture des activités et qualification pour un futur billing.
        </p>
      </section>

      <section className={styles.filters}>
        <div className={styles.filterRow}>
          <label>
            Créées du
            <input
              type="date"
              value={createdFrom}
              onChange={(e) => {
                setPage(1);
                setCreatedFrom(e.target.value);
              }}
            />
          </label>
          <label>
            au
            <input
              type="date"
              value={createdTo}
              onChange={(e) => {
                setPage(1);
                setCreatedTo(e.target.value);
              }}
            />
          </label>
          <button type="button" className={styles.btnSecondary} onClick={load} disabled={loading}>
            Actualiser
          </button>
        </div>
      </section>

      {error && <div className={styles.errorBanner}>{error}</div>}
      {loading && !data && <p className={styles.muted}>Chargement…</p>}

      {summary && (
        <section className={styles.detailGrid}>
          <div className={styles.detailCard}>
            <h3>Résumé</h3>
            <ul className={styles.detailList}>
              <li>Réservations : {summary.total_bookings}</li>
              <li>Montant observable : {fmtMoney(summary.total_observed_amount)}</li>
              <li>Fiabilité : {reliabilityFr(summary.reliability?.bucket)}
                {summary.reliability?.percent != null && ` (${summary.reliability.percent}%)`}
              </li>
            </ul>
          </div>
          <div className={styles.detailCard}>
            <h3>Qualification</h3>
            <ul className={styles.detailList}>
              <li>Éligibles : {summary.eligible}</li>
              <li>Ambiguës : {summary.ambiguous}</li>
              <li>À revoir : {summary.needs_review}</li>
              <li>Exclues : {summary.excluded}</li>
            </ul>
          </div>
          <div className={styles.detailCard}>
            <h3>Origine (agrégé)</h3>
            <ul className={styles.detailList}>
              <li>Institution : {summary.institution}</li>
              <li>Manuel / direct : {summary.manual_direct}</li>
              <li>Inconnu : {summary.unknown_source}</li>
            </ul>
          </div>
        </section>
      )}

      {data?.source_breakdown && (
        <section className={styles.tableSection}>
          <h2 className={styles.sectionTitle}>Décomposition par source (code)</h2>
          <pre className={styles.codeBlock}>{JSON.stringify(data.source_breakdown, null, 2)}</pre>
        </section>
      )}

      {data?.bookings && (
        <section className={styles.tableSection}>
          <h2 className={styles.sectionTitle}>Réservations contributrices</h2>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Créée</th>
                  <th>Statut</th>
                  <th>Source</th>
                  <th>Qualification</th>
                  <th>Montant obs.</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.bookings.map((row) => (
                  <tr key={row.booking_id}>
                    <td>{row.booking_id}</td>
                    <td>{row.created_at || '—'}</td>
                    <td>{row.status}</td>
                    <td>{row.pilotage?.source_code}</td>
                    <td>{row.pilotage?.qualification?.state}</td>
                    <td>{fmtMoney(row.pilotage?.observed_transport_amount)}</td>
                    <td>
                      <Link
                        className={styles.linkDetail}
                        to={`/dashboard/admin/${adminId}/reservations/${row.booking_id}`}
                      >
                        Fiche
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.pagination && data.pagination.total_pages > 1 && (
            <div className={styles.pagination}>
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Précédent
              </button>
              <span>
                Page {data.pagination.page} / {data.pagination.total_pages || 1}
              </span>
              <button
                type="button"
                disabled={page >= (data.pagination.total_pages || 1)}
                onClick={() => setPage((p) => p + 1)}
              >
                Suivant
              </button>
            </div>
          )}
        </section>
      )}
    </div>
  );
};

export default AdminPilotageCompanyDetail;
