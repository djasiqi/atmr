import React, { useCallback, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  fetchBillingPilotageCompanies,
  fetchBillingPilotageSummary,
  downloadBillingPilotageExport,
} from '../../../services/adminService';
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

const AdminInvoices = () => {
  const { public_id: adminId } = useParams();
  const [createdFrom, setCreatedFrom] = useState('');
  const [createdTo, setCreatedTo] = useState('');
  const [summary, setSummary] = useState(null);
  const [companies, setCompanies] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const base = {};
      if (createdFrom) base.created_from = createdFrom;
      if (createdTo) base.created_to = createdTo;
      const [sum, co] = await Promise.all([
        fetchBillingPilotageSummary(base),
        fetchBillingPilotageCompanies({ ...base, page, per_page: 25, sort: 'total_bookings', order: 'desc' }),
      ]);
      setSummary(sum);
      setCompanies(co);
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement');
    } finally {
      setLoading(false);
    }
  }, [createdFrom, createdTo, page]);

  useEffect(() => {
    load();
  }, [load]);

  const kpis = summary?.kpis;
  const anomalyFamilies = summary?.anomaly_families || {};

  const onExport = async () => {
    const base = {};
    if (createdFrom) base.created_from = createdFrom;
    if (createdTo) base.created_to = createdTo;
    await downloadBillingPilotageExport(base);
  };

  return (
    <div>
      <section className={styles.hero}>
        <h2 className={styles.pageHeading}>Pilotage facturable</h2>
        <p className={styles.subtitle}>
          Contrôle qualité du périmètre facturable : activité observée, anomalies, qualification
          (éligible / ambigu / à revoir). Ce n’est pas un relevé LIRIE ni une facture légale.
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
          <button type="button" className={styles.btnSecondary} onClick={onExport} disabled={loading}>
            Export CSV pilotage
          </button>
        </div>
        {summary?.period?.field === 'created_at_default_30d' && (
          <p className={styles.hint}>Période par défaut : 30 derniers jours (création).</p>
        )}
      </section>

      {error && <div className={styles.errorBanner}>{error}</div>}

      {loading && !summary && <p className={styles.muted}>Chargement…</p>}

      {kpis && (
        <>
          <section className={styles.kpiGrid} aria-label="Synthèse activité">
            <div className={styles.kpiCard}>
              <span className={styles.kpiLabel}>Entreprises actives</span>
              <span className={styles.kpiValue}>{kpis.active_companies}</span>
            </div>
            <div className={styles.kpiCard}>
              <span className={styles.kpiLabel}>Réservations totales</span>
              <span className={styles.kpiValue}>{kpis.total_bookings}</span>
            </div>
            <div className={styles.kpiCard}>
              <span className={styles.kpiLabel}>Activité institution</span>
              <span className={styles.kpiValue}>{kpis.activity_institution}</span>
            </div>
            <div className={styles.kpiCard}>
              <span className={styles.kpiLabel}>Activité manuelle / directe</span>
              <span className={styles.kpiValue}>{kpis.activity_manual_direct}</span>
            </div>
            <div className={styles.kpiCard}>
              <span className={styles.kpiLabel}>Montant observable</span>
              <span className={styles.kpiValue}>{fmtMoney(kpis.total_observed_amount)}</span>
            </div>
            <div className={styles.kpiCard}>
              <span className={styles.kpiLabel}>Réservations éligibles</span>
              <span className={styles.kpiValue}>{kpis.reservations_eligible}</span>
            </div>
            <div className={styles.kpiCard}>
              <span className={styles.kpiLabel}>Réservations à revoir</span>
              <span className={styles.kpiValue}>{kpis.reservations_needs_review}</span>
              <span className={styles.kpiSecondary}>
                Ambiguës (secondaire) : {kpis.reservations_ambiguous_secondary ?? 0}
              </span>
            </div>
          </section>

          <section className={styles.anomalySection} aria-label="Anomalies par familles">
            <h2 className={styles.sectionTitle}>Anomalies par familles</h2>
            <div className={styles.anomalyGrid}>
              {['source', 'montant', 'transfert', 'investigation', 'perimeter'].map((key) => (
                <div key={key} className={styles.anomalyChip}>
                  <span className={styles.anomalyLabel}>{key}</span>
                  <span className={styles.anomalyCount}>{anomalyFamilies[key] ?? 0}</span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {companies?.items && (
        <section className={styles.tableSection}>
          <h2 className={styles.sectionTitle}>Entreprises (porteuse)</h2>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Entreprise</th>
                  <th>Total</th>
                  <th>Institution</th>
                  <th>Manuel / direct</th>
                  <th>Inconnu</th>
                  <th>Exécutées</th>
                  <th>Transférées</th>
                  <th>Montant obs.</th>
                  <th>Éligibles</th>
                  <th>Ambiguës</th>
                  <th>À revoir</th>
                  <th>Fiabilité</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {companies.items.map((row) => (
                  <tr key={row.company_id}>
                    <td>
                      <strong>{row.company_name || '—'}</strong>
                      {row.account_suspended && <span className={styles.badgeWarn}>Suspendu</span>}
                    </td>
                    <td>{row.total_bookings}</td>
                    <td>{row.institution}</td>
                    <td>{row.manual_direct}</td>
                    <td>{row.unknown_source}</td>
                    <td>{row.executed}</td>
                    <td>{row.transferred}</td>
                    <td>{fmtMoney(row.total_observed_amount)}</td>
                    <td>{row.eligible}</td>
                    <td>{row.ambiguous}</td>
                    <td>{row.needs_review}</td>
                    <td>
                      {reliabilityFr(row.reliability?.bucket)}
                      {row.reliability?.percent != null && (
                        <span className={styles.muted}> ({row.reliability.percent}%)</span>
                      )}
                    </td>
                    <td>
                      <Link
                        className={styles.linkDetail}
                        to={`/dashboard/admin/${adminId}/invoices/pilotage/companies/${row.company_id}`}
                      >
                        Détail
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {companies.pagination && companies.pagination.total_pages > 1 && (
            <div className={styles.pagination}>
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Précédent
              </button>
              <span>
                Page {companies.pagination.page} / {companies.pagination.total_pages || 1}
              </span>
              <button
                type="button"
                disabled={page >= (companies.pagination.total_pages || 1)}
                onClick={() => setPage((p) => p + 1)}
              >
                Suivant
              </button>
            </div>
          )}
        </section>
      )}

      {summary && (
        <footer className={styles.metaFooter}>
          <span>
            Classification v{summary.classification_version} · Qualification v{summary.qualification_version}
          </span>
        </footer>
      )}
    </div>
  );
};

export default AdminInvoices;
