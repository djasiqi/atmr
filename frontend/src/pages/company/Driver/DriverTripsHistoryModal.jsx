import React, { useEffect, useState, useCallback } from 'react';
import { FiX, FiChevronLeft, FiChevronRight, FiClock, FiMapPin } from 'react-icons/fi';
import { fetchDriverCompletedTrips } from '../../../services/companyService';
import styles from './DriverTripsHistoryModal.module.css';

const PER_PAGE = 25;

function formatDuration(minutes) {
  const total = Number(minutes) || 0;
  const h = Math.floor(total / 60);
  const m = total % 60;
  return h > 0 ? `${h}h${String(m).padStart(2, '0')}` : `${m} min`;
}

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' });
}

/**
 * Historique détaillé des trajets d'un chauffeur — pagination SQL côté serveur,
 * chargée uniquement à l'ouverture de la modale (Lot 5 perf, pas de dump global).
 */
export default function DriverTripsHistoryModal({ driverId, driverName, onClose }) {
  const [page, setPage] = useState(1);
  const [trips, setTrips] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadPage = useCallback(async (targetPage) => {
    if (!driverId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDriverCompletedTrips(driverId, { page: targetPage, perPage: PER_PAGE });
      const list = Array.isArray(data) ? data : Array.isArray(data?.trips) ? data.trips : [];
      setTrips(list);
      setTotal(Array.isArray(data) ? list.length : Number(data?.total) || list.length);
    } catch (e) {
      setError("Impossible de charger l'historique des trajets");
      setTrips([]);
    } finally {
      setLoading(false);
    }
  }, [driverId]);

  useEffect(() => {
    loadPage(page);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [driverId, page]);

  const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className={styles.header}>
          <h3 className={styles.title}>Historique des trajets — {driverName || 'Chauffeur'}</h3>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>

        <div className={styles.body}>
          {loading && <div className={styles.loading}>Chargement…</div>}
          {!loading && error && <div className={styles.error}>{error}</div>}
          {!loading && !error && trips.length === 0 && (
            <div className={styles.empty}>Aucun trajet terminé.</div>
          )}
          {!loading && !error && trips.length > 0 && (
            <ul className={styles.tripList}>
              {trips.map((trip) => (
                <li key={trip.id} className={styles.tripItem}>
                  <div className={styles.tripRoute}>
                    <FiMapPin size={13} className={styles.tripIcon} />
                    <span className={styles.tripFrom}>{trip.pickup_location || '—'}</span>
                    <span className={styles.tripArrow}>→</span>
                    <span className={styles.tripTo}>{trip.dropoff_location || '—'}</span>
                  </div>
                  <div className={styles.tripMeta}>
                    <span>{formatDate(trip.completed_at)}</span>
                    <span className={styles.tripDuration}>
                      <FiClock size={12} />
                      {formatDuration(trip.duration_in_minutes)}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {totalPages > 1 && (
          <div className={styles.footer}>
            <button
              type="button"
              className={styles.pageBtn}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1 || loading}
            >
              <FiChevronLeft size={14} />
              Précédent
            </button>
            <span className={styles.pageInfo}>Page {page} sur {totalPages}</span>
            <button
              type="button"
              className={styles.pageBtn}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages || loading}
            >
              Suivant
              <FiChevronRight size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
