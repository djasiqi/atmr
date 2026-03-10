import React, { useEffect, useMemo, useState } from 'react';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import { fetchRecentBookings } from '../../../services/adminService';
import styles from './AdminReservations.module.css';

const AdminReservations = () => {
  const [bookings, setBookings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const statusConfig = useMemo(
    () => ({
      accepted: { label: 'Acceptee', className: styles.statusAccepted },
      assigned: { label: 'Assignee', className: styles.statusAssigned },
      completed: { label: 'Terminee', className: styles.statusCompleted },
      pending: { label: 'En attente', className: styles.statusPending },
      cancelled: { label: 'Annulee', className: styles.statusCancelled },
      rejected: { label: 'Refusee', className: styles.statusRejected },
    }),
    []
  );

  const completedCount = useMemo(
    () =>
      bookings.filter((booking) => String(booking.status || '').toLowerCase() === 'completed').length,
    [bookings]
  );

  const pendingCount = useMemo(
    () => bookings.length - completedCount,
    [bookings.length, completedCount]
  );

  const getStatusMeta = (status) => {
    const key = String(status || '').toLowerCase();
    return statusConfig[key] || { label: status || '--', className: styles.statusDefault };
  };

  useEffect(() => {
    const loadBookings = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchRecentBookings();
        setBookings(Array.isArray(data) ? data : []);
      } catch (err) {
        const message = err?.response?.data?.message || err?.message || 'Erreur inconnue';
        setError(message);
      } finally {
        setLoading(false);
      }
    };

    loadBookings();
  }, []);

  return (
    <div className={styles.container}>
      <HeaderDashboard />
      <div className={styles.body}>
        <AdminSidebar />
        <main className={styles.content}>
          <header className={styles.header}>
            <h1>Reservations</h1>
            <p>Liste des dernieres reservations effectuees.</p>
          </header>

          <section className={styles.metricsRow} aria-label="Synthese reservations">
            <article className={styles.metricCard}>
              <span>Total affiche</span>
              <strong>{bookings.length}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Terminees</span>
              <strong>{completedCount}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>En cours</span>
              <strong>{pendingCount}</strong>
            </article>
          </section>

          {loading && (
            <div className={styles.feedbackCard}>
              <p>Chargement des reservations...</p>
            </div>
          )}
          {error && (
            <div className={styles.feedbackCard}>
              <p className={styles.error}>Erreur: {error}</p>
            </div>
          )}

          {!loading && !error && (
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Date</th>
                    <th>Client</th>
                    <th>Depart</th>
                    <th>Arrivee</th>
                    <th>Montant</th>
                    <th>Statut</th>
                  </tr>
                </thead>
                <tbody>
                  {bookings.length === 0 ? (
                    <tr>
                      <td colSpan="7" className={styles.empty}>
                        Aucune reservation recente.
                      </td>
                    </tr>
                  ) : (
                    bookings.map((booking) => (
                      <tr key={booking.id ?? booking.booking_id}>
                        <td>
                          <span className={styles.idBadge}>{booking.id ?? booking.booking_id ?? 'N/A'}</span>
                        </td>
                        <td>
                          {booking.date_formatted ?? '--'} {booking.time_formatted ?? ''}
                        </td>
                        <td>{booking.client_name ?? '--'}</td>
                        <td className={styles.locationCell} title={booking.pickup_location ?? '--'}>
                          {booking.pickup_location ?? '--'}
                        </td>
                        <td className={styles.locationCell} title={booking.dropoff_location ?? '--'}>
                          {booking.dropoff_location ?? '--'}
                        </td>
                        <td className={styles.amountCell}>
                          {booking.amount != null ? `${booking.amount} CHF` : '--'}
                        </td>
                        <td>
                          {(() => {
                            const statusMeta = getStatusMeta(booking.status);
                            return (
                              <span className={`${styles.statusBadge} ${statusMeta.className}`}>
                                {statusMeta.label}
                              </span>
                            );
                          })()}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default AdminReservations;
