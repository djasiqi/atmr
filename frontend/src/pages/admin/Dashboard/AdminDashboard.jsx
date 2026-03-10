import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { FaUser, FaCar, FaChartBar, FaFileInvoice } from 'react-icons/fa';
import {
  fetchAdminStats,
  fetchRecentBookings,
  fetchRecentUsers,
} from '../../../services/adminService';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import styles from './AdminDashboard.module.css';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';

const AdminDashboard = () => {
  // La route est /dashboard/admin/:public_id → on récupère public_id
  const { public_id: adminId } = useParams();

  const [stats, setStats] = useState({});
  const [recentBookings, setRecentBookings] = useState([]);
  const [recentUsers, setRecentUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [statsResult, bookingsResult, usersResult] = await Promise.allSettled([
          fetchAdminStats(),
          fetchRecentBookings(),
          fetchRecentUsers(),
        ]);

        if (statsResult.status === 'fulfilled') {
          setStats(statsResult.value || {});
        } else {
          setError('Erreur lors du chargement des statistiques.');
        }

        if (bookingsResult.status === 'fulfilled') {
          setRecentBookings(Array.isArray(bookingsResult.value) ? bookingsResult.value : []);
        } else {
          setRecentBookings([]);
        }

        if (usersResult.status === 'fulfilled') {
          setRecentUsers(Array.isArray(usersResult.value) ? usersResult.value : []);
        } else {
          setRecentUsers([]);
        }
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, [adminId]);

  const statusMeta = (status) => {
    const key = String(status || '').toLowerCase();
    if (['completed', 'done', 'terminee', 'terminée'].includes(key)) {
      return { label: 'Terminee', className: styles.completed };
    }
    if (['pending', 'en_attente'].includes(key)) {
      return { label: 'En attente', className: styles.pending };
    }
    if (['assigned', 'accepted', 'in_progress', 'en_cours'].includes(key)) {
      return { label: 'En cours', className: styles.assigned };
    }
    if (['canceled', 'cancelled', 'annulee', 'annulée', 'rejected'].includes(key)) {
      return { label: 'Annulee', className: styles.canceled };
    }
    return { label: status || '--', className: styles.defaultStatus };
  };

  const inProgressCount = recentBookings.filter((booking) =>
    ['pending', 'assigned', 'accepted', 'in_progress'].includes(
      String(booking.status || '').toLowerCase()
    )
  ).length;

  return (
    <div className={styles.adminContainer}>
      <HeaderDashboard />
      <div className={styles.dashboard}>
        <AdminSidebar />
        <main className={styles.content}>
          <header className={styles.pageHeader}>
            <div>
              <h1>Tableau de bord administrateur</h1>
              <p>Vue d ensemble de l activite, des utilisateurs et des reservations.</p>
            </div>
            <div className={styles.quickActions}>
              <Link to={`/dashboard/admin/${adminId}/reservations`} className={styles.actionButton}>
                Voir reservations
              </Link>
              <Link to={`/dashboard/admin/${adminId}/users`} className={styles.actionButtonGhost}>
                Gérer utilisateurs
              </Link>
            </div>
          </header>

          <section className={styles.pilotageRow}>
            <article className={styles.pilotCard}>
              <span>Reservations recentes</span>
              <strong>{recentBookings.length}</strong>
            </article>
            <article className={styles.pilotCard}>
              <span>Nouveaux utilisateurs</span>
              <strong>{recentUsers.length}</strong>
            </article>
            <article className={styles.pilotCard}>
              <span>A surveiller</span>
              <strong>{inProgressCount}</strong>
            </article>
          </section>

          {loading ? <p className={styles.infoText}>Chargement des statistiques...</p> : null}
          {error ? <p className={styles.error}>{error}</p> : null}

          {/* ✅ Statistiques */}
          <div className={styles.stats}>
            <div className={styles.card}>
              <FaCar className={styles.icon} />
              <div className={styles.cardContent}>
                <h3>Courses realisees</h3>
                <p>{stats.totalBookings || 0}</p>
              </div>
            </div>

            <div className={styles.card}>
              <FaUser className={styles.icon} />
              <div className={styles.cardContent}>
                <h3>Utilisateurs actifs</h3>
                <p>{stats.totalUsers || 0}</p>
              </div>
            </div>

            <div className={styles.card}>
              <FaFileInvoice className={styles.icon} />
              <div className={styles.cardContent}>
                <h3>Factures generees</h3>
                <p>{stats.totalInvoices || 0}</p>
              </div>
            </div>

            <div className={styles.card}>
              <FaChartBar className={styles.icon} />
              <div className={styles.cardContent}>
                <h3>Revenu total (CHF)</h3>
                <p>{stats.totalRevenue || 0} CHF</p>
              </div>
            </div>
          </div>

          {/* ✅ Graphiques */}
          <div className={styles.chartContainer}>
            <h2>Evolution des reservations</h2>
            {stats.bookingTrends && stats.bookingTrends.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={stats.bookingTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="bookings" stroke="#8884d8" name="Réservations" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className={styles.emptyChart}>
                <p>Aucune donnee disponible pour l instant.</p>
                <p className={styles.emptySubText}>
                  Les statistiques d'évolution apparaîtront une fois que des réservations auront été créées.
                </p>
              </div>
            )}
          </div>

          {/* ✅ Tableau des dernières courses */}
          <div className={styles.tableContainer}>
            <h2>Dernieres reservations</h2>
            <div className={styles.tableScroller}>
              <table className={styles.table}>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Heure</th>
                  <th>Client</th>
                  <th>Depart</th>
                  <th>Arrivee</th>
                  <th>Montant</th>
                  <th>Statut</th>
                </tr>
              </thead>
              <tbody>
                {recentBookings.length === 0 ? (
                  <tr>
                    <td colSpan="7" className={styles.emptyLine}>
                      Aucune reservation recente.
                    </td>
                  </tr>
                ) : (
                  recentBookings.map((booking) => {
                    const status = statusMeta(booking.status);
                    return (
                      <tr key={booking.id}>
                        <td>{booking.date_formatted || 'Non specifie'}</td>
                        <td>{booking.time_formatted || '--:--'}</td>
                        <td>{booking.client_name || 'Non specifie'}</td>
                        <td className={styles.locationCell} title={booking.pickup_location || ''}>
                          {booking.pickup_location || 'Non specifie'}
                        </td>
                        <td className={styles.locationCell} title={booking.dropoff_location || ''}>
                          {booking.dropoff_location || 'Non specifie'}
                        </td>
                        <td className={styles.amountCell}>
                          {booking.amount != null ? `${booking.amount} CHF` : '--'}
                        </td>
                        <td>
                          <span className={`${styles.status} ${status.className}`}>{status.label}</span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default AdminDashboard;
