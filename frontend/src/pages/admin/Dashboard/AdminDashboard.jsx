import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
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
import Sidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';

const AdminDashboard = () => {
  // La route est /dashboard/admin/:public_id → on récupère public_id
  const { public_id: adminId } = useParams();

  const [stats, setStats] = useState({});
  const [recentBookings, setRecentBookings] = useState([]);
  const [, setRecentUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStats();
    loadRecentBookings();
    loadRecentUsers();
  }, [adminId]);

  const loadStats = async () => {
    try {
      const data = await fetchAdminStats();
      setStats(data);
    } catch (error) {
      setError('Erreur lors du chargement des statistiques.');
    }
    setLoading(false);
  };

  const loadRecentBookings = async () => {
    try {
      const data = await fetchRecentBookings();
      setRecentBookings(data);
    } catch (error) {
      console.error(
        '🔴 Erreur lors du chargement des réservations :',
        error.response?.data || error.message
      );
    }
  };

  const loadRecentUsers = async () => {
    try {
      const data = await fetchRecentUsers();
      setRecentUsers(data);
    } catch (error) {
      console.error('Erreur chargement des utilisateurs :', error);
    }
  };

  return (
    <div className={styles.adminContainer}>
      <HeaderDashboard />
      <div className={styles.dashboard}>
        <Sidebar adminId={adminId} /> {/* ✅ Passer public_id (adminId) à la Sidebar */}
        <main className={styles.content}>
          <h1>📊 Tableau de bord administrateur</h1>

          {loading ? <p>Chargement des statistiques...</p> : null}
          {error ? <p className={styles.error}>{error}</p> : null}

          {/* ✅ Statistiques */}
          <div className={styles.stats}>
            <div className={styles.card}>
              <FaCar className={styles.icon} />
              <h3>Courses réalisées</h3>
              <p>{stats.totalBookings || 0}</p>
            </div>

            <div className={styles.card}>
              <FaUser className={styles.icon} />
              <h3>Utilisateurs actifs</h3>
              <p>{stats.totalUsers || 0}</p>
            </div>

            <div className={styles.card}>
              <FaFileInvoice className={styles.icon} />
              <h3>Factures générées</h3>
              <p>{stats.totalInvoices || 0}</p>
            </div>

            <div className={styles.card}>
              <FaChartBar className={styles.icon} />
              <h3>Revenu total (CHF)</h3>
              <p>{stats.totalRevenue || 0} CHF</p>
            </div>
          </div>

          {/* ✅ Graphiques */}
          <div className={styles.chartContainer}>
            <h2>📈 Évolution des réservations</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={stats.bookingTrends || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="bookings" stroke="#8884d8" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* ✅ Tableau des dernières courses */}
          <div className={styles.tableContainer}>
            <h2>🚖 Dernières réservations</h2>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>📅 Date</th>
                  <th>🕒 Heure</th>
                  <th>👤 Client</th>
                  <th>📍 Départ</th>
                  <th>📍 Arrivée</th>
                  <th>💰 Montant</th>
                  <th>⚡ Statut</th>
                </tr>
              </thead>
              <tbody>
                {recentBookings.map((booking) => (
                  <tr key={booking.id}>
                    <td>{booking.date_formatted || 'Non spécifié'}</td>
                    <td>{booking.time_formatted || 'Non spécifié'}</td>
                    <td>{booking.customer_name || 'Non spécifié'}</td>
                    <td>{booking.pickup_location || 'Non spécifié'}</td>
                    <td>{booking.dropoff_location || 'Non spécifié'}</td>
                    <td>{booking.amount} CHF</td>
                    <td>
                      <span
                        className={`${styles.status} ${
                          booking.status === 'PENDING'
                            ? styles.pending
                            : booking.status === 'CANCELED'
                              ? styles.canceled
                              : styles.completed
                        }`}
                      >
                        {booking.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </main>
      </div>
    </div>
  );
};

export default AdminDashboard;
