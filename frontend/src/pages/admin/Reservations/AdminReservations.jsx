import React, { useEffect, useState } from "react";
import HeaderDashboard from "../../../components/layout/Header/HeaderDashboard";
import Sidebar from "../../../components/layout/Sidebar/AdminSidebar/AdminSidebar";
import { fetchRecentBookings } from "../../../services/adminService";
import styles from "./AdminReservations.module.css";

const AdminReservations = () => {
  const [bookings, setBookings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadBookings = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchRecentBookings();
        setBookings(Array.isArray(data) ? data : []);
      } catch (err) {
        const message = err?.response?.data?.message || err?.message || "Erreur inconnue";
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
        <Sidebar />
        <main className={styles.content}>
          <header className={styles.header}>
            <h1>Reservations</h1>
            <p>Liste des dernieres reservations effectuees.</p>
          </header>

          {loading && <p>Chargement des reservations...</p>}
          {error && <p className={styles.error}>Erreur: {error}</p>}

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
                      <td colSpan="7" className={styles.empty}>Aucune reservation recente.</td>
                    </tr>
                  ) : (
                    bookings.map((booking) => (
                      <tr key={booking.id ?? booking.booking_id}>
                        <td>{booking.id ?? booking.booking_id ?? "N/A"}</td>
                        <td>{booking.date_formatted ?? "--"} {booking.time_formatted ?? ""}</td>
                        <td>{booking.customer_name ?? "--"}</td>
                        <td>{booking.pickup_location ?? "--"}</td>
                        <td>{booking.dropoff_location ?? "--"}</td>
                        <td>{booking.amount != null ? `${booking.amount} CHF` : "--"}</td>
                        <td>{booking.status ?? "--"}</td>
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

