import React, { useEffect, useState } from 'react';
import { fetchBookings } from '../../../services/bookingService';
import { fetchClient } from '../../../services/clientService';
import styles from './Reservations.module.css';
import { FaMapMarkerAlt, FaCalendarAlt, FaMoneyBillWave, FaFilePdf } from 'react-icons/fa';

import apiClient from '../../../utils/apiClient';
// ✅ SUPPRIMÉ: mergeInvoiceAndQRBill - Génération PDF déplacée vers backend
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import useCompanyData from '../../../hooks/useCompanyData';

const ReservationsPage = () => {
  const { company: _company } = useCompanyData(); // Récupérer les données de l'entreprise
  const [bookings, setBookings] = useState([]);
  const [_clientData, setClientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState('scheduled_time');
  const [filter, setFilter] = useState('all');
  const [selectedMonth, setSelectedMonth] = useState('');
  const [isExporting, setIsExporting] = useState(false);

  useEffect(() => {
    loadClientData(); // D'abord récupérer le public_id du client
  }, []);

  useEffect(() => {
    const publicId = localStorage.getItem('public_id');
    if (publicId) {
      setLoading(true);
      fetchBookings(publicId)
        .then((data) => {
          setBookings(data);
        })
        .catch((_err) => {
          setError('Erreur lors du chargement des réservations.');
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, []);

  const loadClientData = async () => {
    try {
      const client = await fetchClient(); // Cette fonction doit retourner le profil client du user connecté
      setClientData(client);
    } catch (err) {
      console.error('Erreur lors du chargement du profil client :', err);
    }
  };

  // 🎯 Annuler une réservation
  const handleCancelBooking = async (bookingId) => {
    if (!window.confirm('Voulez-vous vraiment annuler cette réservation ?')) return;

    try {
      setBookings((prevBookings) =>
        prevBookings.map((b) => (b.id === bookingId ? { ...b, isCancelling: true } : b))
      );

      const response = await apiClient.delete(`/bookings/${bookingId}`);

      if (response.status === 200) {
        const updatedBookings = await fetchBookings();
        setBookings(updatedBookings.map((b) => ({ ...b, isCancelling: false })));
        alert('Réservation annulée avec succès !');
      } else {
        throw new Error("L'annulation a échoué.");
      }
    } catch (error) {
      console.error("Erreur lors de l'annulation :", error);
      alert("Une erreur s'est produite lors de l'annulation.");
    }
  };

  const handleExportPDF = async () => {
    if (!selectedMonth) {
      alert("Veuillez sélectionner un mois avant d'exporter.");
      return;
    }

    // Vérifier que les infos du client sont chargées
    const publicId = localStorage.getItem('public_id');
    if (!publicId) {
      alert('Les informations client ne sont pas disponibles.');
      return;
    }

    setIsExporting(true);
    try {
      // Filtrer les réservations pour le mois sélectionné ET appartenant au client connecté
      const monthBookings = bookings.filter(
        (booking) => new Date(booking.scheduled_time).getMonth() + 1 === parseInt(selectedMonth, 10)
      );

      if (monthBookings.length === 0) {
        alert('Aucune réservation trouvée pour ce mois.');
        setIsExporting(false);
        return;
      }

      // ✅ TODO: Remplacer par appel API backend pour génération PDF
      // const response = await apiClient.post('/companies/me/invoices', {
      //   client_id: clientData.id,
      //   period_year: selectedMonth.getFullYear(),
      //   period_month: selectedMonth.getMonth() + 1
      // });
      // window.open(response.data.pdf_url, '_blank');

      alert('Génération PDF déplacée vers backend API - À implémenter');

      alert('Facture générée avec succès !');
    } catch (error) {
      console.error("Erreur lors de l'exportation du PDF :", error);
      alert("Une erreur est survenue lors de l'exportation.");
    }
    setIsExporting(false);
  };

  // 📌 Tri et filtrage des réservations
  const sortedBookings = [...bookings].sort((a, b) => {
    if (!a || !b) return 0;

    if (sortBy === 'scheduled_time') {
      return new Date(a.scheduled_time) - new Date(b.scheduled_time);
    } else if (sortBy === 'amount') {
      return parseFloat(b.amount || 0) - parseFloat(a.amount || 0);
    } else if (sortBy === 'status') {
      return a.status.localeCompare(b.status);
    }
    return 0;
  });

  const filteredBookings = sortedBookings.filter((booking) => {
    if (filter === 'all') return true;
    return booking.status === filter;
  });

  const nowTimestamp = Date.now();
  const upcomingBookings = filteredBookings.filter(
    (booking) => Date.parse(booking.scheduled_time) > nowTimestamp
  );
  const pastBookings = filteredBookings.filter(
    (booking) => Date.parse(booking.scheduled_time) <= nowTimestamp
  );

  return (
    <div className={styles.pageContainer}>
      <HeaderDashboard />

      <div className={styles.reservationsContainer}>
        <div className={styles.header}>
          <h1>📌 Mes Réservations</h1>

          {loading && <p className={styles.loading}>Chargement en cours...</p>}
          {error && <p className={styles.error}>{error}</p>}

          <div className={styles.exportControls}>
            <select
              value={selectedMonth}
              onChange={(e) => setSelectedMonth(e.target.value)}
              className={styles.monthSelect}
            >
              <option value="">📅 Sélectionner un mois</option>
              {[...Array(12)].map((_, i) => (
                <option key={i + 1} value={i + 1}>
                  {new Date(2025, i).toLocaleString('fr-FR', {
                    month: 'long',
                  })}
                </option>
              ))}
            </select>

            <button className={styles.exportBtn} onClick={handleExportPDF} disabled={isExporting}>
              <FaFilePdf /> {isExporting ? 'Exportation...' : 'Exporter en PDF'}
            </button>
          </div>
        </div>

        {/* 🚀 Filtres de tri et de statut */}
        <div className={styles.controls}>
          <select onChange={(e) => setSortBy(e.target.value)}>
            <option value="scheduled_time">📅 Trier par Date</option>
            <option value="amount">💰 Trier par Montant</option>
            <option value="status">🔄 Trier par Statut</option>
          </select>

          <select onChange={(e) => setFilter(e.target.value)}>
            <option value="all">📋 Tous</option>
            <option value="pending">⏳ En attente</option>
            <option value="completed">✅ Terminées</option>
            <option value="canceled">❌ Annulées</option>
          </select>
        </div>

        {/* 🚀 Affichage des courses à venir */}
        <h2 className={styles.sectionTitle}>📅 Courses à venir</h2>
        <div className={styles.reservationList}>
          {upcomingBookings.length > 0 ? (
            upcomingBookings.map((booking) => {
              const status = booking.status.toLowerCase();
              return (
                <div className={styles.reservationCard} key={booking.id}>
                  <h3>
                    <FaCalendarAlt /> {new Date(booking.scheduled_time).toLocaleDateString()}
                  </h3>
                  <p>
                    <FaMapMarkerAlt /> <strong>Départ :</strong>{' '}
                    {booking.pickup_location || 'Inconnu'}
                  </p>
                  <p>
                    <FaMapMarkerAlt /> <strong>Arrivée :</strong>{' '}
                    {booking.dropoff_location || 'Inconnu'}
                  </p>
                  <p>
                    🚖 <strong>Entreprise :</strong> {booking.company_name}
                  </p>
                  <p>
                    👨‍✈️ <strong>Chauffeur :</strong> {booking.driver_name}
                  </p>
                  <p>
                    <FaMoneyBillWave /> <strong>Montant :</strong>{' '}
                    {status === 'canceled'
                      ? '0 CHF'
                      : booking.amount
                      ? `${booking.amount} CHF`
                      : 'N/A'}
                  </p>
                  <p>
                    <strong>Statut :</strong>{' '}
                    <span
                      className={
                        status === 'completed'
                          ? styles.statusCompleted
                          : status === 'in_progress'
                          ? styles.statusInProgress
                          : status === 'canceled'
                          ? styles.statusCanceled
                          : styles.statusDefault
                      }
                    >
                      {status === 'completed'
                        ? '✅ Terminé'
                        : status === 'in_progress'
                        ? '🚖 En cours'
                        : status === 'canceled'
                        ? '❌ Annulé'
                        : '🔄 En attente'}
                    </span>
                  </p>
                  {status !== 'canceled' && (
                    <button
                      className={styles.cancelBtn}
                      onClick={() => handleCancelBooking(booking.id)}
                      disabled={booking.isCancelling}
                    >
                      {booking.isCancelling ? 'Annulation...' : 'Annuler'}
                    </button>
                  )}
                </div>
              );
            })
          ) : (
            <p>Aucune course à venir.</p>
          )}
        </div>

        {/* 🚀 Affichage des courses passées */}
        <h2 className={styles.sectionTitle}>📅 Courses passées</h2>
        <div className={styles.reservationList}>
          {pastBookings.length > 0 ? (
            pastBookings.map((booking) => (
              <div className={styles.reservationCard} key={booking.id}>
                <h3>
                  <FaCalendarAlt /> {new Date(booking.scheduled_time).toLocaleDateString()}
                </h3>
                <p>
                  <FaMapMarkerAlt /> <strong>Départ :</strong>{' '}
                  {booking.pickup_location || 'Inconnu'}
                </p>
                <p>
                  <FaMapMarkerAlt /> <strong>Arrivée :</strong>{' '}
                  {booking.dropoff_location || 'Inconnu'}
                </p>
                <p>
                  <FaMoneyBillWave /> <strong>Montant :</strong>{' '}
                  {booking.amount ? `${booking.amount} CHF` : 'N/A'}
                </p>
                <p>
                  <strong>Statut :</strong>{' '}
                  <span
                    className={
                      booking.status === 'completed'
                        ? styles.statusCompleted
                        : booking.status === 'in_progress'
                        ? styles.statusInProgress
                        : booking.status === 'canceled'
                        ? styles.statusCanceled
                        : styles.statusDefault
                    }
                  >
                    {booking.status === 'completed'
                      ? '✅ Terminé'
                      : booking.status === 'in_progress'
                      ? '🚖 En cours'
                      : booking.status === 'canceled'
                      ? '❌ Annulé'
                      : '🔄 En attente'}
                  </span>
                </p>
              </div>
            ))
          ) : (
            <p>Aucune course passée.</p>
          )}
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default ReservationsPage;
