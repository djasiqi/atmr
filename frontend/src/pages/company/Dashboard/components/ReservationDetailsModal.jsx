// src/pages/Dashboard/ReservationDetailsModal.jsx
import React from "react";
import styles from '../CompanyDashboard.module.css';

const ReservationDetailsModal = ({ reservation, onClose }) => {
  return (
    <div className={styles.modal}>
      <div className={styles.modalContent}>
        <h3>Détails de la réservation #{reservation.id}</h3>
        <p>
          <strong>Client :</strong> {reservation.customer_name}
        </p>
        <p>
          <strong>Date / Heure :</strong>{" "}
          {new Date(reservation.scheduled_time).toLocaleString("fr-FR")}
        </p>
        <p>
          <strong>Montant :</strong> {reservation.amount} CHF
        </p>
        <p>
          <strong>Statut :</strong> {reservation.status}
        </p>
        {reservation.phone && (
          <p>
            <strong>Téléphone :</strong> {reservation.phone}
          </p>
        )}
        {reservation.pickup_location && (
          <p>
            <strong>Départ :</strong> {reservation.pickup_location}
          </p>
        )}
        {reservation.dropoff_location && (
          <p>
            <strong>Arrivée :</strong> {reservation.dropoff_location}
          </p>
        )}
        {reservation.instructions && (
          <p>
            <strong>Instructions :</strong> {reservation.instructions}
          </p>
        )}
        <button className={styles.cancelButton} onClick={onClose}>
          Fermer
        </button>
      </div>
    </div>
  );
};

export default ReservationDetailsModal;
