// src/pages/Dashboard/AssignmentModal.jsx
import React from "react";
import styles from "./CompanyDashboard.module.css";

const AssignmentModal = ({ reservation, driver, onAssign, onClose }) => {
  return (
    <div className={styles.modal}>
      <div className={styles.modalContent}>
        <h3>Assigner un chauffeur à la réservation {reservation.id}</h3>
        <p>Sélectionnez un chauffeur disponible :</p>
        <div className={styles.driverList}>
          {driver.map((driver) => (
            <div key={driver.id} className={styles.driverItem}>
              <span className={styles.driverName}>{driver.username}</span>
              <button
              className={styles.assignButton}
              onClick={() => {
                console.log("👤 Assign clicked", driver.id, reservation.id);
                onAssign(reservation.id, driver.id);
              }}
              title="Assigner ce chauffeur"
            >
              Assigner
            </button>
            </div>
          ))}
        </div>
        <button className={styles.cancelButton} onClick={onClose}>
          Annuler
        </button>
      </div>
    </div>
  );
};

export default AssignmentModal;
