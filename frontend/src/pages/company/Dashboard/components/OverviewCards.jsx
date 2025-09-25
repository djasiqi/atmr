// src/pages/Dashboard/OverviewCards.jsx
import React from "react";
import styles from "../CompanyDashboard.module.css";

const OverviewCards = ({ reservations, driver }) => {
  // Calculer quelques indicateurs
  const pendingCount = (reservations || []).filter(
    (r) => r.status.toLowerCase() === "pending"
  ).length;
  const completedCount = (reservations || []).filter(
    (r) =>
      r.status.toLowerCase() === "completed" ||
      r.status.toLowerCase() === "assigned"
  ).length;
  const revenue = (reservations || []).reduce((acc, r) => {
    if (r.status.toLowerCase() === "completed") {
      return acc + (Number(r.amount) || 0);
    }
    return acc;
  }, 0);
  const availableDriver = (driver || []).filter((d) => d.is_active).length;

  return (
    <div className={styles.overviewCards}>
      <div className={styles.card}>
        <h3>Réservations en attente</h3>
        <p>{pendingCount}</p>
      </div>
      <div className={styles.card}>
        <h3>Courses réalisées</h3>
        <p>{completedCount}</p>
      </div>
      <div className={styles.card}>
        <h3>Revenu généré</h3>
        <p>{revenue} CHF</p>
      </div>
      <div className={styles.card}>
        <h3>Chauffeurs disponibles</h3>
        <p>{availableDriver}</p>
      </div>
    </div>
  );
};

export default OverviewCards;
