// Fichier: C:\Users\jasiq\atmr\frontend\src\pages\driver\components\Dashboard\DriverOverview.jsx
import React from "react";
import styles from "./DriverOverview.module.css";
import { FiList, FiTruck, FiCheckSquare } from "react-icons/fi"; // Importation des icônes

const DriverOverview = ({ assignments }) => {
  const total = assignments.length;
  // Correction: inclure 'in_progress' pour une meilleure correspondance avec les statuts de la DB
  const inProgress = assignments.filter(
    (a) => a.status === "assigned" || a.status === "in_progress"
  ).length;
  // Correction: inclure tous les statuts terminés
  const completed = assignments.filter(
    (a) => a.status === "completed" || a.status === "return_completed"
  ).length;

  return (
    <div className={styles.overview}>
      {/* Carte Total */}
      <div className={`${styles.card} ${styles.totalCard}`}>
        <div className={styles.iconWrapper}>
          <FiList />
        </div>
        <div className={styles.cardContent}>
          <h4 className={styles.cardTitle}>Total Courses</h4>
          <p className={styles.cardNumber}>{total}</p>
        </div>
      </div>

      {/* Carte En cours */}
      <div className={`${styles.card} ${styles.inProgressCard}`}>
        <div className={styles.iconWrapper}>
          <FiTruck />
        </div>
        <div className={styles.cardContent}>
          <h4 className={styles.cardTitle}>En cours</h4>
          <p className={styles.cardNumber}>{inProgress}</p>
        </div>
      </div>

      {/* Carte Terminées */}
      <div className={`${styles.card} ${styles.completedCard}`}>
        <div className={styles.iconWrapper}>
          <FiCheckSquare />
        </div>
        <div className={styles.cardContent}>
          <h4 className={styles.cardTitle}>Terminées</h4>
          <p className={styles.cardNumber}>{completed}</p>
        </div>
      </div>
    </div>
  );
};

export default DriverOverview;