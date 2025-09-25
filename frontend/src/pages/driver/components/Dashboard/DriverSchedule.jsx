// Fichier: C:\Users\jasiq\atmr\frontend\src\pages\driver\components\Dashboard\DriverSchedule.jsx
import React from "react";
import styles from "./DriverSchedule.module.css";
import { FiClock, FiMapPin, FiFlag } from "react-icons/fi"; // Importation des icônes

const DriverSchedule = ({ assignments }) => {
  return (
    <div className={styles.scheduleContainer}>
      <h2 className={styles.scheduleTitle}>Planning de la journée</h2>
      
      {!assignments || assignments.length === 0 ? (
        <p className={styles.noAssignments}>Aucune course assignée pour aujourd'hui.</p>
      ) : (
        <ul className={styles.scheduleList}>
          {assignments.map((assignment) => (
            <li key={assignment.id} className={styles.scheduleItem}>
              
              <div className={styles.timeColumn}>
                <FiClock style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                {new Date(assignment.scheduled_time).toLocaleTimeString("fr-FR", {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </div>

              <div className={styles.detailsColumn}>
                <div className={styles.location}>
                  <FiMapPin /> <span>{assignment.pickup_location}</span>
                </div>
                <div className={styles.location}>
                  <FiFlag /> <span>{assignment.dropoff_location}</span>
                </div>
              </div>

            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default DriverSchedule;