// src/components/DriverHistory.jsx
import React from "react";
import styles from "./DriverHistory.module.css";

const DriverHistory = ({ history }) => {
  return (
    <div className={styles.history}>
      <h2>Historique des courses</h2>
      {history.length === 0 ? (
        <p>Aucune course terminée.</p>
      ) : (
        <ul>
          {history.map((course) => (
            <li key={course.id}>
              <span>
                {new Date(course.scheduled_time).toLocaleString("fr-FR")}
              </span>
              <span>
                {course.pickup} → {course.dropoff}
              </span>
              <span>{course.revenue} CHF</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default DriverHistory;
