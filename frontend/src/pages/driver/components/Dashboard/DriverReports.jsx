// src/components/DriverReports.jsx
import React from "react";
import styles from "./DriverReports.module.css";

const DriverReports = ({ reports }) => {
  return (
    <div className={styles.reports}>
      <h2>Rapports du jour</h2>
      <p>Nombre de courses : {reports.totalCourses}</p>
      <p>Temps total de trajet : {reports.totalTime} minutes</p>
      <p>Revenu généré : {reports.totalRevenue} CHF</p>
      {/* Vous pouvez intégrer un graphique ici */}
    </div>
  );
};

export default DriverReports;
