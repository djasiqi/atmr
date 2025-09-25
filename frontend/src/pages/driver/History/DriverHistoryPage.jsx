// src/pages/DriverHistoryPage.jsx
import React from "react";
import HeaderDashboard from "../../../components/layout/Header/HeaderDashboard";
import DriverSidebar from "../../../components/layout/Sidebar/DriverSidebar/DriverSidebar";
import DriverHistory from "../components/Dashboard/DriverHistory";
import styles from '../Dashboard/DriverDashboard.module.css';

const DriverHistoryPage = () => {
  // Remplacez par vos données réelles d'historique
  const dummyHistory = [];

  return (
    <div className={styles.driverDashboard}>
      <HeaderDashboard />
      <DriverSidebar />
      <main className={styles.mainContent}>
        <h1>Historique</h1>
        <DriverHistory history={dummyHistory} />
      </main>
    </div>
  );
};

export default DriverHistoryPage;
