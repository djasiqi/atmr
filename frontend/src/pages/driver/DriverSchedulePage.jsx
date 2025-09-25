// src/pages/DriverSchedulePage.jsx
import React from "react";
import HeaderDashboard from "../../components/layout/Header/HeaderDashboard";
import DriverSidebar from "../../components/layout/Sidebar/DriverSidebar/DriverSidebar";
import DriverSchedule from "./components/Dashboard/DriverSchedule";
import styles from './Dashboard/DriverDashboard.module.css';

const DriverSchedulePage = () => {
  // Vous pouvez récupérer des données spécifiques au planning ici.
  const dummyAssignments = []; // Remplacez par vos données réelles

  return (
    <div className={styles.driverDashboard}>
      <HeaderDashboard />
      <DriverSidebar />
      <main className={styles.mainContent}>
        <h1>Planning</h1>
        <DriverSchedule assignments={dummyAssignments} />
      </main>
    </div>
  );
};

export default DriverSchedulePage;
