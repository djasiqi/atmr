// src/pages/DriverMapPage.jsx
import React from 'react';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import DriverSidebar from '../../../components/layout/Sidebar/DriverSidebar/DriverSidebar';
import DriverMap from '../components/Dashboard/DriverMap';
import styles from '../Dashboard/DriverDashboard.module.css';

const DriverMapPage = () => {
  // Remplacez par vos données réelles
  const dummyAssignments = [];

  return (
    <div className={styles.driverDashboard}>
      <HeaderDashboard />
      <DriverSidebar />
      <main className={styles.mainContent}>
        <h1>Carte</h1>
        <DriverMap assignments={dummyAssignments} />
      </main>
    </div>
  );
};

export default DriverMapPage;
