// src/components/common/ResponsiveSidebar.jsx
import React, { useState } from 'react';
import styles from './ResponsiveSidebar.module.css'; // Chemin relatif dans le même dossier

const ResponsiveSidebar = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  return (
    <>
      {/* Bouton hamburger visible sur mobile */}
      <button className={styles.hamburger} onClick={toggleSidebar}>
        ☰
      </button>

      {/* Sidebar : la classe "open" est ajoutée si sidebarOpen est true */}
      <nav className={`${styles.sidebar} ${sidebarOpen ? styles.open : ''}`}>{children}</nav>
    </>
  );
};

export default ResponsiveSidebar;
