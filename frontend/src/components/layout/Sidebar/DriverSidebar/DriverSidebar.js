// src/components/layout/DriverSidebar/DriverSidebar.js
import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import {
  FaHome,
  FaCalendarAlt,
  FaMapMarkerAlt,
  FaHistory,
  FaCog,
  FaBars,
  FaTimes,
} from 'react-icons/fa';
import styles from './DriverSidebar.module.css';

const DriverSidebar = () => {
  // État du toggle (sauvegardé dans localStorage)
  const [isExpanded, setIsExpanded] = useState(() => {
    const saved = localStorage.getItem('driverSidebarExpanded');
    return saved !== null ? saved === 'true' : true; // Par défaut ouvert
  });

  // Sauvegarder l'état dans localStorage
  useEffect(() => {
    localStorage.setItem('driverSidebarExpanded', String(isExpanded));
  }, [isExpanded]);

  const toggleSidebar = () => {
    setIsExpanded(!isExpanded);
  };

  const menuItems = [
    {
      icon: FaHome,
      label: 'Tableau de bord',
      to: '/driver/dashboard',
    },
    {
      icon: FaCalendarAlt,
      label: 'Planning',
      to: '/driver/schedule',
    },
    {
      icon: FaMapMarkerAlt,
      label: 'Carte',
      to: '/driver/map',
    },
    {
      icon: FaHistory,
      label: 'Historique',
      to: '/driver/history',
    },
    {
      icon: FaCog,
      label: 'Paramètres',
      to: '/driver/settings',
    },
  ];

  return (
    <nav className={`${styles.sidebar} ${isExpanded ? styles.expanded : styles.collapsed}`}>
      {/* Bouton toggle */}
      <button
        type="button"
        className={styles.toggleButton}
        onClick={toggleSidebar}
        aria-label={isExpanded ? 'Réduire la barre latérale' : 'Agrandir la barre latérale'}
        aria-expanded={isExpanded}
      >
        {isExpanded ? <FaTimes /> : <FaBars />}
      </button>

      {/* Liste des liens */}
      <ul className={styles.menuList}>
        {menuItems.map((item) => {
          const Icon = item.icon;
          return (
            <li key={item.to} className={styles.menuItem}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  `${styles.menuLink} ${isActive ? styles.active : ''}`
                }
                title={!isExpanded ? item.label : undefined}
              >
                <Icon className={styles.icon} />
                {isExpanded && <span className={styles.label}>{item.label}</span>}
              </NavLink>
            </li>
          );
        })}
      </ul>
    </nav>
  );
};

export default DriverSidebar;
