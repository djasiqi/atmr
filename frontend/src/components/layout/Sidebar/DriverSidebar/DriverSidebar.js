// src/components/layout/DriverSidebar.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import { FaHome, FaCalendarAlt, FaMapMarkerAlt, FaHistory, FaCog } from 'react-icons/fa';
import styles from './DriverSidebar.module.css';

const DriverSidebar = () => {
  return (
    <nav className={styles.sidebar}>
      <ul>
        <li>
          <NavLink
            to="/driver/dashboard"
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaHome /> Tableau de bord
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/driver/schedule"
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaCalendarAlt /> Planning
          </NavLink>
        </li>
        <li>
          <NavLink to="/driver/map" className={({ isActive }) => (isActive ? styles.active : '')}>
            <FaMapMarkerAlt /> Carte
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/driver/history"
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaHistory /> Historique
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/driver/settings"
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaCog /> Paramètres
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default DriverSidebar;
