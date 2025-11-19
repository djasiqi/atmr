import React from 'react';
import { NavLink, useParams } from 'react-router-dom';
import { FaHome, FaUser, FaCar, FaFileInvoice, FaCog, FaRobot } from 'react-icons/fa';
import styles from './AdminSidebar.module.css';

const Sidebar = () => {
  const { public_id } = useParams(); // Récupération du public_id depuis l'URL
  const adminId = public_id ?? '';

  return (
    <nav className={styles.sidebar}>
      <ul>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}`}
            end
            className={({ isActive }) => (isActive ? styles.active : undefined)}
          >
            <FaHome /> Tableau de bord
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/reservations`}
            className={({ isActive }) => (isActive ? styles.active : undefined)}
          >
            <FaCar /> Réservations
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/users`}
            className={({ isActive }) => (isActive ? styles.active : undefined)}
          >
            <FaUser /> Utilisateurs
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/shadow-mode`}
            className={({ isActive }) => (isActive ? styles.active : undefined)}
          >
            <FaRobot /> Shadow Mode MDI
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/invoices`}
            className={({ isActive }) => (isActive ? styles.active : undefined)}
          >
            <FaFileInvoice /> Factures
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/settings`}
            className={({ isActive }) => (isActive ? styles.active : undefined)}
          >
            <FaCog /> Paramètres
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default Sidebar;
