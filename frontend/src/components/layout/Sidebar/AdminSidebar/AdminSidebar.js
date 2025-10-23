import React from 'react';
import { NavLink, useParams } from 'react-router-dom';
import { FaHome, FaUser, FaCar, FaFileInvoice, FaCog, FaRobot } from 'react-icons/fa';
import styles from './AdminSidebar.module.css';

const Sidebar = () => {
  const { adminId } = useParams(); // Récupération de l'adminId depuis l'URL

  return (
    <nav className={styles.sidebar}>
      <ul>
        <li>
          <NavLink to={`/dashboard/admin/${adminId}`} exact activeClassName={styles.active}>
            <FaHome /> Tableau de bord
          </NavLink>
        </li>
        <li>
          <NavLink to={`/dashboard/admin/${adminId}/reservations`} activeClassName={styles.active}>
            <FaCar /> Réservations
          </NavLink>
        </li>
        <li>
          <NavLink to={`/dashboard/admin/${adminId}/users`} activeClassName={styles.active}>
            <FaUser /> Utilisateurs
          </NavLink>
        </li>
        <li>
          <NavLink to={`/dashboard/admin/${adminId}/shadow-mode`} activeClassName={styles.active}>
            <FaRobot /> Shadow Mode MDI
          </NavLink>
        </li>
        <li>
          <NavLink to={`/dashboard/admin/${adminId}/invoices`} activeClassName={styles.active}>
            <FaFileInvoice /> Factures
          </NavLink>
        </li>
        <li>
          <NavLink to={`/dashboard/admin/${adminId}/settings`} activeClassName={styles.active}>
            <FaCog /> Paramètres
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default Sidebar;
