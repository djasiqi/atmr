// src/components/layout/Sidebar/AdminSidebar/AdminSidebar.js
import React from 'react';
import { NavLink, useParams } from 'react-router-dom';
import { FaHome, FaUser, FaCar, FaFileInvoice, FaCog, FaRobot, FaChartLine } from 'react-icons/fa';
import styles from './AdminSidebar.module.css';

const AdminSidebar = () => {
  const { public_id } = useParams(); // Récupération du public_id depuis l'URL
  const adminId = public_id ?? '';

  return (
    <nav className={styles.sidebar}>
      <ul>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}`}
            end
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaHome /> Tableau de bord
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/reservations`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaCar /> Réservations
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/users`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaUser /> Utilisateurs
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/shadow-mode`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaRobot /> Shadow Mode MDI
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/optuna`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaChartLine /> Optimisation Optuna
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/invoices`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaFileInvoice /> Factures
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/admin/${adminId}/settings`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaCog /> Paramètres
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default AdminSidebar;
