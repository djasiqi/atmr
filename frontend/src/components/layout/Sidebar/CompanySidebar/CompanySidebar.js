// src/components/layout/Sidebar/CompanySidebar/CompanySidebar.js
import React from 'react';
import { NavLink, useParams, useLocation } from 'react-router-dom';
import {
  FaHome,
  FaCar,
  FaUser,
  FaUsers,
  FaFileInvoice,
  FaCog,
  FaChartLine,
  FaChartBar,
} from 'react-icons/fa';
import styles from './CompanySidebar.module.css';

const CompanySidebar = () => {
  const params = useParams();
  const location = useLocation();

  // Récupérer public_id depuis useParams() ou extraire de l'URL
  const public_id =
    params.public_id ||
    (() => {
      const match = location.pathname.match(/\/dashboard\/company\/([^/]+)/);
      return match ? match[1] : null;
    })();

  // Si pas de public_id, ne rien afficher
  if (!public_id) {
    return null;
  }

  return (
    <nav className={styles.sidebar}>
      <ul>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}`}
            end
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaHome /> Tableau de bord
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}/reservations`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaCar /> Réservations
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}/drivers`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaUser /> Chauffeurs
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}/clients`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaUsers /> Gestion Clients
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}/invoices/clients`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaFileInvoice /> Facturation par Client
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}/dispatch`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaChartLine /> Dispatch & Planification
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}/analytics`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaChartBar /> Analytics
          </NavLink>
        </li>
        <li>
          <NavLink
            to={`/dashboard/company/${public_id}/settings`}
            className={({ isActive }) => (isActive ? styles.active : '')}
          >
            <FaCog /> Paramètres
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default CompanySidebar;
