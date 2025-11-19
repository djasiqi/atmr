// src/components/layout/Sidebar/CompanySidebar/CompanySidebar.js
import React, { useMemo } from 'react';
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

  // Ne créer les items que si public_id existe
  const items = useMemo(() => {
    // Si pas de public_id, retourner tableau vide
    if (!public_id) return [];

    return [
      {
        to: `/dashboard/company/${public_id}`,
        label: 'Tableau de bord',
        icon: <FaHome />,
        end: true,
      },
      {
        to: `/dashboard/company/${public_id}/reservations`,
        label: 'Réservations',
        icon: <FaCar />,
      },
      {
        to: `/dashboard/company/${public_id}/drivers`,
        label: 'Chauffeurs',
        icon: <FaUser />,
      },
      {
        to: `/dashboard/company/${public_id}/clients`,
        label: 'Gestion Clients',
        icon: <FaUsers />,
      },
      {
        to: `/dashboard/company/${public_id}/invoices/clients`,
        label: 'Facturation par Client',
        icon: <FaFileInvoice />,
      },
      {
        to: `/dashboard/company/${public_id}/dispatch`,
        label: 'Dispatch & Planification',
        icon: <FaChartLine />,
      },
      {
        to: `/dashboard/company/${public_id}/analytics`,
        label: 'Analytics',
        icon: <FaChartBar />,
      },
      {
        to: `/dashboard/company/${public_id}/settings`,
        label: 'Paramètres',
        icon: <FaCog />,
      },
    ];
  }, [public_id]);

  return (
    <nav className={styles.sidebar}>
      <ul>
        {items.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              end={!!item.end}
              className={({ isActive }) => (isActive ? styles.active : '')}
            >
              {item.icon} {item.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default CompanySidebar;
