// src/components/layout/Sidebar/CompanySidebar/CompanySidebar.js
import React, { useEffect, useMemo, useState } from 'react';
import { NavLink, useParams, useLocation } from 'react-router-dom';
import {
  FaHome,
  FaCar,
  FaUser,
  FaUsers,
  FaFileInvoice,
  FaCog,
  FaChevronRight,
  FaChevronLeft,
  FaChartLine,
  FaChartBar,
} from 'react-icons/fa';
import styles from './CompanySidebar.module.css';

const CompanySidebar = ({ isOpen: _isOpen, onToggle: _onToggle }) => {
  const params = useParams();
  const location = useLocation();
  const [open, setOpen] = useState(false);

  // Récupérer public_id depuis useParams() ou extraire de l'URL
  const public_id =
    params.public_id ||
    (() => {
      const match = location.pathname.match(/\/dashboard\/company\/([^/]+)/);
      return match ? match[1] : null;
    })();

  // Ajuste une variable CSS globale pour pousser le contenu
  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty('--sidebar-w', open ? '240px' : '72px');
    return () => root.style.removeProperty('--sidebar-w');
  }, [open]);

  // Ne créer les items que si public_id existe
  const items = useMemo(() => {
    // Si pas de public_id, retourner tableau vide
    if (!public_id) return [];

    return [
      {
        to: `/dashboard/company/${public_id}`,
        label: 'Tableau de bord',
        icon: <FaHome className={styles.icon} />,
        end: true,
      },
      {
        to: `/dashboard/company/${public_id}/reservations`,
        label: 'Réservations',
        icon: <FaCar className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/drivers`,
        label: 'Chauffeurs',
        icon: <FaUser className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/clients`,
        label: 'Gestion Clients',
        icon: <FaUsers className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/invoices/clients`,
        label: 'Facturation par Client',
        icon: <FaFileInvoice className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/dispatch`,
        label: 'Dispatch & Planification',
        icon: <FaChartLine className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/analytics`,
        label: 'Analytics',
        icon: <FaChartBar className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/settings`,
        label: 'Paramètres',
        icon: <FaCog className={styles.icon} />,
      },
    ];
  }, [public_id]);

  return (
    <aside
      className={`${styles.sidebar} ${open ? styles.open : styles.collapsed}`}
      aria-label="Navigation entreprise"
    >
      <button
        type="button"
        className={styles.toggler}
        aria-expanded={open}
        aria-label={open ? 'Réduire le menu' : 'Développer le menu'}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <FaChevronLeft /> : <FaChevronRight />}
      </button>

      <ul className={styles.menu}>
        {items.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              end={!!item.end} // v6
              className={({ isActive }) => (isActive ? styles.active : undefined)} // v6
            >
              {item.icon}
              <span className={styles.text}>{item.label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </aside>
  );
};

export default CompanySidebar;
