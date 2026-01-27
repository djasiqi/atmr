// src/components/layout/Sidebar/CompanySidebar/CompanySidebar.js
import React, { useState, useEffect } from 'react';
import { NavLink, useParams, useLocation } from 'react-router-dom';
import {
  FaHome,
  FaCar,
  FaUser,
  FaUsers,
  FaFileInvoice,
  FaCheckCircle,
  FaCog,
  FaChartLine,
  FaChartBar,
  FaBars,
  FaTimes,
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

  // État du toggle (sauvegardé dans localStorage)
  const [isExpanded, setIsExpanded] = useState(() => {
    const saved = localStorage.getItem('companySidebarExpanded');
    return saved !== null ? saved === 'true' : true; // Par défaut ouvert
  });

  // Sauvegarder l'état dans localStorage
  useEffect(() => {
    localStorage.setItem('companySidebarExpanded', String(isExpanded));
  }, [isExpanded]);

  // Mettre à jour la variable CSS pour la responsivité du contenu principal
  // S'exécute au montage et à chaque changement d'état
  useEffect(() => {
    const sidebarWidth = isExpanded ? 240 : 72; // Largeur en pixels
    document.documentElement.style.setProperty('--sidebar-w', `${sidebarWidth}px`);
    
    // Cleanup : restaurer la valeur par défaut si nécessaire
    return () => {
      document.documentElement.style.setProperty('--sidebar-w', '72px');
    };
  }, [isExpanded]);

  const toggleSidebar = () => {
    setIsExpanded(!isExpanded);
  };

  // Si pas de public_id, ne rien afficher
  if (!public_id) {
    return null;
  }

  const menuItems = [
    {
      icon: FaHome,
      label: 'Tableau de bord',
      to: `/dashboard/company/${public_id}`,
      end: true,
    },
    {
      icon: FaCar,
      label: 'Réservations',
      to: `/dashboard/company/${public_id}/reservations`,
    },
    {
      icon: FaUser,
      label: 'Chauffeurs',
      to: `/dashboard/company/${public_id}/drivers`,
    },
    {
      icon: FaUsers,
      label: 'Gestion Clients',
      to: `/dashboard/company/${public_id}/clients`,
    },
    {
      icon: FaFileInvoice,
      label: 'Facturation par Client',
      to: `/dashboard/company/${public_id}/invoices/clients`,
    },
    {
      icon: FaCheckCircle,
      label: 'Contrôle Facturation',
      to: `/dashboard/company/${public_id}/billing-review`,
    },
    {
      icon: FaChartLine,
      label: 'Dispatch & Planification',
      to: `/dashboard/company/${public_id}/dispatch`,
    },
    {
      icon: FaChartBar,
      label: 'Analytics',
      to: `/dashboard/company/${public_id}/analytics`,
    },
    {
      icon: FaCog,
      label: 'Paramètres',
      to: `/dashboard/company/${public_id}/settings`,
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
                end={item.end}
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

export default CompanySidebar;
