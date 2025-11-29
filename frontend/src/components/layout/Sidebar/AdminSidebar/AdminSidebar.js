// src/components/layout/Sidebar/AdminSidebar/AdminSidebar.js
import React, { useState, useEffect } from 'react';
import { NavLink, useParams } from 'react-router-dom';
import {
  FaHome,
  FaUser,
  FaCar,
  FaFileInvoice,
  FaCog,
  FaRobot,
  FaChartLine,
  FaBars,
  FaTimes,
} from 'react-icons/fa';
import styles from './AdminSidebar.module.css';

const AdminSidebar = () => {
  const { public_id } = useParams();
  const adminId = public_id ?? '';

  // État du toggle (sauvegardé dans localStorage)
  const [isExpanded, setIsExpanded] = useState(() => {
    const saved = localStorage.getItem('adminSidebarExpanded');
    return saved !== null ? saved === 'true' : true; // Par défaut ouvert
  });

  // Sauvegarder l'état dans localStorage
  useEffect(() => {
    localStorage.setItem('adminSidebarExpanded', String(isExpanded));
  }, [isExpanded]);

  // Mettre à jour la variable CSS pour la responsivité du contenu principal
  // S'exécute au montage et à chaque changement d'état
  useEffect(() => {
    // Sur mobile, toujours en mode collapsed (icônes seulement)
    const isMobile = window.innerWidth <= 768;
    const sidebarWidth = isMobile ? 72 : (isExpanded ? 240 : 72); // Largeur en pixels
    document.documentElement.style.setProperty('--sidebar-w', `${sidebarWidth}px`);
    
    // Écouter les changements de taille d'écran
    const handleResize = () => {
      const mobile = window.innerWidth <= 768;
      const width = mobile ? 72 : (isExpanded ? 240 : 72);
      document.documentElement.style.setProperty('--sidebar-w', `${width}px`);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Cleanup : restaurer la valeur par défaut si nécessaire
    return () => {
      window.removeEventListener('resize', handleResize);
      document.documentElement.style.setProperty('--sidebar-w', '72px');
    };
  }, [isExpanded]);

  const toggleSidebar = () => {
    setIsExpanded(!isExpanded);
  };

  const menuItems = [
    {
      icon: FaHome,
      label: 'Tableau de bord',
      to: `/dashboard/admin/${adminId}`,
      end: true,
    },
    {
      icon: FaCar,
      label: 'Réservations',
      to: `/dashboard/admin/${adminId}/reservations`,
    },
    {
      icon: FaUser,
      label: 'Utilisateurs',
      to: `/dashboard/admin/${adminId}/users`,
    },
    {
      icon: FaRobot,
      label: 'Shadow Mode MDI',
      to: `/dashboard/admin/${adminId}/shadow-mode`,
    },
    {
      icon: FaChartLine,
      label: 'Optimisation Optuna',
      to: `/dashboard/admin/${adminId}/optuna`,
    },
    {
      icon: FaFileInvoice,
      label: 'Factures',
      to: `/dashboard/admin/${adminId}/invoices`,
    },
    {
      icon: FaCog,
      label: 'Paramètres',
      to: `/dashboard/admin/${adminId}/settings`,
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

export default AdminSidebar;
