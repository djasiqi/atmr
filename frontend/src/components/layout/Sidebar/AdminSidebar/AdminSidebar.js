// src/components/layout/Sidebar/AdminSidebar/AdminSidebar.js
import React, { useState, useEffect, useMemo } from 'react';
import { NavLink, useLocation, useParams } from 'react-router-dom';
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
  FaServer,
} from 'react-icons/fa';
import { usePlatformCapabilities, PLATFORM_SEGMENTS } from '../../../../hooks/usePlatformCapabilities';
import styles from './AdminSidebar.module.css';

const AdminSidebar = () => {
  const { public_id } = useParams();
  const location = useLocation();
  const adminId = public_id ?? '';
  const { canAccess, isLoading: platformLoading } = usePlatformCapabilities();

  const showPlatformNav = useMemo(() => {
    if (platformLoading) return true;
    return PLATFORM_SEGMENTS.some((s) => canAccess(s));
  }, [platformLoading, canAccess]);

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

  const base = `/dashboard/admin/${adminId}`;

  const sections = [
    {
      title: 'Exploitation métier',
      items: [
        { icon: FaHome, label: 'Tableau de bord', to: base, end: true, isPlatform: false },
        { icon: FaCar, label: 'Réservations', to: `${base}/reservations`, isPlatform: false },
        { icon: FaFileInvoice, label: 'Factures', to: `${base}/invoices`, isPlatform: false },
        { icon: FaChartLine, label: 'Demandes demo', to: `${base}/demo-requests`, isPlatform: false },
      ],
    },
    {
      title: 'Administration applicative',
      items: [
        { icon: FaUser, label: 'Utilisateurs', to: `${base}/users`, isPlatform: false },
        { icon: FaCog, label: 'Paramètres', to: `${base}/settings`, isPlatform: false },
      ],
    },
    {
      title: 'Intelligence / optimisation',
      items: [
        { icon: FaRobot, label: 'Shadow Mode MDI', to: `${base}/shadow-mode`, isPlatform: false },
        { icon: FaChartLine, label: 'Optimisation Optuna', to: `${base}/optuna`, isPlatform: false },
      ],
    },
  ];

  if (showPlatformNav) {
    sections.push({
      title: 'Plateforme',
      items: [
        {
          icon: FaServer,
          label: 'Ops / Platform',
          to: `${base}/platform-ops/overview`,
          end: false,
          isPlatform: true,
        },
      ],
    });
  }

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

      <ul className={styles.menuList}>
        {sections.map((section) => (
          <React.Fragment key={section.title}>
            <li className={styles.sectionLabelItem} aria-hidden={!isExpanded}>
              <span className={styles.sectionLabel}>{isExpanded ? section.title : '·'}</span>
            </li>
            {section.items.map((item) => {
              const Icon = item.icon;
              return (
                <li key={item.to} className={styles.menuItem}>
                  <NavLink
                    to={item.to}
                    end={item.end ?? false}
                    className={({ isActive }) => {
                      const active = item.isPlatform
                        ? location.pathname.includes('/platform-ops')
                        : isActive;
                      return `${styles.menuLink} ${active ? styles.active : ''}`;
                    }}
                    title={!isExpanded ? item.label : undefined}
                  >
                    <Icon className={styles.icon} />
                    {isExpanded && <span className={styles.label}>{item.label}</span>}
                  </NavLink>
                </li>
              );
            })}
          </React.Fragment>
        ))}
      </ul>
    </nav>
  );
};

export default AdminSidebar;
