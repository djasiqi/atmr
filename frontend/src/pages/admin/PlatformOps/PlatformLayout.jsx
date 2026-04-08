import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { usePlatformCapabilities } from '../../../hooks/usePlatformCapabilities';
import styles from './AdminPlatformOps.module.css';
import adminShell from '../adminShell.module.css';

const SEGMENTS = [
  { id: 'overview', label: 'Vue globale' },
  { id: 'tenants', label: 'Tenants' },
  { id: 'runbooks', label: 'Runbooks' },
  { id: 'audit', label: 'Audit et replay' },
  { id: 'runtime', label: 'Runtime' },
  { id: 'reconciliation', label: 'Réconciliation' },
  { id: 'investigation', label: 'Investigation' },
];

/**
 * Sous-layout plateforme : sous-nav + Outlet uniquement.
 * Header + sidebar : parent AdminLayout.
 */
const PlatformLayout = () => {
  const { canAccess, isLoading } = usePlatformCapabilities();

  const visible = SEGMENTS.filter((s) => canAccess(s.id));

  return (
    <main className={adminShell.content}>
      <nav className={styles.tabBar} role="tablist" aria-label="Segments plateforme">
        {isLoading && (
          <span className={styles.subtle} role="status">
            Chargement des accès…
          </span>
        )}
        {!isLoading &&
          visible.map((s) => (
            <NavLink
              key={s.id}
              to={s.id}
              role="tab"
              className={({ isActive }) =>
                isActive ? styles.tabButtonActive : styles.tabButton
              }
            >
              {s.label}
            </NavLink>
          ))}
      </nav>

      <Outlet />
    </main>
  );
};

export default PlatformLayout;
