import React from 'react';
import { Link, useParams } from 'react-router-dom';
import { useAdminCapabilities } from '../../../hooks/useAdminCapabilities';
import { adminPaths } from '../routing/adminRoutePaths';
import styles from '../PlatformOps/AdminPlatformOps.module.css';

/**
 * Garde de route admin.* — analogue à PlatformSegmentGuard.
 * Le masquage nav ne suffit pas : l’accès URL direct doit être bloqué.
 *
 * @param {{ capability: string, children: import('react').ReactNode }} props
 */
export default function AdminCapabilityGuard({ capability, children }) {
  const { public_id: adminId } = useParams();
  const { can, isLoading, enforced } = useAdminCapabilities();

  if (isLoading) {
    return (
      <div className={styles.loading} role="status">
        Chargement des accès…
      </div>
    );
  }

  if (!can(capability)) {
    return (
      <div className={styles.tabPanel}>
        <div className={styles.errors} role="alert">
          <strong>Accès refusé</strong> — Capacité « {capability} » requise
          {enforced ? ' (enforcement actif).' : '.'}
        </div>
        <p className={styles.tabPanelHint}>
          <Link to={adminPaths.overview(adminId)}>Retour au tableau de bord admin</Link>
        </p>
      </div>
    );
  }

  return children;
}
