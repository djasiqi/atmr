import React from 'react';
import { Link, useParams } from 'react-router-dom';
import { adminPaths } from '../routing/adminRoutePaths';
import styles from './AdminPlatformOps.module.css';

/**
 * Accès refusé à un segment platform — pas de redirection automatique (évite les boucles).
 * Lien de sortie vers la vue d’ensemble admin (hors platform-ops).
 */
export default function PlatformAccessDenied() {
  const { public_id: adminId } = useParams();

  return (
    <div className={styles.tabPanel}>
      <div className={styles.errors} role="alert">
        <strong>Accès refusé</strong> — Vous n’avez pas les droits plateforme nécessaires pour ce
        segment. Contactez un administrateur ou utilisez un compte avec les bundles requis.
      </div>
      <p className={styles.tabPanelHint}>
        <Link to={adminPaths.overview(adminId)}>Retour au tableau de bord admin</Link>
      </p>
    </div>
  );
}
